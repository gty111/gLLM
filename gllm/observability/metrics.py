"""Prometheus observability for gLLM.

Metrics are **off by default** (SGLang-style). Opt in with the
``--enable-metrics`` flag. When off -- or when ``prometheus_client`` is not
installed -- every helper degrades to a cheap no-op so the serving path pays
nothing.

Two collection layers (mirrors vLLM / SGLang):

* **Frontend** (main process) — :class:`FrontendMetrics` records request
  lifecycle events (TTFT / ITL / E2E latencies, token counts, request counts)
  from the OpenAI-compatible entrypoint and engine queue lengths. It renders
  from ``GET /metrics`` on the API server.

* **Worker** (GPU child processes) — each worker accumulates engine-internal
  stats (KV cache pages, batch composition, iteration tokens, GPU memory) and
  ships them back on the existing ``IPCPackage.stats`` field (ZMQ output
  channel, no extra ports). The frontend folds each package's stats into the
  same :class:`FrontendMetrics` registry.

Metric names use the ``gllm_`` prefix.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover - optional dependency
    _HAVE_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _Noop:
        """No-op stand-in matching the metric API surface used below."""

        def __call__(self, *a, **kw):
            return self

        def __getattr__(self, name):
            return lambda *a, **kw: None

    class _NoopRegistry(CollectorRegistry):  # type: ignore[misc]
        pass

    Counter = Gauge = Histogram = _Noop()
    CollectorRegistry = _NoopRegistry

    def generate_latest(registry=None):  # type: ignore[misc]
        return b"# prometheus_client not installed\n"


def metrics_enabled(cli_enabled: bool = False) -> bool:
    """Whether Prometheus metrics are active.

    Metrics are **off by default** (SGLang-style). They turn on only when the
    ``prometheus_client`` library is present *and* the caller has explicitly
    opted in via the ``--enable-metrics`` flag (``cli_enabled``). There is no
    environment-variable switch.
    """
    return _HAVE_PROMETHEUS and bool(cli_enabled)


# ---------------------------------------------------------------------------
# Worker-side stats accumulation
# ---------------------------------------------------------------------------


class EngineStats:
    """Per-worker rolling statistics, flushed into ``IPCPackage.stats``.

    Lives on the worker's ``Scheduler`` (see ``gllm.scheduling.scheduler``).
    Counters (iterations, tokens, steps) are *deltas* since the last flush so
    the frontend can increment its Prometheus counters directly. Gauges (page
    counts, utilization) are absolute snapshots.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.iterations = 0
        self.num_steps = 0
        self.iteration_tokens = 0
        # Per-phase forward-token accounting (scheduled, per step).
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.preempted_seqs = 0
        self.finished_seqs = 0
        # Absolute gauges, latest snapshot wins.
        self.num_requests_running: Optional[int] = None
        self.num_requests_waiting: Optional[int] = None
        self.kv_pages_total: Optional[int] = None
        self.kv_pages_free: Optional[int] = None
        self.kv_cache_utilization_pct: Optional[float] = None
        self.batch_prefill_seqs: Optional[int] = None
        self.batch_decode_seqs: Optional[int] = None
        self.batch_tokens: Optional[int] = None
        self.gpu_memory_allocated_bytes: Optional[float] = None
        self.gpu_memory_reserved_bytes: Optional[float] = None
        # Prefix-cache hit accounting (pages), when available.
        self.prefix_cache_hit_pages: Optional[int] = None
        self.prefix_cache_alloc_pages: Optional[int] = None

    # -- recording -----------------------------------------------------------

    def record_iteration(self, num_tokens: int, num_prefill: int,
                         num_decode: int, preempted: int = 0,
                         finished: int = 0):
        with self._lock:
            self.iterations += 1
            self.iteration_tokens += int(num_tokens)
            self.preempted_seqs += int(preempted)
            self.finished_seqs += int(finished)
            self.batch_prefill_seqs = int(num_prefill)
            self.batch_decode_seqs = int(num_decode)
            self.batch_tokens = int(num_tokens)

    def record_phase_tokens(self, prefill_tokens: int, decode_tokens: int):
        """Accumulate per-phase scheduled token counts for one step.

        Called from the scheduler right after a batch is assembled, using
        each row's ``to_compute_token_num`` and ``computed_prompt`` split,
        so the totals reflect actual forward compute per phase.
        """
        with self._lock:
            self.prefill_tokens += int(prefill_tokens)
            self.decode_tokens += int(decode_tokens)

    def sample_package(self, package):
        """Derive iteration-level counters from a finished ``IPCPackage``.

        This is the overlap-scheduling friendly hook: every finalised output
        package (plain / MTP / relay) passes through it once. ``finished`` is
        the number of seqs freed by this package; ``num_tokens`` is the number
        of acted rows (one per emitted decode row).
        """
        acted = getattr(package, "act_schedule_ids", None) or []
        free = getattr(package, "free_ids", None) or []
        self.record_iteration(
            num_tokens=len(acted),
            num_prefill=0,   # per-row prefill/decode split is not recoverable
            num_decode=len(acted),
            finished=len(free),
        )

    def record_step(self):
        with self._lock:
            self.num_steps += 1

    def record_kv(self, total: Optional[int], free: Optional[int],
                  utilization_pct: Optional[float] = None):
        with self._lock:
            if total is not None:
                self.kv_pages_total = int(total)
            if free is not None:
                self.kv_pages_free = int(free)
            if utilization_pct is not None:
                self.kv_cache_utilization_pct = float(utilization_pct)

    def record_queues(self, running: int, waiting: int):
        with self._lock:
            self.num_requests_running = int(running)
            self.num_requests_waiting = int(waiting)

    def record_prefix_cache(self, hit_pages: int, alloc_pages: int):
        """Record prefix-cache page counters (running totals from the worker).

        The scheduler passes the memory manager's cumulative counters; we keep
        the latest snapshot and ``snapshot`` emits them as-is.  The frontend
        derives the hit *rate* from the two numbers (hits / (hits + allocs)),
        so absolute monotonic values are what we need.
        """
        with self._lock:
            self.prefix_cache_hit_pages = int(hit_pages)
            self.prefix_cache_alloc_pages = int(alloc_pages)

    def record_gpu_memory(self):
        """Best-effort CUDA memory gauges (CPU-side calls, cheap enough)."""
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu_memory_allocated_bytes = float(
                    torch.cuda.memory_allocated()
                )
                self.gpu_memory_reserved_bytes = float(
                    torch.cuda.memory_reserved()
                )
        except Exception:
            pass

    # -- flush ---------------------------------------------------------------

    def snapshot(self) -> dict:
        """Delta counters + absolute gauges, ready for ``IPCPackage.stats``."""
        with self._lock:
            stats = {
                "iterations_delta": self.iterations,
                "num_steps_delta": self.num_steps,
                "iteration_tokens_delta": self.iteration_tokens,
                "prefill_tokens_delta": self.prefill_tokens,
                "decode_tokens_delta": self.decode_tokens,
                "preempted_seqs_delta": self.preempted_seqs,
                "finished_seqs_delta": self.finished_seqs,
            }
            self.iterations = 0
            self.num_steps = 0
            self.iteration_tokens = 0
            self.prefill_tokens = 0
            self.decode_tokens = 0
            self.preempted_seqs = 0
            self.finished_seqs = 0
            if self.prefix_cache_hit_pages:
                stats["prefix_cache_hit_pages_total"] = self.prefix_cache_hit_pages
            if self.prefix_cache_alloc_pages:
                stats["prefix_cache_alloc_pages_total"] = self.prefix_cache_alloc_pages
            for name in (
                "num_requests_running",
                "num_requests_waiting",
                "kv_pages_total",
                "kv_pages_free",
                "kv_cache_utilization_pct",
                "batch_prefill_seqs",
                "batch_decode_seqs",
                "batch_tokens",
                "gpu_memory_allocated_bytes",
                "gpu_memory_reserved_bytes",
            ):
                value = getattr(self, name, None)
                if value is not None:
                    stats[name] = value
            return stats


# ---------------------------------------------------------------------------
# Frontend metrics registry
# ---------------------------------------------------------------------------


_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0,
    1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0,
)
# Per-request token-length buckets. Dense below 1k tokens (typical chat
# traffic) and in integer-K increments above, so dashboard heatmaps can label
# the axis directly in K tokens (1k, 2k, 4k, ..., 128k).
_TOKEN_BUCKETS = (
    256, 512, 1024,
    2048, 3072, 4096, 6144, 8192,
    12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072,
)
_BYTES_BUCKETS = (
    1 << 28, 1 << 30, 1 << 32, 1 << 34, 1 << 36, 1 << 38, 1 << 40,
)


class FrontendMetrics:
    """Prometheus registry + lifecycle helpers for the serving frontend.

    One instance per engine owner (the api-server process). Recording happens
    on the event-loop thread and on the engine-IO executor threads, so every
    mutation of *active request state* takes a single lock; the Prometheus
    primitives themselves are already thread-safe.
    """

    def __init__(self, model_name: str = "", enabled: bool = True):
        self.enabled = enabled and _HAVE_PROMETHEUS
        self.model_name = model_name or "unknown"
        self._lock = threading.Lock()
        self._requests: Dict[str, dict] = {}  # request key -> state
        self._started_at = time.monotonic()
        # Sliding-window token throughput (EMA of per-second rates).
        self._tpot_alpha = 0.3
        self._throughput_ema: Optional[float] = None
        self._prev_window_tokens = 0
        self._prev_window_time = time.monotonic()
        self._prefix_hit_pages = 0
        self._prefix_alloc_pages = 0

        if not self.enabled:
            return

        self.registry = CollectorRegistry(auto_describe=True)
        reg = {"registry": self.registry}
        self._g = lambda name, doc, labels=(): Gauge(name, doc, labels, **reg)
        self._c = lambda name, doc, labels=(): Counter(name, doc, labels, **reg)
        self._h = lambda name, doc, buckets, labels=(): (
            Histogram(name, doc, labels, buckets=buckets, **reg)
        )

        m = self.model_name
        lbl_model = ("model",)
        lbl_req = ("model", "method", "streaming")
        lbl_end = ("model", "finish_reason")

        self.requests_total = self._c(
            "gllm_requests_total",
            "Total inference requests received.",
            lbl_req,
        )
        self.request_success_total = self._c(
            "gllm_request_success_total",
            "Requests completed without server error.",
            lbl_end,
        )
        self.request_e2e_latency_seconds = self._h(
            "gllm_request_e2e_latency_seconds",
            "End-to-end request latency (received -> fully served).",
            _LATENCY_BUCKETS,
        )
        self.time_to_first_token_seconds = self._h(
            "gllm_time_to_first_token_seconds",
            "Time to first generated token.",
            _LATENCY_BUCKETS,
        )
        self.time_per_output_token_seconds = self._h(
            "gllm_time_per_output_token_seconds",
            "Inter-output-token latency (mean over each streamed request).",
            _LATENCY_BUCKETS,
        )
        self.prompt_tokens_total = self._c(
            "gllm_prompt_tokens_total", "Prompt tokens processed.", lbl_model
        )
        self.generation_tokens_total = self._c(
            "gllm_generation_tokens_total", "Output tokens generated.", lbl_model
        )
        self.prompt_tokens_per_request = self._h(
            "gllm_prompt_tokens_per_request",
            "Prompt token length per request.",
            _TOKEN_BUCKETS,
        )
        self.generation_tokens_per_request = self._h(
            "gllm_generation_tokens_per_request",
            "Generated token length per request.",
            _TOKEN_BUCKETS,
        )
        self.num_requests_running = self._g(
            "gllm_num_requests_running",
            "Sequences currently executing in the engine.",
        )
        self.num_requests_waiting = self._g(
            "gllm_num_requests_waiting",
            "New sequences waiting to be admitted (frontend queue).",
        )
        self.queuing_latency_seconds = self._h(
            "gllm_queuing_latency_seconds",
            "Time spent waiting for engine admission before first token.",
            _LATENCY_BUCKETS,
        )
        self.token_throughput_per_second = self._g(
            "gllm_token_throughput_per_second",
            "Output tokens per second (exponentially weighted).",
        )
        self.iteration_tokens_total = self._c(
            "gllm_iteration_tokens_total",
            "Tokens computed per engine iteration (prefill+decode rows).",
        )
        self.prefill_tokens_total = self._c(
            "gllm_prefill_tokens_total",
            "Prompt tokens scheduled for forward compute (per engine step).",
        )
        self.decode_tokens_total = self._c(
            "gllm_decode_tokens_total",
            "Decode query tokens scheduled for forward compute (per engine step).",
        )
        self.num_steps_total = self._c(
            "gllm_num_steps_total", "Engine iterations executed.",
        )
        self.finished_requests_total = self._c(
            "gllm_finished_requests_total",
            "Requests that reached a terminal state in the engine.",
        )
        self.num_preemptions_total = self._c(
            "gllm_num_preemptions_total", "Sequences preempted by the scheduler.",
        )
        # --- worker-reported gauges ---
        self.kv_cache_pages_total = self._g(
            "gllm_kv_cache_pages_total", "KV cache arena pages (worker-reported).",
        )
        self.kv_cache_pages_free = self._g(
            "gllm_kv_cache_pages_free", "Free KV cache pages (worker-reported).",
        )
        self.kv_cache_usage_pct = self._g(
            "gllm_kv_cache_usage_pct",
            "KV cache arena utilization percent (worker-reported).",
        )
        self.batch_prefill_seqs = self._g(
            "gllm_batch_prefill_seqs",
            "Prefill rows in the most recent worker batch.",
        )
        self.batch_decode_seqs = self._g(
            "gllm_batch_decode_seqs",
            "Decode rows in the most recent worker batch.",
        )
        self.batch_tokens = self._g(
            "gllm_batch_tokens", "Total tokens in the most recent worker batch.",
        )
        self.gpu_memory_allocated_bytes = self._h(
            "gllm_gpu_memory_allocated_bytes",
            "CUDA memory allocated by torch (worker-reported).",
            _BYTES_BUCKETS,
        )
        self.gpu_memory_reserved_bytes = self._h(
            "gllm_gpu_memory_reserved_bytes",
            "CUDA memory reserved by torch (worker-reported).",
            _BYTES_BUCKETS,
        )
        self.prefix_cache_hit_pages_total = self._c(
            "gllm_prefix_cache_hit_pages_total",
            "Cumulative prefix-cache page hits (worker-reported).",
        )
        self.prefix_cache_alloc_pages_total = self._c(
            "gllm_prefix_cache_alloc_pages_total",
            "Cumulative prefix-cache page allocations (worker-reported).",
        )
        self.prefix_cache_hit_rate = self._g(
            "gllm_prefix_cache_hit_rate",
            "Prefix-cache hit rate (hits / (hits + allocs)).",
        )
        self.uptime_seconds = self._g(
            "gllm_uptime_seconds", "Engine process uptime.",
        )
        self.last_engine_report_seconds = self._g(
            "gllm_last_engine_report_timestamp_seconds",
            "Unix time of the last worker stats report.",
        )

    # -- request lifecycle ----------------------------------------------------

    def begin_request(self, key: str, method: str, *, streaming: bool,
                      prompt_tokens: int = 0):
        """Start tracking one HTTP inference request."""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            self._requests[key] = {
                "method": method,
                "streaming": streaming,
                "start": now,
                "first_token_at": None,
                "admitted_at": None,
                "tokens": 0,
                "prompt_tokens": prompt_tokens,
                "last_chunk_at": now,
            }
        self.requests_total.labels(
            model=self.model_name,
            method=method,
            streaming="true" if streaming else "false",
        ).inc()
        if prompt_tokens:
            self.prompt_tokens_total.labels(model=self.model_name).inc(
                prompt_tokens
            )
            self.prompt_tokens_per_request.observe(prompt_tokens)

    def first_token(self, key: str):
        """Mark first generated token (sets TTFT / queuing latency)."""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            state = self._requests.get(key)
            if state is None or state["first_token_at"] is not None:
                return
            state["first_token_at"] = now
            ttft = now - state["start"]
        self.time_to_first_token_seconds.observe(ttft)

    def token(self, key: str, n: int = 1):
        """Account for generated tokens arriving on the wire."""
        if not self.enabled:
            return
        with self._lock:
            state = self._requests.get(key)
            if state is None:
                return
            state["tokens"] += n
            if state["first_token_at"] is None:
                state["first_token_at"] = time.monotonic()
        self.generation_tokens_total.labels(model=self.model_name).inc(n)

    def finish_request(self, key: str, *, error: bool = False,
                       finish_reason: Optional[str] = None):
        """Finish tracking; observes E2E + per-token histograms."""
        if not self.enabled:
            return
        with self._lock:
            state = self._requests.pop(key, None)
        if state is None:
            return
        now = time.monotonic()
        e2e = now - state["start"]
        self.request_e2e_latency_seconds.observe(e2e)
        if state["tokens"] > 1 and state["first_token_at"] is not None:
            # Mean inter-token interval over the streaming window.
            itl = (now - state["first_token_at"]) / (state["tokens"] - 1)
            self.time_per_output_token_seconds.observe(itl)
        self.generation_tokens_per_request.observe(state["tokens"])
        if not error:
            reason = finish_reason or "stop"
            self.request_success_total.labels(
                model=self.model_name, finish_reason=reason
            ).inc()
        self._note_throughput(state["tokens"])

    def _note_throughput(self, tokens: int):
        now = time.monotonic()
        dt = now - self._prev_window_time
        if dt <= 0:
            return
        rate = tokens / dt
        if self._throughput_ema is None:
            self._throughput_ema = rate
        else:
            self._throughput_ema = (
                self._tpot_alpha * rate + (1 - self._tpot_alpha) * self._throughput_ema
            )
        self.token_throughput_per_second.set(self._throughput_ema)
        self._prev_window_time = now

    # -- engine-side gauges ----------------------------------------------------

    def set_queue_gauges(self, running: int, waiting: int):
        if not self.enabled:
            return
        self.num_requests_running.set(running)
        self.num_requests_waiting.set(waiting)

    def update_from_engine_stats(self, stats: dict):
        """Fold one worker ``IPCPackage.stats`` payload into the registry."""
        if not self.enabled or not stats:
            return
        model = self.model_name
        if "iterations_delta" in stats:
            self.num_steps_total.inc(stats["iterations_delta"])
        if "iteration_tokens_delta" in stats:
            self.iteration_tokens_total.inc(stats["iteration_tokens_delta"])
        if "prefill_tokens_delta" in stats:
            self.prefill_tokens_total.inc(stats["prefill_tokens_delta"])
        if "decode_tokens_delta" in stats:
            self.decode_tokens_total.inc(stats["decode_tokens_delta"])
        if "finished_seqs_delta" in stats:
            self.finished_requests_total.inc(stats["finished_seqs_delta"])
        if "preempted_seqs_delta" in stats:
            self.num_preemptions_total.inc(stats["preempted_seqs_delta"])
        if "num_requests_running" in stats:
            self.num_requests_running.set(stats["num_requests_running"])
        if "num_requests_waiting" in stats:
            self.num_requests_waiting.set(stats["num_requests_waiting"])
        if "kv_pages_total" in stats:
            self.kv_cache_pages_total.set(stats["kv_pages_total"])
        if "kv_pages_free" in stats:
            self.kv_cache_pages_free.set(stats["kv_pages_free"])
        if "kv_cache_utilization_pct" in stats:
            self.kv_cache_usage_pct.set(stats["kv_cache_utilization_pct"])
        if "batch_prefill_seqs" in stats:
            self.batch_prefill_seqs.set(stats["batch_prefill_seqs"])
        if "batch_decode_seqs" in stats:
            self.batch_decode_seqs.set(stats["batch_decode_seqs"])
        if "batch_tokens" in stats:
            self.batch_tokens.set(stats["batch_tokens"])
        if "gpu_memory_allocated_bytes" in stats:
            self.gpu_memory_allocated_bytes.observe(
                stats["gpu_memory_allocated_bytes"]
            )
        if "gpu_memory_reserved_bytes" in stats:
            self.gpu_memory_reserved_bytes.observe(
                stats["gpu_memory_reserved_bytes"]
            )
        if "prefix_cache_hit_pages_total" in stats:
            self.prefix_cache_hit_pages_total.inc(
                stats["prefix_cache_hit_pages_total"]
            )
        if "prefix_cache_alloc_pages_total" in stats:
            self.prefix_cache_alloc_pages_total.inc(
                stats["prefix_cache_alloc_pages_total"]
            )
        hit = self.prefix_cache_hit_pages_total._value.get()  # type: ignore[attr-defined]
        alloc = self.prefix_cache_alloc_pages_total._value.get()  # type: ignore[attr-defined]
        denom = hit + alloc
        if denom > 0:
            self.prefix_cache_hit_rate.set(hit / denom)
        self.last_engine_report_seconds.set(time.time())

    # -- exposition -------------------------------------------------------------

    def render(self) -> bytes:
        if not self.enabled:
            return b"# gllm metrics disabled\n"
        self.uptime_seconds.set(time.monotonic() - self._started_at)
        with self._lock:
            active = len(self._requests)
        return generate_latest(self.registry)

    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# Module-level singleton (frontend process)
# ---------------------------------------------------------------------------

_FRONTEND: Optional[FrontendMetrics] = None
_LOCK = threading.Lock()


def get_frontend_metrics() -> FrontendMetrics:
    """Lazily-create process singleton for the api-server process."""
    global _FRONTEND
    with _LOCK:
        if _FRONTEND is None:
            _FRONTEND = FrontendMetrics(enabled=metrics_enabled(False))
        return _FRONTEND


def set_frontend_model_name(model_name: str) -> None:
    """Bind the loaded model identity to the frontend singleton.

    Called by the serving entrypoint after the engine knows its model path, so
    request counters carry a meaningful ``model=`` label instead of "unknown".
    """
    global _FRONTEND
    if not model_name:
        return
    with _LOCK:
        if _FRONTEND is None:
            return
        _FRONTEND.model_name = model_name


def init_frontend_metrics(enable_metrics: bool = False) -> FrontendMetrics:
    """Create the frontend singleton with the ``--enable-metrics`` choice.

    Called once at startup (before workers spawn) with the CLI flag so the
    frontend registry reflects it. The singleton caches its ``enabled`` state
    on first creation, so this must run before the first lazy
    :func:`get_frontend_metrics` call that would otherwise default it off.
    """
    global _FRONTEND
    with _LOCK:
        if _FRONTEND is None:
            _FRONTEND = FrontendMetrics(enabled=metrics_enabled(enable_metrics))
        return _FRONTEND
