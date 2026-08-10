"""Breakable piecewise CUDA graphs for dynamic mixed forwards.

Static model regions are captured into CUDA graph segments. Attention/GDN
modules are eager breaks which consume only the real token prefix and publish
their result into a persistent bucket-sized tensor. The next graph segment
therefore sees a stable address even though the dynamic core ran eagerly.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from logger import logger


class PiecewiseCapture:
    """A sequence of CUDA graphs interleaved with eager callables."""

    def __init__(self, segment_pools: list | None = None):
        # Each logical model segment owns one pool, reused across token
        # buckets. Different segments must not share a pool because an output
        # stays live across the intervening eager Attention/GDN call.
        self.segment_pools = segment_pools if segment_pools is not None else []
        self.segments: list[Callable[[], object]] = []
        self.num_graphs = 0
        self.num_eager_breaks = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._capturing = False

    def _begin(self):
        graph = torch.cuda.CUDAGraph()
        segment_index = self.num_graphs
        pool = (
            self.segment_pools[segment_index]
            if segment_index < len(self.segment_pools)
            else None
        )
        graph.capture_begin(pool=pool)
        self._graph = graph
        self._capturing = True

    def _end(self):
        if not self._capturing:
            return
        assert self._graph is not None
        self._graph.capture_end()
        segment_index = self.num_graphs
        if segment_index == len(self.segment_pools):
            self.segment_pools.append(self._graph.pool())
        # Raw stream capture records kernels but does not execute them. Replay
        # the just-closed segment once so the following eager break consumes
        # valid activations and the capture-time call itself is a real forward.
        self._graph.replay()
        self.segments.append(self._graph.replay)
        self.num_graphs += 1
        self._graph = None
        self._capturing = False

    def __enter__(self):
        self._begin()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end()

    def add_eager(self, fn: Callable[[], object]):
        self._end()
        result = fn()
        self.segments.append(fn)
        self.num_eager_breaks += 1
        self._begin()
        return result

    def replay(self):
        for segment in self.segments:
            segment()


class PiecewiseRuntime:
    """Thread-local layer hook active during warmup/capture."""

    _tls = threading.local()

    def __init__(
        self,
        bucket: int,
        num_tokens: int,
        *,
        warmup: bool = False,
        workspace_tokens: Optional[int] = None,
        workspace_token_sizes: Optional[list[int]] = None,
    ):
        self.bucket = bucket
        self.num_tokens = num_tokens
        self.warmup = warmup
        self.workspace_tokens = int(workspace_tokens or bucket)
        self.workspace_token_sizes = tuple(
            int(size) for size in (workspace_token_sizes or [self.workspace_tokens])
        )
        self.capture: Optional[PiecewiseCapture] = None

    @classmethod
    def current(cls) -> Optional["PiecewiseRuntime"]:
        return getattr(cls._tls, "current", None)

    @contextlib.contextmanager
    def activate(self):
        previous = getattr(self._tls, "current", None)
        if previous is not None:
            raise RuntimeError("nested piecewise CUDA graph runtime")
        self._tls.current = self
        try:
            yield self
        finally:
            self._tls.current = previous

    def dynamic_tensor(self, fn: Callable[[torch.Tensor], torch.Tensor], x):
        """Execute a dynamic layer over real rows and return a static buffer."""
        if self.warmup:
            # Warm only the graph-resident regions without advancing KV/GDN.
            return torch.zeros_like(x)
        if self.capture is None:
            return fn(x[: self.num_tokens])

        holder: dict[str, torch.Tensor] = {}

        def eager_call():
            n = self.num_tokens
            value = fn(x[:n])
            output = holder.get("output")
            if output is None:
                output = torch.empty(
                    (self.bucket, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                holder["output"] = output
            output[:n].copy_(value)
            if n < self.bucket:
                output[n:].zero_()
            return output

        return self.capture.add_eager(eager_call)


def piecewise_dynamic_tensor(fn: Callable[[torch.Tensor], torch.Tensor], x):
    """Layer-side hook; eager outside a piecewise capture."""
    runtime = PiecewiseRuntime.current()
    if runtime is None:
        return fn(x)
    return runtime.dynamic_tensor(fn, x)


@dataclass
class _PiecewiseGraph:
    bucket: int
    static_input: torch.Tensor
    runtime: PiecewiseRuntime
    capture: PiecewiseCapture
    output: torch.Tensor


class PiecewiseGraphRunner:
    """Breakable graphs dispatched with the model runner's fixed buckets.

    The bucket policy intentionally matches the existing full CUDA Graph
    runner: a fixed ``capture_sizes`` table and the smallest captured size
    which can contain the real batch. All buckets are captured during model
    initialization; request-time execution is replay-only, with eager fallback
    if initialization did not produce a graph for a bucket.
    """

    def __init__(
        self,
        model: Callable,
        capture_sizes: list[int],
    ):
        self.model = model
        self.capture_sizes = sorted(
            {int(size) for size in capture_sizes if int(size) > 0}
        )
        self.size_to_graph: dict[int, _PiecewiseGraph] = {}
        self.segment_pools: list = []

    @staticmethod
    def build_capture_sizes(max_capture_size: int) -> list[int]:
        """Build CUDA Graph token buckets up to the full token budget.

        Use exact tiny sizes, stride 8 below 256, then stride 16 through 512.
        Mixed MTP batches also contain prefill rows and can approach
        ``max_num_batched_tokens``; above 512 use four evenly spaced buckets per
        power-of-two interval. This covers the full runtime token budget without
        retaining hundreds of large graph/break buffers.
        """
        max_capture_size = int(max_capture_size)
        if max_capture_size <= 0:
            return []
        sizes = [size for size in (1, 2, 4) if size <= max_capture_size]
        if max_capture_size >= 8:
            sizes.extend(range(8, min(max_capture_size + 1, 256), 8))
        dense_max = min(max_capture_size, 512)
        if dense_max >= 256:
            sizes.extend(range(256, dense_max + 1, 16))
        interval_start = 512
        while max_capture_size > interval_start:
            interval_end = min(max_capture_size, interval_start * 2)
            step = interval_start // 4
            sizes.extend(
                range(interval_start + step, interval_end + 1, step)
            )
            if sizes[-1] != interval_end:
                sizes.append(interval_end)
            interval_start *= 2
        return sorted(set(sizes))

    def bucket_for(self, num_tokens: int) -> Optional[int]:
        return next(
            (size for size in self.capture_sizes if size >= num_tokens), None
        )

    def can_run(self, num_tokens: int) -> bool:
        return self.bucket_for(num_tokens) is not None

    @torch.inference_mode()
    def capture_bucket(self, input_data, hidden_states: torch.Tensor):
        n = int(hidden_states.shape[0])
        bucket = self.bucket_for(n)
        if bucket is None:
            return None
        if n != bucket:
            raise ValueError(
                f"initial piecewise capture needs an exact bucket, got {n}->{bucket}"
            )
        if bucket in self.size_to_graph:
            return self.size_to_graph[bucket].output[:n]

        static_input = torch.zeros(
            (bucket, hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        static_input[:n].copy_(hidden_states)

        # Shape/JIT warmup without touching attention or recurrent state.
        warmup = PiecewiseRuntime(
            bucket,
            n,
            warmup=True,
            workspace_tokens=self.capture_sizes[-1],
            workspace_token_sizes=self.capture_sizes,
        )
        with warmup.activate():
            self.model(input_data, static_input)
        torch.cuda.synchronize()
        static_input[:n].copy_(hidden_states)
        if n < bucket:
            static_input[n:].zero_()

        runtime = PiecewiseRuntime(
            bucket,
            n,
            workspace_tokens=self.capture_sizes[-1],
            workspace_token_sizes=self.capture_sizes,
        )
        capture = PiecewiseCapture(segment_pools=self.segment_pools)
        runtime.capture = capture
        with runtime.activate(), capture:
            output = self.model(input_data, static_input)

        graph = _PiecewiseGraph(bucket, static_input, runtime, capture, output)
        self.size_to_graph[bucket] = graph
        logger.info(
            "Captured piecewise CUDA graph: token_bucket=%d real_tokens=%d "
            "graph_segments=%d eager_breaks=%d",
            bucket,
            n,
            capture.num_graphs,
            capture.num_eager_breaks,
        )
        return output[:n]

    @torch.inference_mode()
    def run(self, input_data, hidden_states: torch.Tensor):
        n = int(hidden_states.shape[0])
        bucket = self.bucket_for(n)
        if bucket is None:
            return None
        graph = self.size_to_graph.get(bucket)
        if graph is None:
            # All configured buckets are captured during model initialization.
            # A miss is a safe eager fallback, never an online capture.
            return None
        graph.runtime.num_tokens = n
        graph.static_input[:n].copy_(hidden_states)
        if n < bucket:
            graph.static_input[n:].zero_()
        graph.capture.replay()
        return graph.output[:n]
