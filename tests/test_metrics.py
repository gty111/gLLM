"""Unit tests for gllm.observability.metrics (no GPU / no torch required).

The metrics module has no heavy dependencies, so we load it straight from its
file path. Importing it as ``gllm.observability.metrics`` would trigger
``gllm/__init__.py`` -> ``gllm.engine.llm`` -> ``torch`` (absent in CI here).
"""

import importlib.util
import time
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "gllm" / "observability" / "metrics.py"
)


@pytest.fixture()
def m():
    """Load the metrics module standalone (bypasses the torch-import chain)."""
    spec = importlib.util.spec_from_file_location("gllm_obs_metrics", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frontend_lifecycle(m):
    fm = m.FrontendMetrics(model_name="test-model", enabled=True)

    # begin
    fm.begin_request("req1", "chat_completion", streaming=True, prompt_tokens=128)
    time.sleep(0.01)
    fm.first_token("req1")
    fm.token("req1", 1)
    fm.token("req1", 1)
    fm.finish_request("req1", finish_reason="stop")

    body = fm.render().decode()
    assert "gllm_requests_total" in body
    assert "gllm_prompt_tokens_total" in body
    assert "gllm_generation_tokens_total" in body
    assert "gllm_time_to_first_token_seconds" in body
    assert "gllm_request_e2e_latency_seconds" in body
    assert "gllm_generation_tokens_per_request" in body
    assert "gllm_token_throughput_per_second" in body


def test_frontend_disabled(m):
    fm = m.FrontendMetrics(model_name="test", enabled=False)
    fm.begin_request("r", "chat", streaming=False, prompt_tokens=10)
    fm.finish_request("r")
    body = fm.render().decode()
    assert "disabled" in body.lower()


def test_engine_stats_snapshot(m):
    es = m.EngineStats()
    es.record_iteration(256, 4, 8)
    es.record_kv(total=1000, free=750, utilization_pct=25.0)
    es.record_queues(running=8, waiting=2)
    es.record_gpu_memory()

    snap = es.snapshot()
    # deltas are consumed
    assert snap["iterations_delta"] == 1
    assert snap["iteration_tokens_delta"] == 256
    # second snapshot has zeroed deltas
    snap2 = es.snapshot()
    assert snap2["iterations_delta"] == 0
    # gauges persist
    assert snap["kv_pages_total"] == 1000
    assert snap["kv_pages_free"] == 750


def test_engine_stats_prefix_cache(m):
    es = m.EngineStats()
    # Scheduler passes the memory manager's running totals (latest wins).
    es.record_prefix_cache(hit_pages=100, alloc_pages=40)
    snap = es.snapshot()
    assert snap["prefix_cache_hit_pages_total"] == 100
    assert snap["prefix_cache_alloc_pages_total"] == 40
    es.record_prefix_cache(hit_pages=150, alloc_pages=50)
    snap2 = es.snapshot()
    assert snap2["prefix_cache_hit_pages_total"] == 150
    assert snap2["prefix_cache_alloc_pages_total"] == 50


def test_update_from_engine_stats(m):
    fm = m.FrontendMetrics(model_name="test", enabled=True)
    stats = {
        "iterations_delta": 3,
        "iteration_tokens_delta": 512,
        "num_steps_delta": 3,
        "num_requests_running": 16,
        "num_requests_waiting": 4,
        "kv_pages_total": 2000,
        "kv_pages_free": 500,
        "kv_cache_utilization_pct": 75.0,
        "batch_prefill_seqs": 2,
        "batch_decode_seqs": 14,
        "batch_tokens": 512,
        "prefix_cache_hit_pages_total": 100,
        "prefix_cache_alloc_pages_total": 40,
    }
    fm.update_from_engine_stats(stats)
    body = fm.render().decode()
    assert "gllm_kv_cache_pages_total" in body
    assert "gllm_kv_cache_usage_pct" in body
    assert "gllm_batch_decode_seqs" in body
    assert "gllm_prefix_cache_hit_rate" in body
    # prefix hit rate should be 100/140 ≈ 0.714
    assert "0.71" in body or "0.714" in body

def test_metrics_enabled_default_off(m, monkeypatch):
    """Metrics are OFF by default; ON only with the --enable-metrics flag.

    There is no environment-variable switch (confirmed by checking no env read).
    """
    import unittest.mock as mk
    with mk.patch.object(m, "_HAVE_PROMETHEUS", True):
        assert m.metrics_enabled() is False
        assert m.metrics_enabled(False) is False
        assert m.metrics_enabled(True) is True
    with mk.patch.object(m, "_HAVE_PROMETHEUS", False):
        # Library missing forces OFF even if the flag is set.
        assert m.metrics_enabled(True) is False


def test_enable_metrics_plumbing(m):
    """LLM -> ModelRunner -> Scheduler flag chain is wired end to end.

    We can't import the full engine here (needs torch), so verify the wiring
    by reading the source of the three seams.
    """
    root = Path(__file__).resolve().parents[1]

    # 1. LLM accepts enable_metrics and forwards it to model_runner + singleton.
    llm_src = (root / "gllm" / "engine" / "llm.py").read_text()
    assert "enable_metrics=False" in llm_src
    assert "enable_metrics=enable_metrics" in llm_src
    assert "init_frontend_metrics(enable_metrics)" in llm_src

    # 2. ModelRunner stores it.
    mr_src = (root / "gllm" / "runtime" / "model_runner.py").read_text()
    assert "enable_metrics: bool = False" in mr_src
    assert "self.enable_metrics = bool(enable_metrics)" in mr_src

    # 3. Scheduler reads it to decide on EngineStats.
    sch_src = (root / "gllm" / "scheduling" / "scheduler.py").read_text()
    assert 'getattr(model_runner, "enable_metrics", False)' in sch_src

    # 4. CLI flag is plumbed through engine_kwargs.
    ca_src = (root / "gllm" / "entrypoints" / "cli_args.py").read_text()
    assert '"enable_metrics": args.enable_metrics' in ca_src
    assert '"--enable-metrics"' in ca_src
