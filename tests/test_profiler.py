from pathlib import Path

from gllm.runtime.profiler import TorchProfilerMixin, _env_flag


class _FakeProfiler:
    def __init__(self, trace_bytes: bytes):
        self.trace_bytes = trace_bytes
        self.key_averages_called = False

    def stop(self):
        pass

    def export_chrome_trace(self, path):
        Path(path).write_bytes(self.trace_bytes)

    def key_averages(self):
        self.key_averages_called = True
        raise AssertionError("large traces must not build key_averages")


def test_large_profiler_trace_skips_key_averages(tmp_path):
    runner = TorchProfilerMixin()
    runner.rank = 0
    runner.profile_start_ts = 123
    runner.profile_output_dir = str(tmp_path)
    runner.profile_session_dir = str(tmp_path)
    runner.profile_summary_max_trace_bytes = 8
    profiler = _FakeProfiler(b"0123456789abcdef")
    runner.profiler = profiler

    runner._stop_profiler()

    assert not profiler.key_averages_called
    assert (tmp_path / "trace_rank0_123.json.gz").is_file()
    assert not (tmp_path / "trace_rank0_123.json").exists()
    assert runner.profiler is None


def test_profiler_boolean_environment_flag(monkeypatch):
    monkeypatch.delenv("TEST_PROFILER_FLAG", raising=False)
    assert _env_flag("TEST_PROFILER_FLAG", True)
    assert not _env_flag("TEST_PROFILER_FLAG", False)

    for value in ("0", "false", "NO", "off"):
        monkeypatch.setenv("TEST_PROFILER_FLAG", value)
        assert not _env_flag("TEST_PROFILER_FLAG", True)

    monkeypatch.setenv("TEST_PROFILER_FLAG", "1")
    assert _env_flag("TEST_PROFILER_FLAG", False)
