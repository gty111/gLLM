import argparse
import inspect
from types import SimpleNamespace

import torch

import gllm.runtime.model_runner as model_runner_module
from gllm.engine.llm import LLM
from gllm.entrypoints.cli_args import add_runtime_args
from gllm.runtime.model_runner import ModelRunner
from gllm.runtime.piecewise_cuda_graph import (
    PiecewiseCapture,
    PiecewiseGraphRunner,
    PiecewiseRuntime,
)


def test_capture_sizes_preserve_dense_padding_policy():
    sizes = PiecewiseGraphRunner.build_capture_sizes(8192)
    assert len(sizes) == 67
    assert sizes[:6] == [1, 2, 4, 8, 16, 24]
    assert sizes[-5:] == [4096, 5120, 6144, 7168, 8192]

    non_power_of_two = PiecewiseGraphRunner.build_capture_sizes(6000)
    assert non_power_of_two[-1] == 6000
    assert PiecewiseGraphRunner.build_capture_sizes(0) == []


def test_piecewise_cuda_graph_defaults_on_for_cli_and_library():
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    assert parser.parse_args([]).piecewise_cuda_graph == "on"
    assert (
        inspect.signature(LLM.__init__)
        .parameters["piecewise_cuda_graph"]
        .default
        is True
    )


def test_capture_registers_eager_break_without_executing_it(monkeypatch):
    events = []

    class _Graph:
        def capture_begin(self, pool=None):
            events.append(("begin", pool))

        def capture_end(self):
            events.append(("end", None))

        def replay(self):
            events.append(("graph", None))

        def pool(self):
            return "pool"

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    capture = PiecewiseCapture()
    capture._begin()
    eager_calls = []
    placeholder = torch.zeros(2, 4)

    result = capture.add_eager(
        lambda: eager_calls.append("eager"),
        capture_result=placeholder,
    )

    assert result is placeholder
    assert eager_calls == []
    assert capture.num_eager_breaks == 1
    capture._end()
    capture.replay()
    assert eager_calls == ["eager"]


def test_piecewise_segments_and_buckets_share_one_externalized_pool(monkeypatch):
    capture_pools = []
    next_pool = 0

    class _Graph:
        def capture_begin(self, pool=None):
            capture_pools.append(pool)

        def capture_end(self):
            pass

        def replay(self):
            pass

        def pool(self):
            nonlocal next_pool
            value = f"segment-pool-{next_pool}"
            next_pool += 1
            return value

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    shared_pool = []
    first_bucket = PiecewiseCapture(graph_pool=shared_pool)
    first_bucket._begin()
    first_bucket._end()
    first_bucket._begin()
    first_bucket._end()
    second_bucket = PiecewiseCapture(graph_pool=shared_pool)
    second_bucket._begin()
    second_bucket._end()

    assert shared_pool == ["segment-pool-0"]
    assert capture_pools == [None, "segment-pool-0", "segment-pool-0"]


def test_break_buffers_are_shared_across_buckets_and_use_real_prefix():
    shared = []
    calls = []

    class _Capture:
        def __init__(self):
            self.eager = None

        def add_eager(self, fn, *, capture_result=None):
            self.eager = fn
            return capture_result

    x_large = torch.arange(32, dtype=torch.float32).view(8, 4)
    runtime_large = PiecewiseRuntime(
        bucket=8,
        num_tokens=3,
        workspace_tokens=8,
        break_buffers=shared,
    )
    capture_large = _Capture()
    runtime_large.capture = capture_large
    output_large = runtime_large.dynamic_tensor(
        lambda value: calls.append(value.shape[0]) or value.add(10),
        x_large,
    )

    assert calls == []
    assert output_large.shape == (8, 4)
    assert len(shared) == 1
    capture_large.eager()
    assert calls == [3]
    torch.testing.assert_close(output_large[:3], x_large[:3] + 10)
    assert torch.count_nonzero(output_large[3:]) == 0

    x_small = torch.ones(4, 4)
    runtime_small = PiecewiseRuntime(
        bucket=4,
        num_tokens=2,
        workspace_tokens=8,
        break_buffers=shared,
    )
    capture_small = _Capture()
    runtime_small.capture = capture_small
    output_small = runtime_small.dynamic_tensor(lambda value: value * 2, x_small)
    assert output_small.data_ptr() == output_large.data_ptr()
    assert len(shared) == 1


def test_many_eager_boundaries_use_only_two_ping_pong_buffers():
    shared = []

    class _Capture:
        def add_eager(self, fn, *, capture_result=None):
            return capture_result

    runtime = PiecewiseRuntime(
        bucket=8,
        num_tokens=8,
        workspace_tokens=8,
        break_buffers=shared,
    )
    runtime.capture = _Capture()
    outputs = [
        runtime.dynamic_tensor(lambda value: value, torch.zeros(8, 4))
        for _ in range(7)
    ]
    assert len(shared) == 2
    assert outputs[0].data_ptr() == outputs[2].data_ptr()
    assert outputs[1].data_ptr() == outputs[3].data_ptr()


def test_passthrough_tensor_is_bridged_out_of_graph_owned_storage():
    passthrough_buffers = []

    class _Capture:
        def add_eager(self, fn, *, capture_result=None):
            return capture_result

    residual = torch.arange(32, dtype=torch.float32).view(8, 4)
    runtime = PiecewiseRuntime(
        bucket=8,
        num_tokens=3,
        workspace_tokens=8,
        passthrough_buffers=passthrough_buffers,
    )
    runtime.capture = _Capture()
    output, bridged_residual = runtime.dynamic_tensor(
        lambda value: value,
        torch.zeros(8, 4),
        residual,
    )

    assert output.shape == residual.shape
    assert len(passthrough_buffers) == 1
    assert bridged_residual.data_ptr() != residual.data_ptr()
    torch.testing.assert_close(bridged_residual, residual)


class _FakePiecewiseRunner:
    def __init__(self, max_tokens=8):
        self.max_tokens = max_tokens
        self.calls = 0

    def can_run(self, num_tokens):
        return num_tokens <= self.max_tokens

    def run(self, input_data, hidden_states):
        self.calls += 1
        return hidden_states + 1


class _FakeModel:
    def embed_input_ids(self, token_ids):
        return token_ids.float().unsqueeze(1).expand(-1, 4)


def _generic_runner(*, media=None, max_tokens=8):
    runner = ModelRunner.__new__(ModelRunner)
    runner._piecewise_generic_on = True
    runner._piecewise_runner = _FakePiecewiseRunner(max_tokens=max_tokens)
    runner.input_data = SimpleNamespace(
        embedding_size=0,
        tokens=torch.arange(6),
        seqs=[
            SimpleNamespace(
                mm_contents=media,
                computed_prompt=False,
                to_compute_token_num=6,
            )
        ],
    )
    runner.model = _FakeModel()
    runner.input_hidden_states = torch.zeros(8, 4)
    runner.output_hidden_states = torch.zeros(8, 4)
    runner._prepare_attention_metadata = lambda input_data: None
    return runner


def test_generic_dispatch_writes_output_for_text_prefill():
    runner = _generic_runner()
    assert runner._run_generic_piecewise_forward(6)
    expected = torch.arange(6).float().unsqueeze(1).expand(-1, 4) + 1
    torch.testing.assert_close(runner.output_hidden_states[:6], expected)
    assert runner._piecewise_runner.calls == 1


def test_generic_dispatch_falls_back_for_media_and_oversized_batches():
    media_runner = _generic_runner(media={"image": [object()]})
    assert not media_runner._run_generic_piecewise_forward(6)
    assert media_runner._piecewise_runner.calls == 0

    oversized_runner = _generic_runner(max_tokens=4)
    assert not oversized_runner._run_generic_piecewise_forward(6)
    assert oversized_runner._piecewise_runner.calls == 0


def _backend_validation_runner(requested):
    runner = ModelRunner.__new__(ModelRunner)
    runner.attention_backend = requested
    runner.model_max_length = 128
    runner.max_running_seqs = 8
    runner.page_size = 16
    runner.use_mla = False
    runner.model_loader = SimpleNamespace(config=SimpleNamespace())
    return runner


def test_flashinfer_kernel_probe_failure_falls_back_to_fa4(monkeypatch):
    attempts = []

    class _Backend:
        def __init__(self, name):
            self.name = name

        def smoke_test(self, page_size):
            attempts.append((self.name, page_size))
            if self.name == "flashinfer":
                raise RuntimeError("Unsupported architecture")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model_runner_module,
        "create_qkv_attention_backend",
        lambda name, *_: _Backend(name),
    )
    monkeypatch.setattr(
        model_runner_module, "propagate_serving_config", lambda config: None
    )

    runner = _backend_validation_runner("flashinfer")
    runner.verify_config()
    assert runner.attention_backend == "fa4"
    assert runner._validated_qkv_attention_backend.name == "fa4"
    assert attempts == [("flashinfer", 16)]


def test_auto_backend_keeps_fa4_without_allocating_flashinfer(monkeypatch):
    created = []

    class _Backend:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        model_runner_module,
        "create_qkv_attention_backend",
        lambda name, *_: created.append(name) or _Backend(name),
    )
    monkeypatch.setattr(
        model_runner_module, "propagate_serving_config", lambda config: None
    )

    runner = _backend_validation_runner("auto")
    runner.verify_config()
    assert runner.attention_backend == "fa4"
    assert created == ["fa4"]
