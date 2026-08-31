import torch

import gllm.distributed.parallel_state as parallel_state


def test_ep_all_reduce_reuses_tp_fast_path_without_dp(monkeypatch):
    source = torch.tensor([1.0])
    reduced = torch.tensor([4.0])
    calls = []

    monkeypatch.setattr(parallel_state, "_DP_SIZE", 1)
    monkeypatch.setattr(
        parallel_state,
        "tensor_model_parallel_all_reduce",
        lambda value: calls.append(value) or reduced,
    )

    assert parallel_state.ep_all_reduce(source) is reduced
    assert calls == [source]


def test_ep_all_reduce_keeps_ep_group_under_dp(monkeypatch):
    source = torch.tensor([1.0])
    ep_group = object()
    calls = []

    monkeypatch.setattr(parallel_state, "_DP_SIZE", 2)
    monkeypatch.setattr(parallel_state, "_EP_GROUP", ep_group)
    monkeypatch.setattr(
        parallel_state.dist,
        "all_reduce",
        lambda value, group: calls.append((value, group)),
    )
    monkeypatch.setattr(
        parallel_state,
        "tensor_model_parallel_all_reduce",
        lambda value: (_ for _ in ()).throw(
            AssertionError("DP+EP must not use the TP-only custom all-reduce")
        ),
    )

    assert parallel_state.ep_all_reduce(source) is source
    assert calls == [(source, ep_group)]


def test_tp_all_reduce_pads_small_unaligned_message_for_custom_ar(monkeypatch):
    source = torch.arange(5, dtype=torch.float32)
    calls = []

    class _FakeCustomAllreduce:
        @staticmethod
        def should_custom_ar(value):
            return value.numel() * value.element_size() % 16 == 0

        @staticmethod
        def all_reduce(value):
            calls.append(value.clone())
            return value * 4

    import gllm.distributed as distributed

    monkeypatch.setattr(
        distributed, "get_custom_allreduce", lambda: _FakeCustomAllreduce()
    )
    monkeypatch.setattr(
        parallel_state.dist,
        "all_reduce",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an alignable small message must not fall back to NCCL")
        ),
    )

    actual = parallel_state.tensor_model_parallel_all_reduce(source)
    torch.testing.assert_close(actual, source * 4)
    assert actual.shape == source.shape
    assert calls[0].numel() == 8
    assert torch.count_nonzero(calls[0][5:]) == 0
