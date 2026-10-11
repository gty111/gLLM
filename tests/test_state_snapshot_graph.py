"""Snapshot destinations can change after graph capture, including skipped rows."""

import pytest
import torch

from gllm.layers.ops.state_snapshot import copy_slot_snapshots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_snapshot_replay_uses_current_slots(dtype):
    backing = torch.zeros(12, 3, 8, dtype=dtype, device="cuda")
    state = backing[:, 1]
    state[2].fill_(17)
    state[5].fill_(29)
    sources = torch.tensor([2, 5], dtype=torch.int32, device="cuda")
    targets = torch.tensor([-1, -1], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        copy_slot_snapshots(state, sources, targets)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_slot_snapshots(state, sources, targets)
    targets.copy_(torch.tensor([7, -1], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.testing.assert_close(state[7], state[2], rtol=0, atol=0)
    assert state[0].count_nonzero() == 0
    state[2].fill_(41)
    targets.copy_(torch.tensor([8, 9], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.testing.assert_close(state[8], state[2], rtol=0, atol=0)
    torch.testing.assert_close(state[9], state[5], rtol=0, atol=0)
    assert backing[:, 0].count_nonzero() == 0
    assert backing[:, 2].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype,width", [(torch.float32, 196608), (torch.bfloat16, 6144), (torch.int64, 8)]
)
@torch.inference_mode()
def test_prefill_snapshot_matches_gather_scatter(dtype, width):
    from types import SimpleNamespace

    from gllm.models.qwen3_5 import Qwen3_5GatedDeltaNet

    backing = torch.randint(0, 100, (9, 2, width), device="cuda").to(dtype)
    conv = backing[:, 0]
    temporal = backing[:, 1]
    sources = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int64)
    targets = torch.tensor([6, 8, 7], device="cuda", dtype=torch.int64)
    expected = backing.clone()
    for view in [expected[:, 0], expected[:, 1]]:
        view.index_copy_(0, targets, view.index_select(0, sources))
    publications = []
    data = SimpleNamespace(
        max_query_len=129,
        memory_manager=SimpleNamespace(ssm_segment=object()),
        get_ssm_snapshot_copy_indices=lambda row_start: (sources, targets),
        mark_ssm_snapshot_writes_enqueued=lambda: publications.append(True),
    )
    Qwen3_5GatedDeltaNet._maybe_snapshot_state(None, data, conv, temporal, row_start=2)
    torch.testing.assert_close(backing, expected, atol=0, rtol=0)
    assert publications == [True]
