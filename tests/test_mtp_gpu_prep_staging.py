"""Pinned staging must survive launch-current/collect-previous overlap."""

from types import SimpleNamespace

import pytest
import torch

from gllm.layers.ops.state_snapshot import copy_slot_snapshots
from gllm.speculative.async_state import MtpAsyncBatchState
from gllm.speculative.gpu_prep import META_CTX, META_X1, MtpGpuPrep
from gllm.speculative.staging import MtpStagingBuffers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("second_epoch", [2, 3])
def test_overlapping_staging_keeps_first_host_payload(second_epoch):
    prep = MtpGpuPrep(
        max_bs=2,
        max_blocks=8,
        bt_width=4,
        page_size=16,
        uses_mrope=False,
        device=torch.device("cuda"),
    )
    seq = SimpleNamespace(
        page_table=[2, 3],
        ssm_block_table=[1, 2, 3, 4],
        ssm_num_accepted=1,
    )
    first_meta = torch.empty_like(prep._d_meta)
    first_pages = torch.empty_like(prep._d_pt)
    first_states = torch.empty_like(prep._d_bt)
    second_meta = torch.empty_like(prep._d_meta)
    stream = torch.cuda.Stream()
    # Warm copy machinery before delaying the stream; lazy initialization
    # would otherwise synchronize and hide premature host-buffer reuse.
    prep.push_meta([seq], 2, epoch=0, ctx_lens=[10], x1=[1], dummy_page=0)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        prep.push_meta([seq], 2, epoch=1, ctx_lens=[20], x1=[123], dummy_page=0)
        first_meta.copy_(prep._d_meta)
        first_pages.copy_(prep._d_pt)
        first_states.copy_(prep._d_bt)
        seq.page_table = [5, 6]
        seq.ssm_block_table = [5, 6, 7, 8]
        # Consecutive epochs alternate slots; epoch 3 exercises guarded reuse.
        prep.push_meta(
            [seq],
            2,
            epoch=second_epoch,
            ctx_lens=[30],
            x1=[456],
            dummy_page=0,
        )
        second_meta.copy_(prep._d_meta)
    stream.synchronize()
    assert first_meta[:, META_CTX].tolist() == [20, 1]
    assert first_meta[:, META_X1].tolist() == [123, 1]
    assert first_pages[0, :2].tolist() == [2, 3]
    assert first_states[0].tolist() == [1, 2, 3, 4]
    assert second_meta[:, META_CTX].tolist() == [30, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_overlapping_remaps_preserve_request_row_indices():
    state = MtpAsyncBatchState(
        max_batch_size=3,
        k=3,
        hidden_size=2,
        hidden_dtype=torch.float32,
        device=torch.device("cuda"),
    )
    state.seq_ids = (1, 2, 3)
    state.batch_size = 3
    state.remap((3, 1, 2))
    torch.cuda.synchronize()
    state.seq_ids = (1, 2, 3)
    state.batch_size = 3
    state.context_lens.copy_(torch.tensor([20, 30, 40], device="cuda"))
    state.relay_tokens.copy_(torch.tensor([120, 130, 140], device="cuda"))
    state.resume_num_accepted.copy_(torch.tensor([1, 2, 3], device="cuda"))
    state.relay_hidden.copy_(
        torch.tensor([[20, 21], [30, 31], [40, 41]], device="cuda")
    )
    snapshots = [torch.empty((n, 5), device="cuda") for n in (3, 2, 1)]
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        for ids, out in zip(((3, 1, 2), (2, 3), (3,)), snapshots):
            state.remap(ids)
            n = len(ids)
            out[:, 0].copy_(state.context_lens[:n])
            out[:, 1].copy_(state.relay_tokens[:n])
            out[:, 2].copy_(state.resume_num_accepted[:n])
            out[:, 3:].copy_(state.relay_hidden[:n])
    stream.synchronize()
    rows = {1: [20, 120, 1, 20, 21], 2: [30, 130, 2, 30, 31], 3: [40, 140, 3, 40, 41]}
    for ids, out in zip(((3, 1, 2), (2, 3), (3,)), snapshots):
        assert out.tolist() == [rows[i] for i in ids]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_completed_prefill_staging_preserves_each_launch():
    staging = MtpStagingBuffers(2, 8, 2, torch.float32, torch.device("cuda"))
    snapshots = [
        torch.empty((2, 1), device="cuda", dtype=torch.int64) for _ in range(3)
    ]
    stream = torch.cuda.Stream()
    staging.stage_completed_prefills([0], [10])
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        for (idx, ctx), out in zip(((1, 40), (0, 100), (1, 200)), snapshots):
            indices, contexts = staging.stage_completed_prefills([idx], [ctx])
            out[0].copy_(indices)
            out[1].copy_(contexts)
    stream.synchronize()
    assert [out.tolist() for out in snapshots] == [
        [[1], [40]],
        [[0], [100]],
        [[1], [200]],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_overlapping_token_boundary_patches_preserve_sources():
    staging = MtpStagingBuffers(2, 8, 2, torch.float32, torch.device("cuda"))
    shifted = torch.zeros(3, device="cuda", dtype=torch.int64)
    gpu_src = torch.tensor([101, 202], device="cuda")
    snapshots = [torch.empty_like(shifted) for _ in range(3)]
    stream = torch.cuda.Stream()
    staging.patch_shifted_tokens(shifted, [(0, 1)], gpu_src, [(1, 0)])
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        for (host, gpu), out in zip(
            (([(0, 13)], [(1, 0)]), ([(1, 23)], [(2, 1)]), ([(2, 33)], [(0, 0)])),
            snapshots,
        ):
            shifted.zero_()
            staging.patch_shifted_tokens(shifted, host, gpu_src, gpu)
            out.copy_(shifted)
    stream.synchronize()
    assert [out.tolist() for out in snapshots] == [
        [13, 101, 0],
        [0, 23, 202],
        [101, 0, 33],
    ]


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
