"""Paged QSA must widen int32 page IDs before addressing arenas over 4 GB."""

import pytest
import torch

from gllm.layers.attention.paged_sparse import (
    paged_sparse_attention,
    select_paged_query,
    update_compressed_index,
)


@torch.inference_mode()
def test_paged_qsa_addresses_beyond_signed_int32_elements():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # One sparse-strided allocation exercises all kernels without initializing
    # the arena. The page ID and stride individually fit int32; their product
    # does not, matching the actual serving arena's layout.
    stride = 65536
    pages = 32770
    elements = (pages - 1) * stride + 16 * 512
    if torch.cuda.mem_get_info()[0] < elements * 2 + (1 << 30):
        pytest.skip("Requires approximately 4 GB of free GPU memory")
    storage = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
    index = storage.as_strided((pages, 16, 384), (stride, 384, 1))
    page = pages - 1
    table = torch.tensor([[page]], device="cuda", dtype=torch.int32)
    index[page, :2].zero_()
    index[page, :2, :128] = 1
    index[page, :2, 256:272] = 1  # Identity rotary cosine, zero sine.
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    requests = torch.zeros(2, device="cuda", dtype=torch.int64)
    norm = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    update_compressed_index(
        index,
        table,
        positions,
        requests,
        norm,
        page_size=16,
        ratio=2,
        rotary_dim=32,
        eps=1e-6,
    )
    torch.testing.assert_close(index[page, 0, 128:256], torch.ones_like(norm))
    selected = select_paged_query(
        torch.ones((1, 1, 128), device="cuda", dtype=torch.bfloat16),
        index,
        table,
        positions[1:],
        requests[:1],
        max_context=4,
        page_size=16,
        ratio=2,
        budget=2,
    )
    assert sorted(selected[0][selected[0] >= 0].tolist()) == [0, 1]

    keys = storage.as_strided((pages, 16, 1, 64), (stride, 512, 64, 1))
    values = storage.as_strided(
        (pages, 16, 1, 64), (stride, 512, 64, 1), storage_offset=256
    )
    keys[page, :2].zero_()
    values[page, 0].fill_(1)
    values[page, 1].fill_(3)
    query = torch.zeros((1, 2, 64), device="cuda", dtype=torch.bfloat16)
    output = paged_sparse_attention(
        query, keys, values, table, selected, requests[:1], page_size=16, scale=0.125
    )
    torch.testing.assert_close(output, torch.full_like(query, 2))

    # Snapshot metadata also carries int32 slots into the same large arena.
    from gllm.layers.ops.state_snapshot import copy_slot_snapshots

    states = storage.as_strided((pages, 16), (stride, 1))
    expected = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    states[0].copy_(expected)
    sources = torch.tensor([0], device="cuda", dtype=torch.int32)
    targets = torch.tensor([page], device="cuda", dtype=torch.int32)
    copy_slot_snapshots(states, sources, targets)
    torch.testing.assert_close(states[page], expected)
    targets.fill_(1)
    sources.fill_(page)
    copy_slot_snapshots(states, sources, targets)
    torch.testing.assert_close(states[1], expected)
