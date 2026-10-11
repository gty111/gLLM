"""Sparse GQA inference against an independent FP32 attention reference."""

import pytest
import torch
from torch.nn import functional as F

from gllm.layers.attention.paged_sparse import (
    _paged_gqa,
    paged_sparse_attention,
    select_paged_query,
    update_compressed_index,
)
from gllm.layers.attention.qsa import apply_indexer_rope, indexer_rotary_cache
from gllm.layers.layernorm import GemmaRMSNorm
from gllm.layers.ops.qsa import prepare_qsa
from gllm.layers.rotary_embedding import MRotaryEmbedding, RotaryEmbedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("context", [16, 64, 256, 1024, 4096])
@torch.inference_mode()
def test_allocated_page_bound_preserves_visible_candidates(context):
    from gllm.layers.attention.paged_sparse import select_paged_query

    torch.manual_seed(101)
    rows, ratio, page, dim = 32, 4, 16, 128
    q = torch.randn(rows, 4, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(256, page, dim * 2, device="cuda", dtype=q.dtype)
    table = torch.randperm(256, device="cuda").int()[None].expand(rows, -1).contiguous()
    positions = torch.arange(rows, device="cuda") * (context - 1) // (rows - 1)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32)

    def select(bound):
        return select_paged_query(
            q,
            cache,
            table,
            positions,
            requests,
            max_context=4096,
            page_size=page,
            ratio=ratio,
            budget=256,
            search_context=bound,
        )

    full, bounded = select(None), select(context)
    for row in range(rows):
        a, b = full[row], bounded[row]
        torch.testing.assert_close(a[a >= 0].sort().values, b[b >= 0].sort().values)
    # Full-range graph signatures must tolerate later growth and batch reorder.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select(None)
    positions.fill_(4095)
    q.normal_()
    graph.replay()
    torch.testing.assert_close(captured, select(None), atol=0, rtol=0)


def reference(q, k, v, table, selected, requests):
    outputs = []
    group = q.shape[1] // k.shape[2]
    for row in range(q.shape[0]):
        ids = selected[row][selected[row] >= 0].long()
        if not ids.numel():
            outputs.append(torch.zeros_like(q[row]))
            continue
        pages = table[requests[row], ids // k.shape[1]].long()
        keys = k[pages, ids % k.shape[1]].float().repeat_interleave(group, 1)
        values = v[pages, ids % k.shape[1]].float().repeat_interleave(group, 1)
        score = torch.einsum("hd,shd->hs", q[row].float(), keys) / q.shape[2] ** 0.5
        outputs.append(
            torch.einsum("hs,shd->hd", score.softmax(-1), values).to(q.dtype)
        )
    return torch.stack(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    "rows,heads,kv_heads,dim,width",
    [
        (1, 6, 1, 256, 2051),
        (32, 6, 1, 256, 2051),
        (128, 6, 1, 256, 517),
        (257, 8, 2, 64, 131),
    ],
)
@torch.inference_mode()
def test_sparse_gqa_tensor_core_and_splitk(rows, heads, kv_heads, dim, width):
    torch.manual_seed(17)
    k = torch.randn((160, 16, kv_heads, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn((rows, heads, dim), device="cuda", dtype=k.dtype)
    table = torch.randperm(160, device="cuda").reshape(2, 80).int()
    requests = torch.arange(rows, device="cuda") % 2
    selected = torch.randint(0, 1280, (rows, width), device="cuda", dtype=torch.int64)
    selected[:, width // 2 :] = -1
    selected[0].fill_(-1)
    expected = reference(q, k, v, table, selected, requests)
    actual = paged_sparse_attention(
        q, k, v, table, selected, requests, page_size=16, scale=dim**-0.5
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
    if rows == 32:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = paged_sparse_attention(
                q, k, v, table, selected, requests, page_size=16, scale=dim**-0.5
            )
        selected.copy_(selected.roll(1, 0))
        requests.copy_(requests.roll(1, 0))
        q.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(
            captured,
            reference(q, k, v, table, selected, requests),
            atol=2e-3,
            rtol=1e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_ragged_prefill_rows_reuse_the_compiled_kernel():
    k = torch.randn((16, 16, 1, 64), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(16, device="cuda", dtype=torch.int32)[None]
    cache = _paged_gqa.device_caches[torch.cuda.current_device()][0]
    sizes = []
    for rows in (257, 259, 272):
        q = torch.randn((rows, 6, 64), device="cuda", dtype=k.dtype)
        selected = torch.arange(67, device="cuda").expand(rows, -1).contiguous()
        requests = torch.zeros(rows, device="cuda", dtype=torch.int64)
        paged_sparse_attention(
            q, k, v, table, selected, requests, page_size=16, scale=0.125
        )
        sizes.append(len(cache))
    assert sizes[0] == sizes[1] == sizes[2], (
        "Row count triggered another prefill compilation"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("rows", [1, 32])
@torch.inference_mode()
def test_tensorcore_index_scores_and_fused_expansion(rows):
    import triton

    from gllm.layers.attention.paged_sparse import _index_scores, select_paged_query

    torch.manual_seed(51)
    heads, dim, page, ratio, blocks = 4, 128, 16, 4, 128
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(32, page, 1, dim * 2, device="cuda", dtype=q.dtype)
    table = torch.randperm(32, device="cuda").int()[None].expand(rows, -1).contiguous()
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) * 13
    requests = torch.arange(rows, device="cuda", dtype=torch.int64)
    outputs = []
    for use_dot in (False, True):
        tile = 64 if use_dot else 16
        score = torch.empty(rows, blocks, device="cuda")
        lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
        offsets = torch.empty_like(lengths)
        _index_scores[(rows, triton.cdiv(blocks, tile))](
            q,
            cache,
            table,
            positions,
            requests,
            score,
            lengths,
            offsets,
            q.stride(0),
            q.stride(1),
            cache.stride(0),
            cache.stride(1),
            table.stride(0),
            blocks,
            page,
            ratio,
            heads,
            dim,
            dim,
            tile,
            use_dot,
        )
        torch.testing.assert_close(lengths, (positions + 1) // ratio)
        torch.testing.assert_close(offsets, torch.zeros_like(offsets))
        outputs.append(score)
    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-5, rtol=2e-5)
    values, chosen = torch.topk(outputs[1], 16, -1)
    selected = (
        chosen[:, :, None] * ratio + torch.arange(ratio, device="cuda")
    ).flatten(1)
    selected = torch.where(values.isfinite().repeat_interleave(ratio, 1), selected, -1)
    first = ((positions + 1) // ratio) * ratio
    tail = first[:, None] + torch.arange(ratio - 1, device="cuda")
    tail = torch.where(tail <= positions[:, None], tail, -1)
    expected = torch.cat((selected, tail), -1)
    actual = select_paged_query(
        q,
        cache,
        table,
        positions,
        requests,
        max_context=512,
        page_size=page,
        ratio=ratio,
        budget=64,
    )
    # Selection order is backend-specific; sparse attention consumes the set.
    torch.testing.assert_close(
        actual.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("rows", [1, 32])
@torch.inference_mode()
def test_ragged_topk_graph_growing_context_and_ties(rows, monkeypatch):
    from gllm.layers.attention.paged_sparse import select_paged_query

    torch.manual_seed(117)
    page, ratio, context, budget, dim = 16, 4, 8192, 2048, 128
    q = torch.zeros(rows, 4, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(context // page, page, 2 * dim, device="cuda", dtype=q.dtype)
    table = torch.randperm(context // page, device="cuda").int()[None]
    positions = torch.full((rows,), -1, device="cuda", dtype=torch.int64)
    requests = torch.zeros(rows, device="cuda", dtype=torch.int32)

    def select():
        return select_paged_query(
            q,
            cache,
            table,
            positions,
            requests,
            max_context=context,
            page_size=page,
            ratio=ratio,
            budget=budget,
        )

    select()  # Compile before capture, including FlashInfer's graph-safe path.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select()

    def reject_torch_topk(*args, **kwargs):
        raise AssertionError("QSA must use FlashInfer selection")

    monkeypatch.setattr(torch, "topk", reject_torch_topk)
    cases = torch.tensor([-1, 0, 1, 2, 3, 4, 2047, 2048, 4095, 8191], device="cuda")
    for shift in range(cases.numel()):
        positions.copy_(
            cases[(torch.arange(rows, device="cuda") + shift) % cases.numel()]
        )
        graph.replay()
        torch.testing.assert_close(captured, select(), atol=0, rtol=0)
        for row in range(rows):
            position = int(positions[row])
            complete = min((position + 1) // ratio, budget // ratio) * ratio
            # Equal scores choose the smallest block IDs deterministically.
            first_tail = (position + 1) // ratio * ratio
            expected = torch.cat(
                (
                    torch.arange(complete, device="cuda"),
                    torch.arange(first_tail, position + 1, device="cuda"),
                )
            )
            actual = captured[row]
            torch.testing.assert_close(actual[actual >= 0].sort().values, expected)
    q.normal_()
    graph.replay()
    torch.testing.assert_close(captured, select(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_preprocess_widens_large_cache_page_offsets():
    pages, stride = 32770, 65536
    elements = (pages - 1) * stride + 16 * 768
    if torch.cuda.mem_get_info()[0] < elements * 2 + (1 << 30):
        pytest.skip("Requires approximately 4 GB of free GPU memory")
    storage = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
    keys = storage.as_strided((pages, 16, 1, 64), (stride, 768, 64, 1))
    values = storage.as_strided((pages, 16, 1, 64), (stride, 768, 64, 1), 128)
    cache = storage.as_strided((pages, 16, 384), (stride, 768, 1), 256)
    rope = RotaryEmbedding(64, 32, 128, 10000, True)
    qkv = torch.randn(1, 384, device="cuda", dtype=torch.bfloat16)
    index = torch.randn(1, 256, device="cuda", dtype=qkv.dtype)
    norm = torch.zeros(64, device="cuda", dtype=qkv.dtype)
    inorm = torch.zeros(128, device="cuda", dtype=qkv.dtype)
    slot = torch.tensor([(pages - 1) * 16], device="cuda", dtype=torch.int32)
    position = torch.tensor([37], device="cuda", dtype=torch.int64)
    bounds = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    prepare_qsa(
        qkv,
        index,
        norm,
        norm,
        inorm,
        position,
        norm,
        slot,
        bounds,
        keys,
        values,
        cache,
        rope.cos_sin_cache,
        heads=2,
        kv_heads=1,
        index_heads=1,
        head_dim=64,
        index_dim=128,
        rotary_dim=32,
        eps=1e-6,
        page_size=16,
    )
    torch.testing.assert_close(values[-1, 0, 0], qkv[0, -64:], atol=0, rtol=0)
    torch.testing.assert_close(cache[-1, 0, :128], index[0, 128:], atol=0, rtol=0)
    expected = torch.nn.functional.pad(rope.cos_sin_cache[37].to(qkv.dtype), (0, 96))
    torch.testing.assert_close(cache[-1, 0, 256:], expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    "rows,mode", [(1, "text"), (32, "section"), (128, "interleaved"), (128, "text")]
)
@torch.inference_mode()
def test_fused_qsa_norm_rope_paged_publication_and_replay(rows, mode):
    torch.manual_seed(71)
    heads, kv_heads, dim, index_heads, idim, rot = 6, 1, 256, 4, 128, 64
    dtype = torch.bfloat16
    rope = (
        RotaryEmbedding(dim, rot, 2048, 10000, True)
        if mode == "text"
        else MRotaryEmbedding(
            dim, rot, 512, 10000, True, [16, 8, 8], mode == "interleaved"
        )
    )
    if mode == "interleaved":
        axes = rope.interleaved_axis[: rot // 2]
    else:
        axes = torch.tensor(
            [0] * 16 + [1] * 8 + [2] * 8, device="cuda", dtype=torch.int32
        )
    norms = [GemmaRMSNorm(d, 1e-6).to(dtype) for d in (dim, dim, idim)]
    for norm in norms:
        norm.weight.normal_(std=0.2)
    qkv = torch.randn(
        rows, (heads * 2 + kv_heads * 2) * dim, device="cuda", dtype=dtype
    )
    index = torch.randn(rows, (index_heads + 1) * idim, device="cuda", dtype=dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) + 37
    if mode != "text":
        positions = torch.stack((positions, positions + 19, positions + 43))
    slots = torch.randperm(256, device="cuda")[:rows]
    bounds = torch.tensor(
        [0, rows] if rows == 1 else [0, 3, rows], device="cuda", dtype=torch.int32
    )
    keys = torch.zeros(16, 16, kv_heads, dim, device="cuda", dtype=dtype)
    values = torch.zeros_like(keys)
    cache = torch.zeros(16, 16, 384, device="cuda", dtype=dtype)

    def fused():
        return prepare_qsa(
            qkv,
            index,
            *(n.weight for n in norms),
            positions,
            axes,
            slots,
            bounds,
            keys,
            values,
            cache,
            rope.cos_sin_cache,
            heads=heads,
            kv_heads=kv_heads,
            index_heads=index_heads,
            head_dim=dim,
            index_dim=idim,
            rotary_dim=rot,
            eps=1e-6,
            page_size=16,
        )

    def check(actual):
        qg, k, v = qkv.split([heads * 2 * dim, kv_heads * dim, kv_heads * dim], -1)
        q = qg.reshape(rows, heads, dim * 2)[..., :dim]
        q = norms[0](q).reshape(rows, -1)
        k = norms[1](k.reshape(rows, kv_heads, dim)).reshape(rows, -1)
        q, k = rope(positions, q, k)
        iq, raw = index.split([index_heads * idim, idim], -1)
        iq = apply_indexer_rope(
            norms[2](iq.reshape(rows, index_heads, idim)), positions, rope
        )
        entry = F.pad(
            torch.cat(
                (
                    raw,
                    torch.zeros_like(raw),
                    indexer_rotary_cache(positions, rope).to(dtype),
                ),
                -1,
            ),
            (0, 64),
        )
        pages, offsets = slots // 16, slots % 16
        torch.testing.assert_close(
            actual[0], q.reshape(rows, heads, dim), atol=0.016, rtol=0.01
        )
        torch.testing.assert_close(actual[1], iq, atol=0.016, rtol=0.01)
        torch.testing.assert_close(
            keys[pages, offsets], k.reshape(rows, kv_heads, dim), atol=0.016, rtol=0.01
        )
        torch.testing.assert_close(
            values[pages, offsets], v.reshape(rows, kv_heads, dim), atol=0, rtol=0
        )
        torch.testing.assert_close(cache[pages, offsets], entry, atol=0, rtol=0)
        torch.testing.assert_close(
            actual[2].long(),
            torch.bucketize(torch.arange(rows, device="cuda"), bounds[1:], right=True),
        )
        untouched = torch.ones(256, device="cuda", dtype=torch.bool)
        untouched[slots] = False
        assert not keys.flatten(0, 1)[untouched].count_nonzero()
        assert not cache.flatten(0, 1)[untouched].count_nonzero()

    for _ in range(3):
        actual = fused()
    check(actual)
    if rows == 32:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused()
        qkv.normal_()
        index.normal_()
        positions.add_(11)
        slots.copy_(slots.roll(1))
        bounds[1] = 9
        graph.replay()
        check(actual)


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
