"""Sparse GQA inference against an independent FP32 attention reference."""

import pytest
import torch

from gllm.layers.attention.paged_sparse import _paged_gqa, paged_sparse_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
