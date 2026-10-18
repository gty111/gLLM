"""Paged sparse attention and compressed index-cache updates on GPU."""

import torch
import triton
import triton.language as tl
from flashinfer import top_k_ragged_transform


@triton.jit
def _compress_index(
    K,
    Table,
    Pos,
    Req,
    Norm,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    DIM: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Cache arenas exceed 4 GB; widen physical page IDs before stride products.
    token = tl.program_id(0)
    position = tl.load(Pos + token)
    if (position + 1) % RATIO == 0:
        request = tl.load(Req + token)
        first = position + 1 - RATIO
        d = tl.arange(0, BLOCK)
        pooled = tl.full((BLOCK,), 0, tl.float32)
        for offset in range(RATIO):
            logical = first + offset
            page = tl.load(Table + request * TS + logical // PAGE).to(tl.int64)
            pooled += tl.load(
                K + page * KS0 + (logical % PAGE) * KS1 + d, d < DIM, 0
            ).to(tl.float32)
        pooled = (pooled / RATIO).to(K.dtype.element_ty).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / DIM + EPS)
        weight = tl.load(Norm + d, d < DIM, 0).to(tl.float32)
        normed = (pooled * inv * (weight + 1)).to(K.dtype.element_ty).to(tl.float32)
        # Rotate the normalized pooled key with the first token's frequency pairs.
        page = tl.load(Table + request * TS + first // PAGE).to(tl.int64)
        base = K + page * KS0 + (first % PAGE) * KS1
        paired = tl.where(d < ROT // 2, d + ROT // 2, d - ROT // 2)
        other = tl.full((BLOCK,), 0, tl.float32)
        # Recompute the paired normalization without a cross-warp gather.
        for offset in range(RATIO):
            logical = first + offset
            source_page = tl.load(Table + request * TS + logical // PAGE).to(tl.int64)
            other += tl.load(
                K + source_page * KS0 + (logical % PAGE) * KS1 + paired, d < ROT, 0
            ).to(tl.float32)
        other = (other / RATIO).to(K.dtype.element_ty).to(tl.float32)
        other_w = tl.load(Norm + paired, d < ROT, 0).to(tl.float32)
        other = (other * inv * (other_w + 1)).to(K.dtype.element_ty).to(tl.float32)
        cos = tl.load(base + 2 * DIM + d % (ROT // 2), d < ROT, 0).to(tl.float32)
        sin = tl.load(base + 2 * DIM + ROT // 2 + d % (ROT // 2), d < ROT, 0).to(
            tl.float32
        )
        rotated = normed * cos + tl.where(d < ROT // 2, -other, other) * sin
        tl.store(base + DIM + d, tl.where(d < ROT, rotated, normed), d < DIM)


@triton.jit
def _index_scores(
    Q,
    K,
    Table,
    Pos,
    Requests,
    Scores,
    Lengths,
    Offsets,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    TS: tl.constexpr,
    NB: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BD: tl.constexpr,
    BB: tl.constexpr,
    USE_DOT: tl.constexpr = False,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    blocks = tile * BB + tl.arange(0, BB)
    d = tl.arange(0, BD)
    visible = (tl.load(Pos + row) + 1) // RATIO
    if tile == 0:
        # Publish mutable GPU row lengths in the scoring launch. Graph replay
        # can select every block directly when this prefix fits the budget.
        tl.store(Lengths + row, tl.minimum(tl.maximum(visible, 0), NB))
        tl.store(Offsets + row, 0)
    valid = (blocks < NB) & (blocks < visible)
    score = tl.full((BB,), -float("inf"), tl.float32)
    # A full-range graph must retain its score shape as contexts grow, but
    # tiles beyond this row's visible prefix need no key loads or dot products.
    if tile * BB < visible:
        request = tl.load(Requests + row)
        logical = blocks * RATIO
        page = tl.load(Table + request * TS + logical // PAGE, valid, 0).to(tl.int64)
        keys = tl.load(
            K
            + page[:, None] * KS0
            + (logical % PAGE)[:, None] * KS1
            + DIM
            + d[None, :],
            valid[:, None] & (d[None, :] < DIM),
            0,
        )
        if USE_DOT:
            heads = tl.arange(0, 16)
            query = tl.load(
                Q + row * Q0 + heads[:, None] * Q1 + d[None, :],
                (heads[:, None] < HEADS) & (d[None, :] < DIM),
                0,
            )
            dots = tl.dot(query, tl.trans(keys))
            score = tl.sum(tl.maximum(dots, 0), 0)
        else:
            keys = keys.to(tl.float32)
            score = tl.full((BB,), 0, tl.float32)
            for head in range(HEADS):
                query = tl.load(Q + row * Q0 + head * Q1 + d, d < DIM, 0).to(tl.float32)
                score += tl.maximum(tl.sum(keys * query[None, :], 1), 0)
        score *= DIM**-0.5
    tl.store(
        Scores + row * NB + blocks, tl.where(valid, score, -float("inf")), blocks < NB
    )


@triton.jit
def _paged_sparse(
    Q,
    K,
    V,
    Table,
    Indices,
    Requests,
    O,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    NI: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    request = tl.load(Requests + row)
    d = tl.arange(0, BD)
    s = tl.arange(0, BS)
    query = tl.load(Q + row * Q0 + head * Q1 + d, d < DIM, 0).to(tl.float32)
    maximum = tl.full((), -1.0e30, tl.float32)
    denominator = tl.full((), 0, tl.float32)
    accumulator = tl.full((BD,), 0, tl.float32)
    for tile in range(triton.cdiv(NI, BS)):
        ordinal = tile * BS + s
        logical = tl.load(Indices + row * NI + ordinal, ordinal < NI, -1)
        valid = (ordinal < NI) & (logical >= 0)
        page = tl.load(Table + request * TS + logical // PAGE, valid, 0).to(tl.int64)
        offsets = (
            page[:, None] * KS0
            + (logical % PAGE)[:, None] * KS1
            + (head // GROUP) * KS2
            + d[None, :]
        )
        keys = tl.load(K + offsets, valid[:, None] & (d[None, :] < DIM), 0).to(
            tl.float32
        )
        values = tl.load(V + offsets, valid[:, None] & (d[None, :] < DIM), 0).to(
            tl.float32
        )
        scores = tl.where(valid, tl.sum(keys * query[None, :], 1) * SCALE, -1.0e30)
        updated = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp(maximum - updated)
        probabilities = tl.where(valid, tl.exp(scores - updated), 0)
        denominator = denominator * alpha + tl.sum(probabilities, 0)
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * values, 0)
        maximum = updated
    tl.store(
        O + row * Q0 + head * Q1 + d,
        accumulator / tl.maximum(denominator, 1.0e-20),
        d < DIM,
    )


def update_compressed_index(
    cache, table, positions, requests, norm, *, page_size, ratio, rotary_dim, eps
):
    dim = norm.numel()
    _compress_index[(positions.numel(),)](
        cache,
        table,
        positions,
        requests,
        norm,
        cache.stride(0),
        cache.stride(1),
        table.stride(0),
        page_size,
        ratio,
        dim,
        rotary_dim,
        eps,
        triton.next_power_of_2(dim),
    )


@triton.jit(do_not_specialize=["ROWS"])
def _paged_gqa(
    Q,
    K,
    V,
    Table,
    Indices,
    Requests,
    O,
    Partial,
    Max,
    Den,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    TS: tl.constexpr,
    ROWS,
    HEADS: tl.constexpr,
    NI: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row, kv_head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = kv_head * GROUP + tl.arange(0, BM)
    d = tl.arange(0, DIM)
    n = tl.arange(0, BN)
    request = tl.load(Requests + row)
    q = tl.load(
        Q + row * Q0 + h[:, None] * Q1 + d[None, :],
        tl.arange(0, BM)[:, None] < GROUP,
        0,
    )
    maximum = tl.full((BM,), -1.0e30, tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    accumulator = tl.full((BM, DIM), 0, tl.float32)
    for tile in range(split, triton.cdiv(NI, BN), SPLITS):
        ordinal = tile * BN + n
        logical = tl.load(Indices + row * NI + ordinal, ordinal < NI, -1)
        valid = (ordinal < NI) & (logical >= 0)
        page = tl.load(
            Table + request * TS + tl.maximum(logical, 0) // PAGE, valid, 0
        ).to(tl.int64)
        offset = tl.maximum(logical, 0) % PAGE
        k = tl.load(
            K + page[None, :] * K0 + offset[None, :] * K1 + kv_head * K2 + d[:, None],
            valid[None, :],
            0,
        )
        v = tl.load(
            V + page[:, None] * V0 + offset[:, None] * V1 + kv_head * V2 + d[None, :],
            valid[:, None],
            0,
        )
        scores = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        scores = tl.where(valid[None, :], scores, -1.0e30)
        updated = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - updated)
        probability = tl.where(valid[None, :], tl.exp2(scores - updated[:, None]), 0)
        accumulator = tl.dot(probability.to(v.dtype), v, accumulator * alpha[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = updated
    if SPLITS == 1:
        tl.store(
            O + row * Q0 + h[:, None] * Q1 + d[None, :],
            accumulator / tl.maximum(denominator[:, None], 1.0e-20),
            tl.arange(0, BM)[:, None] < GROUP,
        )
    else:
        head_row = (split * ROWS + row) * HEADS + h
        tl.store(
            Partial + head_row[:, None] * DIM + d[None, :],
            accumulator,
            tl.arange(0, BM)[:, None] < GROUP,
        )
        tl.store(Max + head_row, maximum, tl.arange(0, BM) < GROUP)
        tl.store(Den + head_row, denominator, tl.arange(0, BM) < GROUP)


@triton.jit(do_not_specialize=["ROWS"])
def _merge_gqa(
    Partial,
    Max,
    Den,
    O,
    ROWS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, DIM)
    offset = (s * ROWS + row) * HEADS + head
    maximum = tl.load(Max + offset, s < SPLITS, -1.0e30)
    weight = tl.exp2(maximum - tl.max(maximum, 0))
    denominator = tl.load(Den + offset, s < SPLITS, 0)
    values = tl.load(
        Partial + offset[:, None] * DIM + d[None, :], s[:, None] < SPLITS, 0
    )
    total = tl.sum(denominator * weight, 0)
    output = tl.sum(values * weight[:, None], 0) / tl.maximum(total, 1.0e-20)
    tl.store(O + (row * HEADS + head) * DIM + d, output)


@triton.jit
def _expand_selected_blocks(
    Chosen,
    Positions,
    Output,
    COUNT: tl.constexpr,
    RATIO: tl.constexpr,
    WIDTH: tl.constexpr,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * B + tl.arange(0, B)
    selected = column < COUNT * RATIO
    ordinal = column // RATIO
    chosen = tl.load(Chosen + row * COUNT + ordinal, selected, -1)
    position = tl.load(Positions + row)
    tail = ((position + 1) // RATIO) * RATIO + column - COUNT * RATIO
    logical = tl.where(selected, chosen * RATIO + column % RATIO, tail)
    valid = tl.where(selected, chosen >= 0, tail <= position)
    tl.store(
        Output + row * WIDTH + column, tl.where(valid, logical, -1), column < WIDTH
    )


@triton.jit
def _select_visible_tokens(Pos, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ids = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    position = tl.load(Pos + row)
    tl.store(Out + row * WIDTH + ids, tl.where(ids <= position, ids, -1), ids < WIDTH)


def select_paged_query(
    query,
    cache,
    table,
    positions,
    requests,
    *,
    max_context,
    page_size,
    ratio,
    budget,
    search_context=None,
):
    # Allocated page-table width bounds every GPU-patched position. Bucket it
    # to keep the eager kernel cache small. Captured graphs use the full range;
    # their query positions may grow without changing the replay signature.
    context = (
        max_context
        if search_context is None
        else min(max_context, triton.next_power_of_2(max(page_size, search_context)))
    )
    blocks = max(1, triton.cdiv(context, ratio))
    count = min(budget // ratio, blocks)
    width = count * ratio + ratio - 1
    selected = torch.empty(
        (query.shape[0], width), device=query.device, dtype=torch.int64
    )
    if search_context is not None and context <= budget:
        # Every visible token fits. Scores cannot remove any key, so skip the
        # index GEMM/top-k and emit the complete causal prefix directly.
        _select_visible_tokens[(query.shape[0], triton.cdiv(width, 256))](
            positions, selected, width, 256
        )
        return selected
    scores = torch.empty(
        query.shape[0], blocks, device=query.device, dtype=torch.float32
    )
    lengths = torch.empty(query.shape[0], device=query.device, dtype=torch.int32)
    offsets = torch.empty_like(lengths)
    use_dot = (
        query.dtype in (torch.bfloat16, torch.float16)
        and query.shape[0] >= 8
        and query.dtype == cache.dtype
        and query.shape[1] <= 16
        and query.shape[2] >= 16
    )
    tile = 64 if use_dot else 16
    _index_scores[(query.shape[0], triton.cdiv(blocks, tile))](
        query,
        cache,
        table,
        positions,
        requests,
        scores,
        lengths,
        offsets,
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        cache.stride(1),
        table.stride(0),
        blocks,
        page_size,
        ratio,
        query.shape[1],
        query.shape[2],
        triton.next_power_of_2(query.shape[2]),
        tile,
        use_dot,
    )
    chosen = top_k_ragged_transform(
        scores,
        offsets,
        lengths,
        count,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=True,
    )
    _expand_selected_blocks[(query.shape[0], triton.cdiv(width, 256))](
        chosen, positions, selected, count, ratio, width, 256
    )
    return selected


def paged_sparse_attention(
    query, keys, values, table, selected, requests, *, page_size, scale
):
    query = query.contiguous()
    output = torch.empty_like(query)
    if query.dtype in (torch.bfloat16, torch.float16) and query.shape[2] in (
        64,
        128,
        256,
    ):
        rows, heads, dim = query.shape
        kv_heads = keys.shape[2]
        # Small decode/verify grids split the sparse KV work to fill the GPU.
        # Prefill already has enough query rows and avoids the merge launch.
        programs = rows * kv_heads
        target_splits = (
            16
            if programs <= 16
            else (
                8
                if programs <= 32
                else 4
                if programs <= 64
                else 2
                if programs <= 128
                else 1
            )
        )
        splits = min(triton.cdiv(selected.shape[1], 64), target_splits)
        partial = (
            torch.empty(
                (splits, rows, heads, dim), device=query.device, dtype=torch.float32
            )
            if splits > 1
            else output
        )
        maximum = (
            torch.empty((splits, rows, heads), device=query.device, dtype=torch.float32)
            if splits > 1
            else output
        )
        denominator = torch.empty_like(maximum) if splits > 1 else output
        _paged_gqa[(rows, kv_heads, splits)](
            query,
            keys,
            values,
            table,
            selected,
            requests,
            output,
            partial,
            maximum,
            denominator,
            query.stride(0),
            query.stride(1),
            *keys.stride()[:3],
            *values.stride()[:3],
            table.stride(0),
            rows,
            heads,
            selected.shape[1],
            page_size,
            heads // kv_heads,
            dim,
            scale,
            splits,
            max(16, triton.next_power_of_2(heads // kv_heads)),
            64,
            num_warps=4,
        )
        if splits > 1:
            _merge_gqa[(rows, heads)](
                partial,
                maximum,
                denominator,
                output,
                rows,
                heads,
                dim,
                splits,
                triton.next_power_of_2(splits),
            )
        return output
    _paged_sparse[(query.shape[0], query.shape[1])](
        query,
        keys,
        values,
        table,
        selected,
        requests,
        output,
        query.stride(0),
        query.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        table.stride(0),
        selected.shape[1],
        page_size,
        query.shape[1] // keys.shape[2],
        query.shape[2],
        scale,
        triton.next_power_of_2(query.shape[2]),
        32,
        num_warps=4,
    )
    return output
