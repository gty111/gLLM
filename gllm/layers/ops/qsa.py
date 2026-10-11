"""Fused QSA query normalization, rotary embedding and paged cache publication."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(
    Source,
    Norm,
    CosSin,
    Pos,
    Axes,
    base,
    width: tl.constexpr,
    row,
    POS_STRIDE: tl.constexpr,
    MROPE: tl.constexpr,
    DIM: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUNDED_ROPE: tl.constexpr,
):
    d = tl.arange(0, BLOCK)
    x = tl.load(Source + base + d, d < DIM, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / DIM + EPS)
    w = tl.load(Norm + d, d < DIM, 0).to(tl.float32)
    normalized = (x * inv * (1 + w)).to(Source.dtype.element_ty).to(tl.float32)
    paired = tl.where(d < ROT // 2, d + ROT // 2, d - ROT // 2)
    xp = tl.load(Source + base + paired, d < ROT, 0).to(tl.float32)
    wp = tl.load(Norm + paired, d < ROT, 0).to(tl.float32)
    other = (xp * inv * (1 + wp)).to(Source.dtype.element_ty).to(tl.float32)
    freq = d % (ROT // 2)
    axis = tl.load(Axes + freq, d < ROT, 0) if MROPE else 0
    position = tl.load(Pos + axis * POS_STRIDE + row)
    cos = tl.load(CosSin + position * width + freq, d < ROT, 0)
    sin = tl.load(CosSin + position * width + ROT // 2 + freq, d < ROT, 0)
    if ROUNDED_ROPE:
        # The published indexer path uses separate low-precision torch products.
        cos = cos.to(Source.dtype.element_ty).to(tl.float32)
        sin = sin.to(Source.dtype.element_ty).to(tl.float32)
        left = (normalized * cos).to(Source.dtype.element_ty).to(tl.float32)
        right = (other * sin).to(Source.dtype.element_ty).to(tl.float32)
        rotated = left + tl.where(d < ROT // 2, -right, right)
    else:
        rotated = normalized * cos + tl.where(d < ROT // 2, -other, other) * sin
    return tl.where(d < ROT, rotated, normalized)


@triton.jit(do_not_specialize=["REQUESTS"])
def _prepare_qsa(
    QKV,
    Index,
    QNorm,
    KNorm,
    INorm,
    CosSin,
    Pos,
    Axes,
    Slots,
    Boundaries,
    KCache,
    VCache,
    ICache,
    Q,
    IQ,
    Requests,
    QKV_S: tl.constexpr,
    I_S: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    MROPE: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    INDEX_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    IDIM: tl.constexpr,
    ROT: tl.constexpr,
    COS_WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    PAGE: tl.constexpr,
    REQUESTS,
    ENTRY: tl.constexpr,
    BD: tl.constexpr,
    BI: tl.constexpr,
    BE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    if head < HEADS:
        q = _norm_rope(
            QKV,
            QNorm,
            CosSin,
            Pos,
            Axes,
            row * QKV_S + head * 2 * DIM,
            COS_WIDTH,
            row,
            POS_STRIDE,
            MROPE,
            DIM,
            ROT,
            EPS,
            BD,
            False,
        )
        d = tl.arange(0, BD)
        tl.store(Q + (row * HEADS + head) * DIM + d, q, d < DIM)
    slot = tl.load(Slots + row).to(tl.int64)
    page, offset = slot // PAGE, slot % PAGE
    if head < KV_HEADS:
        k = _norm_rope(
            QKV,
            KNorm,
            CosSin,
            Pos,
            Axes,
            row * QKV_S + 2 * HEADS * DIM + head * DIM,
            COS_WIDTH,
            row,
            POS_STRIDE,
            MROPE,
            DIM,
            ROT,
            EPS,
            BD,
            False,
        )
        d = tl.arange(0, BD)
        v = tl.load(
            QKV + row * QKV_S + (2 * HEADS + KV_HEADS) * DIM + head * DIM + d,
            d < DIM,
            0,
        )
        tl.store(KCache + page * K0 + offset * K1 + head * K2 + d, k, d < DIM)
        tl.store(VCache + page * V0 + offset * V1 + head * V2 + d, v, d < DIM)
    if head < INDEX_HEADS:
        iq = _norm_rope(
            Index,
            INorm,
            CosSin,
            Pos,
            Axes,
            row * I_S + head * IDIM,
            COS_WIDTH,
            row,
            POS_STRIDE,
            MROPE,
            IDIM,
            ROT,
            EPS,
            BI,
            True,
        )
        d = tl.arange(0, BI)
        tl.store(IQ + (row * INDEX_HEADS + head) * IDIM + d, iq, d < IDIM)
    if head == 0:
        d = tl.arange(0, BE)
        raw = tl.load(Index + row * I_S + INDEX_HEADS * IDIM + d, d < IDIM, 0)
        freq = (d - 2 * IDIM) % (ROT // 2)
        axis = (
            tl.load(Axes + freq, (d >= 2 * IDIM) & (d < 2 * IDIM + ROT), 0)
            if MROPE
            else 0
        )
        position = tl.load(Pos + axis * POS_STRIDE + row)
        cache_offset = tl.where(d < 2 * IDIM + ROT // 2, freq, ROT // 2 + freq)
        rotary = tl.load(
            CosSin + position * COS_WIDTH + cache_offset,
            (d >= 2 * IDIM) & (d < 2 * IDIM + ROT),
            0,
        )
        entry = tl.where(d < IDIM, raw, tl.where(d >= 2 * IDIM, rotary, 0))
        tl.store(ICache + page * I0 + offset * I1 + d, entry, d < ENTRY)
        low, high = 0, REQUESTS
        while low < high:
            mid = (low + high) // 2
            boundary = tl.load(Boundaries + mid + 1)
            goes_right = row >= boundary
            low = tl.where(goes_right, mid + 1, low)
            high = tl.where(goes_right, high, mid)
        tl.store(Requests + row, low)


def prepare_qsa(
    qkv,
    index,
    q_weight,
    k_weight,
    index_weight,
    positions,
    axes,
    slots,
    boundaries,
    keys,
    values,
    index_cache,
    cos_sin,
    *,
    heads,
    kv_heads,
    index_heads,
    head_dim,
    index_dim,
    rotary_dim,
    eps,
    page_size,
):
    """Publish raw caches before the separate compressed-index update launch."""
    rows = qkv.shape[0]
    q = qkv.new_empty((rows, heads, head_dim))
    iq = index.new_empty((rows, index_heads, index_dim))
    requests = torch.empty(rows, device=qkv.device, dtype=torch.int32)
    _prepare_qsa[(rows, max(heads, kv_heads, index_heads))](
        qkv,
        index,
        q_weight,
        k_weight,
        index_weight,
        cos_sin,
        positions,
        axes,
        slots,
        boundaries,
        keys,
        values,
        index_cache,
        q,
        iq,
        requests,
        qkv.stride(0),
        index.stride(0),
        *keys.stride()[:3],
        *values.stride()[:3],
        *index_cache.stride()[:2],
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.ndim == 2,
        heads,
        kv_heads,
        index_heads,
        head_dim,
        index_dim,
        rotary_dim,
        cos_sin.shape[1],
        eps,
        page_size,
        boundaries.numel() - 1,
        index_cache.shape[-1],
        triton.next_power_of_2(head_dim),
        triton.next_power_of_2(index_dim),
        triton.next_power_of_2(index_cache.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, iq, requests
