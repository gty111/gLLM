"""Fused decode-step kernel for the DeepSeek-V4 learned KV compressor.

The reference implementation in :mod:`...deepseek_v4.compressor` expresses one
decode step as ~22 separate PyTorch ops -- scatter the new token into the
rolling state, concatenate the two overlap halves, softmax over the pooled
axis, weighted-sum, then shift the state on a group boundary. At decode sizes
every one of those is a launch-bound kernel of a couple of microseconds, and
the model runs 41 compressors per step (21 at ratio 4, 20 at ratio 128).
Profiling a single-token decode attributed 3.45 ms/step and 892 kernels to
that one function.

All of it fits in one program per (row, channel block): the pooled axis is
``2*ratio`` (overlap) or ``ratio``, always a power of two and at most 128, so
the whole tile lives in registers and the softmax is a register reduction.

The kernel is numerically equivalent to the reference, not an approximation --
same fp32 accumulation, same softmax. ``tests/test_deepseek_v4_compressor.py``
pins it against the reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_decode_kernel(
    kv_ptr,           # (B, 1, C) new token
    score_ptr,        # (B, 1, C)
    ape_ptr,          # (RATIO, C)
    pos_ptr,          # (B,) int64 absolute positions
    state_kv_ptr,     # (B, P, C)
    state_score_ptr,  # (B, P, C)
    out_ptr,          # (B, 1, HEAD_DIM)
    boundary_ptr,     # (B,) int8
    head_dim,
    C: tl.constexpr,          # coff * head_dim
    P: tl.constexpr,          # coff * ratio -- pooled axis, power of two
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D

    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    pos = tl.load(pos_ptr + row)
    cursor = pos % RATIO
    dst = cursor + RATIO if OVERLAP else cursor
    boundary = ((pos + 1) % RATIO) == 0
    if d0 == 0:
        tl.store(boundary_ptr + row, boundary.to(tl.int8))

    offs_p = tl.arange(0, P)
    # Overlap pools the low half of the first ``RATIO`` rows against the high
    # half of the rest; without overlap C == head_dim and this is just offs_d.
    col = tl.where((offs_p < RATIO)[:, None], offs_d[None, :],
                   head_dim + offs_d[None, :])
    tile = row * P * C + offs_p[:, None] * C + col
    tile_mask = tl.broadcast_to(mask_d[None, :], (P, BLOCK_D))

    # The new token, in the same column layout as the tile it is inserted into.
    ins_kv = tl.load(kv_ptr + row * C + col, mask=tile_mask, other=0.0)
    ins_ape = tl.load(ape_ptr + cursor * C + col, mask=tile_mask, other=0.0)
    ins_score = tl.load(score_ptr + row * C + col, mask=tile_mask, other=0.0) + ins_ape

    # Write the new token into the rolling state. Both halves must land: the
    # low half is what the *next* group reads after the boundary shift.
    tl.store(state_kv_ptr + row * P * C + dst * C + offs_d,
             tl.load(kv_ptr + row * C + offs_d, mask=mask_d, other=0.0),
             mask=mask_d)
    tl.store(state_score_ptr + row * P * C + dst * C + offs_d,
             tl.load(score_ptr + row * C + offs_d, mask=mask_d, other=0.0)
             + tl.load(ape_ptr + cursor * C + offs_d, mask=mask_d, other=0.0),
             mask=mask_d)
    if OVERLAP:
        hi = head_dim + offs_d
        mask_hi = offs_d < head_dim
        tl.store(state_kv_ptr + row * P * C + dst * C + hi,
                 tl.load(kv_ptr + row * C + hi, mask=mask_hi, other=0.0),
                 mask=mask_hi)
        tl.store(state_score_ptr + row * P * C + dst * C + hi,
                 tl.load(score_ptr + row * C + hi, mask=mask_hi, other=0.0)
                 + tl.load(ape_ptr + cursor * C + hi, mask=mask_hi, other=0.0),
                 mask=mask_hi)

    # Pool. Substituting the inserted row in registers avoids depending on the
    # store above having landed.
    is_dst = (offs_p == dst)[:, None]
    pooled_kv = tl.where(is_dst, ins_kv,
                         tl.load(state_kv_ptr + tile, mask=tile_mask, other=0.0))
    pooled_score = tl.where(
        is_dst, ins_score,
        tl.load(state_score_ptr + tile, mask=tile_mask, other=-float("inf")))

    m = tl.max(pooled_score, axis=0)
    w = tl.exp(pooled_score - m[None, :])
    w = w / tl.sum(w, axis=0)[None, :]
    tl.store(out_ptr + row * head_dim + offs_d,
             tl.sum(pooled_kv * w, axis=0), mask=mask_d)

    # On a group boundary the overlap half becomes the new leading half.
    if OVERLAP:
        if boundary:
            offs_lo = tl.arange(0, RATIO)
            src = offs_lo + RATIO
            base = row * P * C
            src_is_dst = (src == dst)[:, None]
            for half in tl.static_range(2):
                cc = offs_d[None, :] + half * head_dim
                cc = tl.broadcast_to(cc, (RATIO, BLOCK_D))
                m2 = tl.broadcast_to(mask_d[None, :], (RATIO, BLOCK_D))
                new_kv = tl.load(kv_ptr + row * C + cc, mask=m2, other=0.0)
                new_ape = tl.load(ape_ptr + cursor * C + cc, mask=m2, other=0.0)
                new_sc = tl.load(score_ptr + row * C + cc, mask=m2, other=0.0) + new_ape
                k = tl.where(src_is_dst, new_kv,
                             tl.load(state_kv_ptr + base + src[:, None] * C + cc,
                                     mask=m2, other=0.0))
                s = tl.where(src_is_dst, new_sc,
                             tl.load(state_score_ptr + base + src[:, None] * C + cc,
                                     mask=m2, other=-float("inf")))
                tl.store(state_kv_ptr + base + offs_lo[:, None] * C + cc, k, mask=m2)
                tl.store(state_score_ptr + base + offs_lo[:, None] * C + cc, s, mask=m2)


def compress_decode_batch_fused(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    positions: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    block_d: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-token compressor step for a whole decode batch, in one kernel."""
    batch, _, C = kv.shape
    overlap = ratio == 4
    coff = 1 + overlap
    head_dim = C // coff
    P = coff * ratio

    out = kv.new_empty((batch, 1, head_dim))
    boundary = torch.empty(batch, device=kv.device, dtype=torch.int8)
    if block_d is None:
        # The whole (P, BLOCK_D) tile is held in registers, so a wide block
        # spills: at ratio 128 a 256-wide block measured 20x slower than a
        # 32-wide one (109.7 us vs 5.6 us) and 1024 ran out of memory. 32 was
        # the sweep optimum at both ratios -- 10.6x over the reference at
        # ratio 4, 16.6x at ratio 128.
        block_d = min(triton.next_power_of_2(head_dim), 32)
    grid = (batch, triton.cdiv(head_dim, block_d))
    _compress_decode_kernel[grid](
        kv, score, ape, positions,
        state_kv, state_score, out, boundary,
        head_dim,
        C=C, P=P, RATIO=ratio, OVERLAP=overlap, BLOCK_D=block_d,
        num_warps=4,
    )
    return out, boundary.to(torch.bool)


__all__ = ["compress_decode_batch_fused"]
