"""Fused mHC (multi-stream hyper-connection) kernels for DeepSeek-V4.

``mhc_pre`` and ``mhc_post`` bracket every layer, 43 layers deep, so the whole
region is launch-bound at decode sizes. Written as PyTorch ops a single-token
decode step measured 4.02 ms and 1326 kernels across the pair, plus a 20-
iteration Sinkhorn gate that on its own accounted for ~3.3k of the ~3.8k
reduction kernels in a step.

Three things are fused here:

* the gate -- ``_split_sinkhorn_kernel`` keeps the 4x4 combination matrix in
  registers for all 20 iterations, one program per token, and applies the RMS
  scale itself so the caller needs no separate launch;
* the mix projection -- ``_mhc_mix_kernel`` sweeps ``x`` once for both the 24
  mix logits and the RMS norm's sum of squares. This replaces
  ``F.linear(x_fp32, hc_fn)``, a ``(1, hc_mult*H) @ (hc_mult*H, 24)`` fp32
  matvec that cuBLAS served with a SIMT (non-tensor-core) SGEMM at ~13 us --
  1.57 MB of weights, so a ~0.2 us problem at HBM speed;
* the residual-stream reduction and expansion.

Arithmetic matches the reference: fp32 accumulation, same operation order, and
a split-K reduction with fixed summation order so results are reproducible run
to run. ``tests/test_mhc.py`` pins all of it against a plain-PyTorch oracle.

vLLM and SGLang fuse the same region with TileLang kernels; this is the Triton
equivalent, kept in-tree so the model has no new build dependency.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _split_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    mixes_row_stride,
    eps,
    sumsq_ptr,
    inv_k,
    norm_eps,
    ITERS: tl.constexpr,
    HC: tl.constexpr,
    RMS: tl.constexpr,
):
    token = tl.program_id(0)
    src = mixes_ptr + token.to(tl.int64) * mixes_row_stride

    # When the caller hands over raw mix logits plus the RMS norm's sum of
    # squares, scale here instead of paying a separate elementwise launch.
    rms = 1.0
    if RMS:
        rms = tl.rsqrt(tl.load(sumsq_ptr + token) * inv_k + norm_eps)

    lanes = tl.arange(0, HC)
    scale0 = tl.load(scale_ptr + 0)
    scale1 = tl.load(scale_ptr + 1)
    scale2 = tl.load(scale_ptr + 2)

    # -- pre / post gates -------------------------------------------------
    pre_logits = tl.load(src + lanes).to(tl.float32) * rms
    pre = tl.sigmoid(pre_logits * scale0 + tl.load(base_ptr + lanes).to(tl.float32))
    tl.store(pre_ptr + token.to(tl.int64) * HC + lanes, pre + eps)

    post_logits = tl.load(src + HC + lanes).to(tl.float32) * rms
    post = tl.sigmoid(
        post_logits * scale1 + tl.load(base_ptr + HC + lanes).to(tl.float32)
    )
    tl.store(post_ptr + token.to(tl.int64) * HC + lanes, 2.0 * post)

    # -- combination matrix ----------------------------------------------
    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    flat = rows * HC + cols
    comb = tl.load(src + 2 * HC + flat).to(tl.float32) * rms * scale2 + tl.load(
        base_ptr + 2 * HC + flat
    ).to(tl.float32)

    # Row softmax, matching torch.softmax's max-subtracted form, then the
    # alternating normalizations. Everything stays in registers.
    comb = comb - tl.max(comb, axis=1, keep_dims=True)
    comb = tl.exp(comb)
    comb = comb / tl.sum(comb, axis=1, keep_dims=True)
    comb = comb + eps
    comb = comb / (tl.sum(comb, axis=0, keep_dims=True) + eps)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1, keep_dims=True) + eps)
        comb = comb / (tl.sum(comb, axis=0, keep_dims=True) + eps)

    tl.store(comb_ptr + token.to(tl.int64) * HC * HC + flat, comb)


def hc_split_sinkhorn_fused(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    sumsq: torch.Tensor | None = None,
    reduction_width: int = 0,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch equivalent of the reference gate split + Sinkhorn."""
    leading = mixes.shape[:-1]
    flat = mixes.reshape(-1, mixes.shape[-1])
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    tokens = flat.shape[0]
    pre = torch.empty(tokens, hc_mult, dtype=torch.float32, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty(
        tokens, hc_mult, hc_mult, dtype=torch.float32, device=mixes.device
    )
    if tokens:
        _split_sinkhorn_kernel[(tokens,)](
            flat,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            flat.stride(0),
            eps,
            sumsq if sumsq is not None else flat,
            (1.0 / reduction_width) if reduction_width else 0.0,
            norm_eps,
            ITERS=sinkhorn_iters,
            HC=hc_mult,
            RMS=sumsq is not None,
            num_warps=1,
        )
    return (
        pre.view(*leading, hc_mult),
        post.view(*leading, hc_mult),
        comb.view(*leading, hc_mult, hc_mult),
    )


# Split-K width for the mix projection; see ``_mhc_mix_kernel``.
_SPLIT = 8


@triton.jit
def _mhc_mix_kernel(
    x_ptr,            # (tokens, K) bfloat16, K = hc_mult * hidden
    fn_ptr,           # (MIX, K) float32
    part_ptr,         # (tokens, MIX+1, SPLIT) float32 partials
    K,
    chunk,
    MIX: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    s = tl.program_id(2)

    # Only 25 outputs, so without splitting the reduction the grid leaves most
    # of the GPU idle while each program streams 64 KB of weights.
    lo = s * chunk
    hi = tl.minimum(lo + chunk, K)

    x_base = x_ptr + token * K
    acc = 0.0
    if j == MIX:
        # The RMS norm's reduction, folded into the same sweep over x.
        for k0 in range(lo, hi, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < hi
            v = tl.load(x_base + offs, mask=m, other=0.0).to(tl.float32)
            acc += tl.sum(v * v)
    else:
        w_base = fn_ptr + j * K
        for k0 in range(lo, hi, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            m = offs < hi
            v = tl.load(x_base + offs, mask=m, other=0.0).to(tl.float32)
            w = tl.load(w_base + offs, mask=m, other=0.0)
            acc += tl.sum(v * w)
    tl.store(part_ptr + (token * (MIX + 1) + j) * SPLIT + s, acc)


@triton.jit
def _mhc_mix_reduce_kernel(
    part_ptr,         # (tokens, MIX+1, SPLIT) float32
    mix_ptr,
    sumsq_ptr,
    MIX: tl.constexpr,
    SPLIT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    offs = tl.arange(0, SPLIT)
    # Fixed summation order, so the result is reproducible run to run.
    v = tl.sum(tl.load(part_ptr + (token * (MIX + 1) + j) * SPLIT + offs))
    if j == MIX:
        tl.store(sumsq_ptr + token, v)
    else:
        tl.store(mix_ptr + token * MIX + j, v)


@triton.jit
def _mhc_pre_combine_kernel(
    x_ptr,            # (tokens, HC, H) bfloat16
    pre_ptr,          # (tokens, HC) float32
    out_ptr,          # (tokens, H) bfloat16
    H,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for m in tl.static_range(HC):
        w = tl.load(pre_ptr + token * HC + m)
        v = tl.load(x_ptr + token * HC * H + m * H + offs, mask=mask, other=0.0)
        acc += w * v.to(tl.float32)
    tl.store(out_ptr + token * H + offs, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _mhc_post_kernel(
    x_ptr,            # (tokens, H) bfloat16
    res_ptr,          # (tokens, HC, H) bfloat16
    post_ptr,         # (tokens, HC) float32
    comb_ptr,         # (tokens, HC, HC) float32
    out_ptr,          # (tokens, HC, H) bfloat16
    H,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    xv = tl.load(x_ptr + token * H + offs, mask=mask, other=0.0).to(tl.float32)
    res_base = res_ptr + token * HC * H
    for n in tl.static_range(HC):
        acc = tl.load(post_ptr + token * HC + n) * xv
        for m in tl.static_range(HC):
            # The reference sums over comb's *first* axis (``dim=-3`` of the
            # broadcast product), so the contraction reads comb[m, n].
            c = tl.load(comb_ptr + token * HC * HC + m * HC + n)
            r = tl.load(res_base + m * H + offs, mask=mask, other=0.0)
            acc += c * r.to(tl.float32)
        tl.store(out_ptr + token * HC * H + n * H + offs,
                 acc.to(tl.bfloat16), mask=mask)


def mhc_mix_and_sumsq(
    x_flat: torch.Tensor, hc_fn: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw mix logits and the RMS norm's sum of squares, in one sweep."""
    tokens, K = x_flat.shape
    mix = hc_fn.shape[0]
    out = torch.empty(tokens, mix, device=x_flat.device, dtype=torch.float32)
    sumsq = torch.empty(tokens, device=x_flat.device, dtype=torch.float32)
    split = _SPLIT
    part = torch.empty(
        tokens, mix + 1, split, device=x_flat.device, dtype=torch.float32
    )
    chunk = triton.cdiv(K, split)
    _mhc_mix_kernel[(tokens, mix + 1, split)](
        x_flat, hc_fn, part, K, chunk,
        MIX=mix, BLOCK_K=1024, SPLIT=split, num_warps=8,
    )
    _mhc_mix_reduce_kernel[(tokens, mix + 1)](
        part, out, sumsq, MIX=mix, SPLIT=split, num_warps=1
    )
    return out, sumsq


def mhc_pre_combine(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """``sum_m pre[m] * x[m]`` over the residual-stream axis."""
    tokens, hc, hidden = x.shape
    out = torch.empty(tokens, hidden, device=x.device, dtype=x.dtype)
    block = min(triton.next_power_of_2(hidden), 1024)
    _mhc_pre_combine_kernel[(tokens, triton.cdiv(hidden, block))](
        x, pre, out, hidden, HC=hc, BLOCK_H=block, num_warps=4
    )
    return out


def mhc_post_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """``post[n] * x + sum_m comb[n, m] * residual[m]`` for every stream."""
    tokens, hc, hidden = residual.shape
    out = torch.empty_like(residual)
    block = min(triton.next_power_of_2(hidden), 1024)
    _mhc_post_kernel[(tokens, triton.cdiv(hidden, block))](
        x, residual, post, comb, out, hidden,
        HC=hc, BLOCK_H=block, num_warps=4,
    )
    return out


__all__ = [
    "hc_split_sinkhorn_fused",
    "mhc_mix_and_sumsq",
    "mhc_pre_combine",
    "mhc_post_fused",
]
