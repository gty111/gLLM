"""Multi-stream hyper-connection (mHC) reference operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fused_mhc_usable(x: torch.Tensor, hc_fn: torch.Tensor) -> bool:
    """The fused path needs contiguous bf16 streams and fp32 mix weights."""
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.is_contiguous()
        and hc_fn.dtype is torch.float32
        and hc_fn.is_contiguous()
    )


def hc_split_sinkhorn_reference(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split mHC parameters and apply the reference Sinkhorn iterations."""
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.dtype != torch.float32:
        raise TypeError(f"mHC mixes must be float32, got {mixes.dtype}")
    if mixes.shape[-1] != mix_hc:
        raise ValueError(
            f"last mixes dimension must be {mix_hc}, got {mixes.shape[-1]}"
        )
    if tuple(hc_scale.shape) != (3,) or tuple(hc_base.shape) != (mix_hc,):
        raise ValueError(
            f"hc_scale/hc_base must have shapes (3,) and ({mix_hc},)"
        )
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be positive")

    pre_logits, post_logits, comb_logits = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        post_logits * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = (
        comb_logits * hc_scale[2] + hc_base[2 * hc_mult :]
    ).view(*mixes.shape[:-1], hc_mult, hc_mult)

    # The first row normalization is softmax, followed by epsilon addition;
    # subsequent iterations are plain alternating row/column normalization.
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def mhc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce ``hc_mult`` residual streams to one layer input."""
    if x.ndim < 3 or x.shape[-2] != hc_mult:
        raise ValueError(
            f"x must end in [hc_mult, hidden]=[{hc_mult}, H], got {x.shape}"
        )
    shape, dtype = x.shape, x.dtype
    width = shape[-2] * shape[-1]
    expected_fn = ((2 + hc_mult) * hc_mult, width)
    if tuple(hc_fn.shape) != expected_fn:
        raise ValueError(f"hc_fn must have shape {expected_fn}, got {hc_fn.shape}")

    if _fused_mhc_usable(x, hc_fn):
        # Three launches instead of ~10, and the mix projection avoids cuBLAS'
        # SIMT fp32 SGEMM -- see gllm/layers/ops/deepseek_v4/mhc.py.
        from gllm.layers.ops.deepseek_v4 import (
            hc_split_sinkhorn_fused,
            mhc_mix_and_sumsq,
            mhc_pre_combine,
        )

        flat = x.reshape(-1, width)
        mixes, sumsq = mhc_mix_and_sumsq(flat, hc_fn)
        pre, post, comb = hc_split_sinkhorn_fused(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=hc_eps,
            sumsq=sumsq,
            reduction_width=width,
            norm_eps=norm_eps,
        )
        layer_input = mhc_pre_combine(
            x.reshape(-1, hc_mult, shape[-1]), pre
        ).view(*shape[:-2], shape[-1])
        lead = shape[:-2]
        return (
            layer_input,
            post.view(*lead, hc_mult),
            comb.view(*lead, hc_mult, hc_mult),
        )

    x_fp32 = x.flatten(-2).float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_fp32, hc_fn) * inv_rms
    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )
    layer_input = torch.sum(
        pre.unsqueeze(-1) * x_fp32.view(shape), dim=-2
    )
    return layer_input.to(dtype), post, comb


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand a layer output back into the residual stream dimension."""
    if residual.shape[:-2] != x.shape[:-1] or residual.shape[-1] != x.shape[-1]:
        raise ValueError("x and residual leading/hidden dimensions do not match")
    hc_mult = residual.shape[-2]
    if tuple(post.shape) != (*x.shape[:-1], hc_mult):
        raise ValueError("post shape does not match x and residual")
    if tuple(comb.shape) != (*x.shape[:-1], hc_mult, hc_mult):
        raise ValueError("comb shape does not match x and residual")
    if (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.is_contiguous()
        and residual.is_contiguous()
        and residual.dtype is torch.bfloat16
    ):
        from gllm.layers.ops.deepseek_v4 import mhc_post_fused

        hidden = x.shape[-1]
        return mhc_post_fused(
            x.reshape(-1, hidden),
            residual.reshape(-1, hc_mult, hidden),
            post.reshape(-1, hc_mult).float(),
            comb.reshape(-1, hc_mult, hc_mult).float(),
        ).view(*residual.shape)

    output = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
    )
    return output.to(x.dtype)


def mhc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    hc_mult: int = 4,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    """Collapse the final mHC residual streams into the LM-head input."""
    if x.ndim < 3 or x.shape[-2] != hc_mult:
        raise ValueError(
            f"x must end in [hc_mult, hidden]=[{hc_mult}, H], got {x.shape}"
        )
    shape, dtype = x.shape, x.dtype
    flat = x.flatten(-2).float()
    expected_fn = (hc_mult, flat.shape[-1])
    if tuple(hc_fn.shape) != expected_fn:
        raise ValueError(f"hc_fn must have shape {expected_fn}, got {hc_fn.shape}")
    if tuple(hc_scale.shape) != (1,) or tuple(hc_base.shape) != (hc_mult,):
        raise ValueError(
            f"hc_scale/hc_base must have shapes (1,) and ({hc_mult},)"
        )
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, hc_fn) * inv_rms
    weights = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return torch.sum(weights.unsqueeze(-1) * flat.view(shape), dim=-2).to(dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the mHC gates and Sinkhorn-normalize the combination matrix.

    Dispatches to the fused kernel on CUDA. The reference form is ~170 launches
    over a 4x4 matrix, run twice per decoder layer; on 43 layers that was the
    single largest cost in a single-request decode step, larger than every GEMM
    combined. The fused kernel is one launch and normalizes in a different
    summation order, so it agrees to ~1e-6 relative rather than bitwise -- the
    same trade vLLM and SGLang make with their TileLang mHC kernels.
    """
    if (
        mixes.is_cuda
        and hc_scale.is_cuda
        and hc_base.is_cuda
        and mixes.dtype is torch.float32
        and hc_mult > 1
        and (hc_mult & (hc_mult - 1)) == 0
        and sinkhorn_iters >= 1
        and mixes.shape[-1] == (2 + hc_mult) * hc_mult
    ):
        from gllm.layers.ops.deepseek_v4.mhc import (
            hc_split_sinkhorn_fused,
        )

        return hc_split_sinkhorn_fused(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )
    return hc_split_sinkhorn_reference(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )


__all__ = [
    "hc_split_sinkhorn",
    "hc_split_sinkhorn_reference",
    "mhc_head",
    "mhc_post",
    "mhc_pre",
]
