"""Numerically conservative Gemma RMSNorm kernels.

The reduction intentionally remains an ``aten::mean`` operation.  Besides
being well tuned, that preserves the reduction tree used by the established
PyTorch reference path.  The surrounding casts and pointwise operations are
collapsed into small Triton kernels, avoiding several launches without
changing the value presented to the reduction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _square_to_fp32_kernel(
    x,
    work,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    value = tl.load(x + row * stride_x_row + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(work + row * n_cols + cols, value * value, mask=mask)


@triton.jit
def _normalize_fp32_kernel(
    x,
    variance,
    work,
    eps: tl.constexpr,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    value = tl.load(x + row * stride_x_row + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    var = tl.load(variance + row).to(tl.float32)
    normalized = value * tl.rsqrt(var + eps)
    tl.store(work + row * n_cols + cols, normalized, mask=mask)


@triton.jit
def _gemma_scale_kernel(
    work,
    weight,
    out,
    n_cols: tl.constexpr,
    stride_out_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    normalized = tl.load(work + row * n_cols + cols, mask=mask, other=0.0)
    scale = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    tl.store(
        out + row * stride_out_row + cols,
        normalized * scale,
        mask=mask,
    )


def gemma_rms_norm_reference_reduction(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Gemma RMSNorm with the PyTorch-reference FP32 reduction order.

    Inputs may have any leading dimensions, but the feature dimension must be
    contiguous.  This is the same layout contract as the other RMSNorm
    backends used by gLLM.
    """
    if input.shape != out.shape:
        raise ValueError(f"input/out shape mismatch: {input.shape} vs {out.shape}")
    if input.shape[-1] != weight.numel():
        raise ValueError(f"input/weight shape mismatch: {input.shape} vs {weight.shape}")
    if input.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("Gemma RMSNorm requires a contiguous feature dimension")

    n_cols = input.shape[-1]
    n_rows = input.numel() // n_cols
    input_2d = input.reshape(n_rows, n_cols)
    out_2d = out.reshape(n_rows, n_cols)
    work = torch.empty((n_rows, n_cols), dtype=torch.float32, device=input.device)
    block = min(1024, triton.next_power_of_2(n_cols))
    grid = (n_rows, triton.cdiv(n_cols, block))

    _square_to_fp32_kernel[grid](
        input_2d,
        work,
        n_cols,
        input_2d.stride(0),
        BLOCK=block,
    )
    # Keep this reduction as aten::mean: model correctness was established
    # against its FP32 reduction tree on SM100.
    variance = work.mean(dim=-1, keepdim=True)
    _normalize_fp32_kernel[grid](
        input_2d,
        variance,
        work,
        epsilon,
        n_cols,
        input_2d.stride(0),
        BLOCK=block,
    )
    _gemma_scale_kernel[grid](
        work,
        weight,
        out_2d,
        n_cols,
        out_2d.stride(0),
        BLOCK=block,
    )
