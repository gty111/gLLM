"""Numerically conservative Gemma RMSNorm dispatch and kernels.

Small, common shapes use bitwise-verified fused CUDA/Triton kernels. Other
shapes retain the established ``aten::mean`` FP32 reduction tree and fuse only
the surrounding casts and pointwise work. Every path is checked against the
same PyTorch reference in ``tests/test_gemma_rmsnorm_exact.py``.
"""

import torch
import triton
import triton.language as tl


_CUDA_EXTENSION_MAX_ROWS = 256
_TRITON_VEC4_WIDTH = 128


@triton.jit
def _gemma_rms_norm_vec4_kernel(
    x,
    residual,
    weight,
    out,
    eps: tl.constexpr,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_residual_row: tl.constexpr,
    stride_out_row: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_THREADS: tl.constexpr,
):
    row = tl.program_id(0)
    thread = tl.arange(0, BLOCK_THREADS)
    lane = thread % 32
    reduction_lane = thread < 32
    acc0 = tl.zeros((BLOCK_THREADS,), tl.float32)
    acc1 = tl.zeros((BLOCK_THREADS,), tl.float32)
    acc2 = tl.zeros((BLOCK_THREADS,), tl.float32)
    acc3 = tl.zeros((BLOCK_THREADS,), tl.float32)

    for block_idx in tl.range(
        0, n_cols // 128, num_stages=1, loop_unroll_factor=1
    ):
        cols = block_idx * 128 + lane * 4
        x_ptrs = x + row * stride_x_row + cols
        value0 = tl.load(x_ptrs, mask=reduction_lane, other=0.0).to(tl.float32)
        value1 = tl.load(x_ptrs + 1, mask=reduction_lane, other=0.0).to(tl.float32)
        value2 = tl.load(x_ptrs + 2, mask=reduction_lane, other=0.0).to(tl.float32)
        value3 = tl.load(x_ptrs + 3, mask=reduction_lane, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual_ptrs = residual + row * stride_residual_row + cols
            value0 += tl.load(
                residual_ptrs, mask=reduction_lane, other=0.0
            ).to(tl.float32)
            value1 += tl.load(
                residual_ptrs + 1, mask=reduction_lane, other=0.0
            ).to(tl.float32)
            value2 += tl.load(
                residual_ptrs + 2, mask=reduction_lane, other=0.0
            ).to(tl.float32)
            value3 += tl.load(
                residual_ptrs + 3, mask=reduction_lane, other=0.0
            ).to(tl.float32)
            value0 = value0.to(
                residual.dtype.element_ty, fp_downcast_rounding="rtne"
            )
            value1 = value1.to(
                residual.dtype.element_ty, fp_downcast_rounding="rtne"
            )
            value2 = value2.to(
                residual.dtype.element_ty, fp_downcast_rounding="rtne"
            )
            value3 = value3.to(
                residual.dtype.element_ty, fp_downcast_rounding="rtne"
            )
            tl.store(residual_ptrs, value0, mask=reduction_lane)
            tl.store(residual_ptrs + 1, value1, mask=reduction_lane)
            tl.store(residual_ptrs + 2, value2, mask=reduction_lane)
            tl.store(residual_ptrs + 3, value3, mask=reduction_lane)
            value0 = value0.to(tl.float32)
            value1 = value1.to(tl.float32)
            value2 = value2.to(tl.float32)
            value3 = value3.to(tl.float32)
        acc0 += value0 * value0
        acc1 += value1 * value1
        acc2 += value2 * value2
        acc3 += value3 * value3

    thread_sum = ((acc0 + acc1) + acc2) + acc3
    warp_sums = tl.sum(
        tl.reshape(thread_sum, (BLOCK_THREADS // 32, 32)), axis=1
    )
    reduced = tl.sum(warp_sums, axis=0)
    mean_factor = tl.full((), 1.0 / n_cols, tl.float32)
    variance = tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=(reduced, mean_factor),
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    inv_rms = tl.rsqrt(variance + eps)

    for block_idx in tl.range(
        0,
        tl.cdiv(n_cols, BLOCK_THREADS),
        num_stages=1,
        loop_unroll_factor=1,
    ):
        cols = block_idx * BLOCK_THREADS + thread
        mask = cols < n_cols
        source = residual if HAS_RESIDUAL else x
        source_stride = stride_residual_row if HAS_RESIDUAL else stride_x_row
        value = tl.load(
            source + row * source_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
        tl.store(
            out + row * stride_out_row + cols,
            value * inv_rms * scale,
            mask=mask,
        )


def _dispatch_fused_kernel(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    epsilon: float,
    has_residual: bool,
) -> bool:
    """Run a bitwise-verified fused backend when the shape is supported."""
    n_cols = input.shape[-1]
    if n_cols < _TRITON_VEC4_WIDTH or n_cols % _TRITON_VEC4_WIDTH != 0:
        return False
    n_rows = input.numel() // n_cols
    input_2d = input.reshape(n_rows, n_cols)
    residual_2d = residual.reshape(n_rows, n_cols)
    out_2d = out.reshape(n_rows, n_cols)
    if (
        input.dtype == torch.bfloat16
        and n_cols > _TRITON_VEC4_WIDTH
        and n_rows <= _CUDA_EXTENSION_MAX_ROWS
    ):
        from gllm.layers.ops.gemma_rmsnorm_cuda import gemma_rmsnorm_bf16

        gemma_rmsnorm_bf16(
            input_2d,
            residual_2d,
            weight,
            out_2d,
            epsilon,
            has_residual,
        )
        return True
    if (
        input.dtype not in (torch.bfloat16, torch.float16)
        or n_cols != _TRITON_VEC4_WIDTH
    ):
        return False
    block_threads = 32
    _gemma_rms_norm_vec4_kernel[(n_rows,)](
        input_2d,
        residual_2d,
        weight,
        out_2d,
        epsilon,
        n_cols,
        input_2d.stride(0),
        residual_2d.stride(0),
        out_2d.stride(0),
        HAS_RESIDUAL=has_residual,
        BLOCK_THREADS=block_threads,
        num_warps=block_threads // 32,
    )
    return True


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
def _square_strided_3d_to_fp32_kernel(
    x,
    work,
    n_cols: tl.constexpr,
    n_heads: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_head: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize squares from a token/head view without compacting it."""
    row = tl.program_id(0)
    token = row // n_heads
    head = row - token * n_heads
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    value = tl.load(
        x + token * stride_x_token + head * stride_x_head + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(work + row * n_cols + cols, value * value, mask=mask)


@triton.jit
def _add_square_to_fp32_kernel(
    x,
    residual,
    work,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_residual_row: tl.constexpr,
    INPUT_IS_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold the residual and materialize its exact rounded FP32 square."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    value = tl.load(
        x + row * stride_x_row + cols, mask=mask, other=0.0
    ).to(tl.float32)
    prior = tl.load(
        residual + row * stride_residual_row + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if INPUT_IS_FP32:
        folded = prior + value
    else:
        folded = (prior + value).to(
            residual.dtype.element_ty, fp_downcast_rounding="rtne"
        )
    tl.store(
        residual + row * stride_residual_row + cols,
        folded,
        mask=mask,
    )
    folded_fp32 = folded.to(tl.float32)
    tl.store(
        work + row * n_cols + cols,
        folded_fp32 * folded_fp32,
        mask=mask,
    )


@triton.jit
def _normalize_gemma_scale_kernel(
    x,
    variance,
    weight,
    out,
    eps: tl.constexpr,
    n_cols: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_out_row: tl.constexpr,
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
    scale = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    tl.store(
        out + row * stride_out_row + cols,
        normalized * scale,
        mask=mask,
    )


@triton.jit
def _normalize_strided_3d_gemma_scale_kernel(
    x,
    variance,
    weight,
    out,
    eps: tl.constexpr,
    n_cols: tl.constexpr,
    n_heads: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_head: tl.constexpr,
    stride_out_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // n_heads
    head = row - token * n_heads
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    value = tl.load(
        x + token * stride_x_token + head * stride_x_head + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    var = tl.load(variance + row).to(tl.float32)
    normalized = value * tl.rsqrt(var + eps)
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
    out_2d = out.reshape(n_rows, n_cols)

    # Qwen3.5's gated QKV projection exposes Q/K as [token, head, dim]
    # views whose feature dimension is contiguous but whose token stride also
    # spans the neighboring gate/K/V fields.  Flattening those views with
    # reshape first materializes a large BF16 copy.  Read the two leading
    # strides directly while preserving the exact same compact FP32 work
    # matrix and aten::mean reduction used by the established path.  Small
    # row counts stay on the single-launch CUDA kernel, where compacting is
    # cheaper than replacing that kernel with the multi-launch fallback.
    strided_3d = (
        input.ndim == 3
        and not input.is_contiguous()
        and out.is_contiguous()
        and n_rows > 256
    )
    if strided_3d:
        work = torch.empty(
            (n_rows, n_cols), dtype=torch.float32, device=input.device
        )
        block = min(1024, triton.next_power_of_2(n_cols))
        grid = (n_rows, triton.cdiv(n_cols, block))
        n_heads = input.shape[-2]
        _square_strided_3d_to_fp32_kernel[grid](
            input,
            work,
            n_cols,
            n_heads,
            input.stride(-3),
            input.stride(-2),
            BLOCK=block,
        )
        variance = work.mean(dim=-1, keepdim=True)
        _normalize_strided_3d_gemma_scale_kernel[grid](
            input,
            variance,
            weight,
            out_2d,
            epsilon,
            n_cols,
            n_heads,
            input.stride(-3),
            input.stride(-2),
            out_2d.stride(0),
            BLOCK=block,
        )
        return

    input_2d = input.reshape(n_rows, n_cols)
    if _dispatch_fused_kernel(input, input, weight, out, epsilon, False):
        return
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
    _normalize_gemma_scale_kernel[grid](
        input_2d,
        variance,
        weight,
        out_2d,
        epsilon,
        n_cols,
        input_2d.stride(0),
        out_2d.stride(0),
        BLOCK=block,
    )


def gemma_fused_add_rms_norm_reference_reduction(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """In-place residual fold plus bitwise-reference Gemma RMSNorm."""
    if input.shape != residual.shape:
        raise ValueError(
            f"input/residual shape mismatch: {input.shape} vs {residual.shape}"
        )
    if input.shape[-1] != weight.numel():
        raise ValueError(
            f"input/weight shape mismatch: {input.shape} vs {weight.shape}"
        )
    if input.dtype != residual.dtype:
        raise ValueError(
            f"input/residual dtype mismatch: {input.dtype} vs {residual.dtype}"
        )
    if input.stride(-1) != 1 or residual.stride(-1) != 1:
        raise ValueError("Gemma RMSNorm requires a contiguous feature dimension")

    n_cols = input.shape[-1]
    n_rows = input.numel() // n_cols
    input_2d = input.reshape(n_rows, n_cols)
    residual_2d = residual.reshape(n_rows, n_cols)
    if _dispatch_fused_kernel(input, residual, weight, input, epsilon, True):
        return
    work = torch.empty((n_rows, n_cols), dtype=torch.float32, device=input.device)
    block = min(1024, triton.next_power_of_2(n_cols))
    grid = (n_rows, triton.cdiv(n_cols, block))

    _add_square_to_fp32_kernel[grid](
        input_2d,
        residual_2d,
        work,
        n_cols,
        input_2d.stride(0),
        residual_2d.stride(0),
        INPUT_IS_FP32=input.dtype == torch.float32,
        BLOCK=block,
    )
    variance = work.mean(dim=-1, keepdim=True)
    _normalize_gemma_scale_kernel[grid](
        residual_2d,
        variance,
        weight,
        input_2d,
        epsilon,
        n_cols,
        residual_2d.stride(0),
        input_2d.stride(0),
        BLOCK=block,
    )
