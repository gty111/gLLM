# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from gllm.layers.ops.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    T,  # runtime: the row count changes with every ragged batch, and a
        # ``tl.constexpr`` here made each new token count a fresh Triton
        # specialization (~640 ms of JIT on the first launch).  It only bounds
        # the block pointers below, which accept runtime extents.
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def l2norm_fwd_4d_strided_kernel(
    x,
    y,
    eps,
    T,  # runtime: token count. Only feeds ``T * H`` below, and specializing on
        # it costs a Triton compile per distinct ragged batch size.
    H: tl.constexpr,
    D: tl.constexpr,
    SX0,
    SX1,
    SX2,
    R,  # runtime: total logical rows (``T * H`` per batch entry). Only feeds
        # the ``rows < R`` mask.
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """L2-normalize logical rows of a last-dimension-contiguous 4D view."""
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    row_mask = rows < R
    col_mask = cols < D

    rows_per_batch = T * H
    batch = rows // rows_per_batch
    batch_row = rows - batch * rows_per_batch
    token = batch_row // H
    head = batch_row - token * H
    x_offsets = batch * SX0 + token * SX1 + head * SX2

    b_x = tl.load(
        x + x_offsets[:, None] + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    tl.store(
        y + rows[:, None] * D + cols[None, :],
        b_y,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    use_4d_strided_kernel = (
        x.ndim == 4
        and x.shape[-1] <= 512
        and x.stride(-1) == 1
        and not x.is_contiguous()
    )
    if not use_4d_strided_kernel:
        # Preserve the historical compact-input path for all other layouts.
        x = x.contiguous().view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty(x_shape_og, dtype=x.dtype, device=x.device)
    else:
        y = torch.empty(x_shape_og, dtype=output_dtype, device=x.device)
    assert y.stride(-1) == 1
    R, D = x.numel() // x.shape[-1], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:

        def grid(meta):
            return (triton.cdiv(R, meta["BT"]),)

        if use_4d_strided_kernel:
            l2norm_fwd_4d_strided_kernel[grid](
                x,
                y,
                eps,
                T=x.shape[1],
                H=x.shape[2],
                D=D,
                SX0=x.stride(0),
                SX1=x.stride(1),
                SX2=x.stride(2),
                R=R,
                BT=16,
                BD=BD,
                num_warps=8,
                num_stages=3,
            )
        else:
            l2norm_fwd_kernel[grid](
                x,
                y,
                eps,
                T=R,
                D=D,
                BD=BD,
                BT=16,
                num_warps=8,
                num_stages=3,
            )
    else:
        l2norm_fwd_kernel1[(R,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        return l2norm_fwd(x, eps, output_dtype)


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):

    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
