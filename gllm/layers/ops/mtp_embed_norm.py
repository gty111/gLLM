"""Exact fused input preparation for Qwen3.8's MTP block."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mtp_embed_hidden_square_kernel(
    token_ids,
    embed_table,
    previous_hidden,
    work,
    table_stride,
    hidden_stride,
    num_rows,
    hidden_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden_size
    token = tl.load(token_ids + row).to(tl.int64)
    embedding = tl.load(
        embed_table + token * table_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    hidden = tl.load(
        previous_hidden + row * hidden_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        work + row * hidden_size + cols,
        embedding * embedding,
        mask=mask,
    )
    tl.store(
        work + (num_rows + row) * hidden_size + cols,
        hidden * hidden,
        mask=mask,
    )


@triton.jit
def _mtp_embed_hidden_normalize_kernel(
    token_ids,
    embed_table,
    previous_hidden,
    variance,
    embedding_weight,
    hidden_weight,
    out,
    table_stride,
    hidden_stride,
    out_stride,
    num_rows,
    eps: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden_size
    token = tl.load(token_ids + row).to(tl.int64)
    embedding = tl.load(
        embed_table + token * table_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    hidden = tl.load(
        previous_hidden + row * hidden_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    embedding_var = tl.load(variance + row).to(tl.float32)
    hidden_var = tl.load(variance + num_rows + row).to(tl.float32)
    embedding_scale = (
        tl.load(embedding_weight + cols, mask=mask, other=0.0).to(tl.float32)
        + 1.0
    )
    hidden_scale = (
        tl.load(hidden_weight + cols, mask=mask, other=0.0).to(tl.float32)
        + 1.0
    )
    embedding = embedding * tl.rsqrt(embedding_var + eps) * embedding_scale
    hidden = hidden * tl.rsqrt(hidden_var + eps) * hidden_scale
    tl.store(out + row * out_stride + cols, embedding, mask=mask)
    tl.store(
        out + row * out_stride + hidden_size + cols,
        hidden,
        mask=mask,
    )


def fused_mtp_embed_hidden_gemma_norm(
    token_ids: torch.Tensor,
    embed_table: torch.Tensor,
    previous_hidden: torch.Tensor,
    embedding_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Gather, exactly normalize, and concatenate MTP embedding/hidden rows.

    The FP32 squares for both inputs share one launch and one ``aten::mean``
    reduction over ``[2N, H]``.  Keeping that reduction is what makes this
    path bitwise-identical to the conservative Gemma RMSNorm implementation;
    the final normalization writes directly into the ``[N, 2H]`` FC input.
    """
    ids = token_ids.reshape(-1)
    if previous_hidden.ndim != 2 or embed_table.ndim != 2:
        raise ValueError((previous_hidden.shape, embed_table.shape))
    num_rows, hidden_size = previous_hidden.shape
    if ids.numel() != num_rows or embed_table.shape[1] != hidden_size:
        raise ValueError((ids.shape, previous_hidden.shape, embed_table.shape))
    if embedding_weight.shape != (hidden_size,) or hidden_weight.shape != (
        hidden_size,
    ):
        raise ValueError(
            (embedding_weight.shape, hidden_weight.shape, hidden_size)
        )
    if embed_table.dtype != previous_hidden.dtype:
        raise ValueError(
            f"embedding/hidden dtype mismatch: {embed_table.dtype} vs "
            f"{previous_hidden.dtype}"
        )
    if not ids.is_cuda or not embed_table.is_cuda or not previous_hidden.is_cuda:
        raise ValueError("fused MTP embed/norm requires CUDA tensors")
    if (
        embed_table.stride(-1) != 1
        or previous_hidden.stride(-1) != 1
        or not ids.is_contiguous()
    ):
        raise ValueError("fused MTP embed/norm requires contiguous feature rows")

    out = torch.empty(
        (num_rows, 2 * hidden_size),
        dtype=previous_hidden.dtype,
        device=previous_hidden.device,
    )
    if num_rows == 0:
        return out
    work = torch.empty(
        (2 * num_rows, hidden_size),
        dtype=torch.float32,
        device=previous_hidden.device,
    )
    block = min(1024, triton.next_power_of_2(hidden_size))
    grid = (num_rows, triton.cdiv(hidden_size, block))
    _mtp_embed_hidden_square_kernel[grid](
        ids,
        embed_table,
        previous_hidden,
        work,
        embed_table.stride(0),
        previous_hidden.stride(0),
        num_rows,
        hidden_size=hidden_size,
        BLOCK=block,
    )
    variance = work.mean(dim=-1, keepdim=True)
    _mtp_embed_hidden_normalize_kernel[grid](
        ids,
        embed_table,
        previous_hidden,
        variance,
        embedding_weight,
        hidden_weight,
        out,
        embed_table.stride(0),
        previous_hidden.stride(0),
        out.stride(0),
        num_rows,
        eps=eps,
        hidden_size=hidden_size,
        BLOCK=block,
    )
    return out
