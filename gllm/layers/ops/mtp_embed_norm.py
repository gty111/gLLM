"""Exact fused input preparation for Qwen3.8's MTP block."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mtp_embed_hidden_gemma_norm_kernel(
    ids_ptr,
    table_ptr,
    hidden_ptr,
    embedding_weight_ptr,
    hidden_weight_ptr,
    out_ptr,
    table_row_stride,
    hidden_row_stride,
    out_row_stride,
    eps,
    mean_factor,
    hidden_size,
    BLOCK: tl.constexpr,
):
    """Gather one embedding row, norm it and the hidden row, write both.

    The reduction is written exactly as in
    ``gllm.layers.ops.gemma_rmsnorm`` -- same ``tl.sum`` over the same
    ``next_power_of_2`` block, same ``* mean_factor + eps`` order -- so this
    fused path stays bitwise-identical to calling that norm twice. That
    equivalence is what ``tests/test_mtp_embed_norm.py`` asserts.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden_size

    token = tl.load(ids_ptr + row)
    embedding = tl.load(
        table_ptr + token * table_row_stride + offs, mask=mask, other=0.0
    ).to(tl.float32)
    hidden = tl.load(
        hidden_ptr + row * hidden_row_stride + offs, mask=mask, other=0.0
    ).to(tl.float32)

    inv_embedding = 1.0 / tl.sqrt(
        tl.sum(embedding * embedding, axis=0) * mean_factor + eps
    )
    inv_hidden = 1.0 / tl.sqrt(
        tl.sum(hidden * hidden, axis=0) * mean_factor + eps
    )
    embedding_gain = tl.load(
        embedding_weight_ptr + offs, mask=mask, other=0.0
    ).to(tl.float32) + 1.0
    hidden_gain = tl.load(
        hidden_weight_ptr + offs, mask=mask, other=0.0
    ).to(tl.float32) + 1.0

    base = out_ptr + row * out_row_stride
    tl.store(base + offs, embedding * inv_embedding * embedding_gain, mask=mask)
    tl.store(
        base + hidden_size + offs,
        hidden * inv_hidden * hidden_gain,
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
    """Gather, normalize and concatenate MTP embedding/hidden rows.

    One launch writes the ``[N, 2H]`` FC input directly. It used to take two
    launches plus an ``aten::mean`` over a ``[2N, H]`` FP32 scratch buffer,
    because matching that reduction tree was how the path was kept
    bitwise-identical to the Gemma RMSNorm implementation; both now share the
    same in-kernel ``tl.sum``, so the scratch buffer and the extra launch are
    gone and the equivalence still holds.
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
    _mtp_embed_hidden_gemma_norm_kernel[(num_rows,)](
        ids,
        embed_table,
        previous_hidden,
        embedding_weight,
        hidden_weight,
        out,
        embed_table.stride(0),
        previous_hidden.stride(0),
        out.stride(0),
        eps,
        1.0 / hidden_size,
        hidden_size,
        BLOCK=triton.next_power_of_2(hidden_size),
    )
    return out
