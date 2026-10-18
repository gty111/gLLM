"""DeepSeek-V4 sparse-attention indexing and numerical reference."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from gllm.layers.quantization.fp8 import per_token_group_fp8_fake_quant_inplace


def serving_max_length(config) -> int:
    """Longest sequence this process can actually serve.

    DeepSeek-V4 checkpoints advertise ``max_position_embeddings=1048576`` while
    a server is normally launched with a far smaller ``--model-max-length``.
    Sizing RoPE tables and CUDA-graph-safe candidate bounds from the checkpoint
    value instead of the runtime one costs hundreds of MB of static tables and
    makes decode-graph capture allocate a worst case that can never occur (a
    32 GB indexer gather at ``max_cuda_graph_bs=512``).  The runner publishes
    the resolved limit as ``config.model_max_length``.
    """
    resolved = getattr(config, "model_max_length", None)
    if resolved:
        return min(int(resolved), int(config.max_position_embeddings))
    return int(config.max_position_embeddings)


@lru_cache(maxsize=8)
def precompute_rope_frequencies(
    dim: int,
    sequence_length: int,
    *,
    original_sequence_length: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the complex RoPE table with DeepSeek-V4's exact YaRN formula."""
    if dim <= 0 or dim % 2:
        raise ValueError(f"RoPE dimension must be positive and even, got {dim}")

    frequencies = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        )
    )
    if original_sequence_length > 0:
        def correction_dim(rotations: int) -> float:
            return dim * math.log(
                original_sequence_length / (rotations * 2 * math.pi)
            ) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (
            torch.arange(dim // 2, dtype=torch.float32, device=device) - low
        ) / (high - low)
        smooth = 1 - ramp.clamp(0, 1)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth

    phases = torch.outer(
        torch.arange(sequence_length, dtype=torch.float32, device=device),
        frequencies,
    )
    return torch.polar(torch.ones_like(phases), phases)


def apply_rope_inplace(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply interleaved complex RoPE with the official BF16 round-trip."""
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"DeepSeek-V4 RoPE input must be bfloat16, got {x.dtype}")
    # Normalize the frequency rank from x's shape alone: materializing
    # ``complex_x`` first would pay for an fp32 copy of x that the fused path
    # never needs.
    pairs = x.shape[-1] // 2
    if x.ndim == 3:
        # Contiguous reference prefills pass [S, D/2], while the packed online
        # path passes one position per batch row as [B, 1, D/2].
        if frequencies.ndim == 2:
            frequencies = frequencies.view(1, x.size(1), pairs)
        elif frequencies.ndim != 3:
            raise ValueError("DeepSeek-V4 RoPE frequencies must be 2D or 3D")
    elif x.ndim == 4:
        if frequencies.ndim == 2:
            frequencies = frequencies.view(1, x.size(1), 1, pairs)
        elif frequencies.ndim == 3:
            frequencies = frequencies.unsqueeze(-2)
        else:
            raise ValueError("DeepSeek-V4 RoPE frequencies must be 2D or 3D")
    else:
        raise ValueError("DeepSeek-V4 RoPE expects a 3D or 4D tensor")

    if _rope_fused_usable(x, frequencies):
        from gllm.layers.ops.deepseek_v4 import apply_rope_inplace_fused

        # The kernel negates the imaginary part itself: ``.conj()`` returns a
        # lazy conjugate view whose flag ``view_as_real`` would not honour.
        return apply_rope_inplace_fused(x, frequencies, inverse=inverse)

    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(complex_x * frequencies).flatten(-2)
    x.copy_(rotated)
    return x


def _rope_fused_usable(x: torch.Tensor, frequencies: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.stride(-1) == 1
        and frequencies.is_cuda
        and frequencies.dtype is torch.complex64
    )


def fp8_fake_quantize_inplace(
    x: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Apply V4's BF16 -> E4M3/E8M0 -> BF16 KV QAT round-trip."""
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"DeepSeek-V4 FP8 QAT input must be bfloat16, got {x.dtype}")
    if x.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {x.shape[-1]} must be divisible by {group_size}"
        )
    return per_token_group_fp8_fake_quant_inplace(
        x,
        group_size,
        round_scale=True,
    )


def window_indices(
    window_size: int,
    batch_size: int,
    sequence_length: int,
    start_pos: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the reference circular sliding-window indices."""
    if start_pos >= window_size - 1:
        cursor = start_pos % window_size
        matrix = torch.cat(
            [
                torch.arange(cursor + 1, window_size, device=device),
                torch.arange(0, cursor + 1, device=device),
            ]
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(sequence_length, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(sequence_length, window_size), device=device
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return (
        matrix.to(torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .contiguous()
    )


def compressed_indices(
    ratio: int,
    batch_size: int,
    sequence_length: int,
    start_pos: int,
    offset: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build causal indices for non-indexed compressed attention (C128)."""
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
    else:
        matrix = torch.arange(sequence_length // ratio, device=device).repeat(
            sequence_length, 1
        )
        valid_count = (
            torch.arange(1, sequence_length + 1, device=device).unsqueeze(1)
            // ratio
        )
        matrix = torch.where(matrix >= valid_count, -1, matrix + offset)
    return (
        matrix.to(torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .contiguous()
    )


def sparse_attention_reference(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference DeepSeek-V4 MQA with sparse indices and attention sinks.

    ``query`` has shape ``[B,S,H,D]``, ``kv`` is the shared MQA cache
    ``[B,T,D]``, and ``indices`` is ``[B,S,K]`` with ``-1`` padding.  A sink
    contributes to the softmax denominator but has zero value, matching the
    official kernel.
    """
    if query.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("query, kv and indices must be 4D, 3D and 3D")
    b, s, h, d = query.shape
    if tuple(kv.shape[:1] + kv.shape[2:]) != (b, d):
        raise ValueError("kv batch/head dimension does not match query")
    if tuple(indices.shape[:2]) != (b, s):
        raise ValueError("indices batch/sequence dimensions do not match query")
    if tuple(sinks.shape) != (h,):
        raise ValueError(f"sinks must have shape ({h},)")
    if kv.shape[1] == 0:
        return torch.zeros_like(query)

    valid = indices >= 0
    safe_indices = indices.clamp(min=0).to(torch.int64)
    batch = torch.arange(b, device=kv.device)[:, None, None]
    selected = kv[batch, safe_indices]  # [B,S,K,D]
    scores = torch.einsum("bshd,bskd->bshk", query.float(), selected.float())
    scores = scores * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)

    sink_logits = sinks.float().view(1, 1, h, 1)
    max_score = torch.maximum(scores.amax(dim=-1, keepdim=True), sink_logits)
    weights = torch.exp(scores - max_score)
    weights = torch.where(valid.unsqueeze(2), weights, 0.0)
    denominator = weights.sum(dim=-1, keepdim=True) + torch.exp(
        sink_logits - max_score
    )
    output = torch.einsum("bshk,bskd->bshd", weights, selected.float())
    return (output / denominator).to(query.dtype)


def sparse_attention_fused(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run SGLang's FlashMLA sparse-prefill kernel on dense workspaces.

    The kernel consumes a single flat KV workspace while the numerical oracle
    above accepts one workspace per batch row.  Rebase every valid index while
    flattening the batch, pad the sparse width to the kernel's 128-entry tile,
    and pad TP-sharded query heads to its 64-head specialization.  Invalid
    ``-1`` entries may occur anywhere in a row and remain invalid after the
    transformation.

    This is the fused compute half of SGLang's paged prefill path: serving code
    gathers each request's paged SWA/compressed cache into ``kv`` before this
    call, then writes newly produced cache rows back to their paged locations.
    Keeping the gather/store policy outside this primitive also makes it
    directly comparable with :func:`sparse_attention_reference`.
    """
    if query.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("query, kv and indices must be 4D, 3D and 3D")
    batch, sequence_length, num_heads, head_dim = query.shape
    if tuple(kv.shape[:1] + kv.shape[2:]) != (batch, head_dim):
        raise ValueError("kv batch/head dimension does not match query")
    if tuple(indices.shape[:2]) != (batch, sequence_length):
        raise ValueError("indices batch/sequence dimensions do not match query")
    if tuple(sinks.shape) != (num_heads,):
        raise ValueError(f"sinks must have shape ({num_heads},)")
    if not query.is_cuda or not kv.is_cuda or not indices.is_cuda:
        raise ValueError("FlashMLA sparse prefill requires CUDA tensors")
    if query.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise TypeError("FlashMLA sparse prefill requires bfloat16 Q/KV")
    if indices.dtype is not torch.int32:
        raise TypeError("FlashMLA sparse prefill requires int32 indices")
    if head_dim != 512:
        raise ValueError(f"FlashMLA sparse prefill requires head_dim=512, got {head_dim}")
    if num_heads > 64:
        raise ValueError(
            "FlashMLA sparse prefill currently supports at most 64 local heads"
        )

    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    kv_length = kv.shape[1]
    offsets = (
        torch.arange(batch, device=indices.device, dtype=torch.int32)
        * kv_length
    ).view(batch, 1, 1)
    rebased = torch.where(indices >= 0, indices + offsets, indices)
    sparse_width = ((rebased.shape[-1] + 127) // 128) * 128
    if sparse_width != rebased.shape[-1]:
        rebased = torch.nn.functional.pad(
            rebased, (0, sparse_width - rebased.shape[-1]), value=-1
        )

    query_flat = query.reshape(batch * sequence_length, num_heads, head_dim)
    if num_heads < 64:
        query_padded = query.new_zeros(
            batch * sequence_length, 64, head_dim
        )
        query_padded[:, :num_heads].copy_(query_flat)
        sink_padded = sinks.new_zeros(64)
        sink_padded[:num_heads].copy_(sinks)
    else:
        query_padded = query_flat
        sink_padded = sinks

    output, _, _ = flash_mla_sparse_fwd(
        q=query_padded.contiguous(),
        kv=kv.reshape(batch * kv_length, 1, head_dim).contiguous(),
        indices=rebased.reshape(
            batch * sequence_length, 1, sparse_width
        ).contiguous(),
        sm_scale=softmax_scale,
        d_v=head_dim,
        attn_sink=sink_padded.contiguous(),
        # Rows may contain an invalid gap between the fixed-width SWA region
        # and compressed candidates.  The kernel handles -1 sentinels directly;
        # a prefix length would incorrectly truncate candidates after the gap.
        topk_length=None,
    )
    return output[:, :num_heads].view(
        batch, sequence_length, num_heads, head_dim
    )


__all__ = [
    "apply_rope_inplace",
    "compressed_indices",
    "fp8_fake_quantize_inplace",
    "precompute_rope_frequencies",
    "serving_max_length",
    "sparse_attention_reference",
    "sparse_attention_fused",
    "window_indices",
]
