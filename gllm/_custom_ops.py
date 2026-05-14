"""
Custom ops wrapper for gllm.

This module provides the kernel operation interface used throughout gllm.
All kernel calls go through this abstraction, making it the single point
where we can swap backends (sgl-kernel, flashinfer, Triton, etc.).

Previously used vllm's compiled _C.so and _moe_C.so;
now uses sgl-kernel + flashinfer + custom Triton kernels.
"""

from typing import Optional

import torch

# sgl-kernel imports
from sgl_kernel import (
    fused_add_rmsnorm as _sgl_fused_add_rmsnorm,
    moe_align_block_size as _sgl_moe_align_block_size,
    moe_fused_gate as _sgl_moe_fused_gate,
    moe_sum as _sgl_moe_sum,
    rmsnorm as _sgl_rmsnorm,
    rotary_embedding as _sgl_rotary_embedding,
    silu_and_mul as _sgl_silu_and_mul,
    gelu_and_mul as _sgl_gelu_and_mul,
    topk_softmax as _sgl_topk_softmax,
    merge_state_v2 as _sgl_merge_state_v2,
    sgl_per_token_quant_fp8 as _sgl_per_token_quant_fp8,
)

# Custom Triton kernels
from gllm.layers.ops.cache_kernels import (
    concat_and_cache_mla as _triton_concat_and_cache_mla,
    gather_and_maybe_dequant_cache as _triton_gather_and_maybe_dequant_cache,
    reshape_and_cache_flash as _triton_reshape_and_cache_flash,
)
from gllm.layers.ops.batched_rotary_kernel import (
    batched_rotary_embedding as _triton_batched_rotary_embedding,
)


# =============================================================================
# Cache ops
# =============================================================================


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    _triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    _triton_concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    _triton_gather_and_maybe_dequant_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, kv_cache_dtype, scale, seq_starts
    )


# =============================================================================
# Merge attention states
# =============================================================================


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    """
    Merge two partial attention outputs using log-sum-exp trick.

    sgl_kernel.merge_state_v2 signature:
        merge_state_v2(v_a, s_a, v_b, s_b, v_merged=None, s_merged=None)
    """
    _sgl_merge_state_v2(
        v_a=prefix_output,
        s_a=prefix_lse,
        v_b=suffix_output,
        s_b=suffix_lse,
        v_merged=output,
        s_merged=output_lse,
    )


# =============================================================================
# Activation ops
# =============================================================================


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """
    Fused SiLU activation: out = silu(x[..., :d]) * x[..., d:]

    Note: sgl_kernel requires 16-byte alignment on last dim.
    Falls back to PyTorch native when alignment is not met.
    """
    # sgl_kernel requires output rows to be 16-byte aligned for subsequent ops
    d = x.shape[-1] // 2
    if d * x.element_size() % 16 != 0:
        # Fallback: output dim not 16-byte aligned (e.g., vision encoder dim=3420 in bf16)
        x_flat = x.view(-1, x.shape[-1])
        out_flat = out.view(-1, d)
        out_flat.copy_(torch.nn.functional.silu(x_flat[..., :d]) * x_flat[..., d:])
        return
    _sgl_silu_and_mul(x, out=out)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """
    Fused GELU activation: out = gelu(x[..., :d]) * x[..., d:]

    Note: sgl_kernel requires 16-byte alignment on last dim.
    Falls back to PyTorch native when alignment is not met.
    """
    d = x.shape[-1] // 2
    if d * x.element_size() % 16 != 0:
        x_flat = x.view(-1, x.shape[-1])
        out_flat = out.view(-1, d)
        out_flat.copy_(torch.nn.functional.gelu(x_flat[..., :d]) * x_flat[..., d:])
        return
    _sgl_gelu_and_mul(x, out=out)


# =============================================================================
# Position encoding ops
# =============================================================================


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Apply rotary positional embedding in-place."""
    _sgl_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


def batched_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor,
) -> None:
    """Apply batched rotary embedding with per-token cache offsets."""
    _triton_batched_rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox, rot_dim, cos_sin_cache_offsets
    )


# =============================================================================
# Layer norm ops
# =============================================================================


def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """
    RMS normalization: out = (input / RMS(input)) * weight

    sgl_kernel.rmsnorm requires 2D input; reshape if needed.
    """
    input = input.contiguous()
    orig_shape = input.shape
    if input.ndim != 2:
        input = input.view(-1, input.shape[-1])
        out_2d = out.view(-1, out.shape[-1])
        _sgl_rmsnorm(input, weight, eps=epsilon, out=out_2d)
    else:
        _sgl_rmsnorm(input, weight, eps=epsilon, out=out)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """
    Fused residual + RMS norm (in-place):
      Step 1: residual += input
      Step 2: input = (residual / RMS(residual)) * weight

    sgl_kernel requires 2D tensors; reshape if needed.
    """
    orig_shape = input.shape
    if input.ndim != 2:
        input_2d = input.view(-1, input.shape[-1])
        residual_2d = residual.view(-1, residual.shape[-1])
        _sgl_fused_add_rmsnorm(input_2d, residual_2d, weight, eps=epsilon)
    else:
        _sgl_fused_add_rmsnorm(input, residual, weight, eps=epsilon)


# =============================================================================
# MoE ops
# =============================================================================


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indicies: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    """
    Compute top-k softmax for MoE routing.

    Note: sgl_kernel.topk_softmax doesn't use token_expert_indicies.
    The parameter is kept for API compatibility but ignored.
    """
    _sgl_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
    )


def moe_sum(input: torch.Tensor, output: torch.Tensor):
    """Sum expert outputs: output += sum(input, dim=1)"""
    _sgl_moe_sum(input, output)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """
    Align token distribution across experts to be compatible with block size.

    Note: sgl_kernel requires an additional cumsum_buffer and uses num_experts+1
    to account for the -1 expert id in expert parallelism.
    """
    # sgl_kernel expects num_experts + 1 (for EP filtered expert id = -1)
    # and cumsum_buffer of size num_experts + 2
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )
    _sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        False,  # pad_sorted_token_ids - match caller's buffer allocation
    )


def grouped_topk(
    scores: torch.Tensor,
    scores_with_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    """
    Two-stage expert selection (DeepSeek-V2/V3 style).

    Uses sgl_kernel.moe_fused_gate for the fused implementation.
    """
    # moe_fused_gate expects the bias to be separate from scores
    # It takes (input_tensor, bias, num_expert_group, topk_group, topk, ...)
    # and returns (topk_weights, topk_ids)
    bias = scores_with_bias - scores  # Extract the bias component

    topk_weights, topk_ids = _sgl_moe_fused_gate(
        input_tensor=scores,
        bias=bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
    )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# =============================================================================
# Quantization ops
# =============================================================================


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8.

    Uses sgl_kernel.sgl_per_token_quant_fp8 for per-token quantization.
    """
    output_q = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    output_s = torch.empty(
        input.shape[0], dtype=torch.float32, device=input.device
    )
    _sgl_per_token_quant_fp8(input, output_q, output_s)
    return output_q, output_s
