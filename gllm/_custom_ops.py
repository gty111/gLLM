"""
Custom ops wrapper for gllm.

This module provides the kernel operation interface used throughout gllm.
All kernel calls go through this abstraction, making it the single point
where we can swap backends (sgl-kernel, Triton, etc.).

Previously used vllm's compiled _C.so and _moe_C.so;
now uses sgl-kernel + custom Triton kernels.
"""

from typing import Optional

import torch

# sgl-kernel imports
from sgl_kernel import (
    fused_add_rmsnorm as _sgl_fused_add_rmsnorm,
    moe_align_block_size as _sgl_moe_align_block_size,
    moe_fused_gate as _sgl_moe_fused_gate,
    moe_sum as _sgl_moe_sum,
    moe_sum_reduce as _sgl_moe_sum_reduce,
    rmsnorm as _sgl_rmsnorm,
    rotary_embedding as _sgl_rotary_embedding,
    silu_and_mul as _sgl_silu_and_mul,
    gelu_and_mul as _sgl_gelu_and_mul,
    topk_softmax as _sgl_topk_softmax,
    topk_sigmoid as _sgl_topk_sigmoid,
    merge_state_v2 as _sgl_merge_state_v2,
    moe_wna16_marlin_gemm as _sgl_moe_wna16_marlin_gemm,
    sgl_per_token_quant_fp8 as _sgl_per_token_quant_fp8,
    gptq_marlin_repack as _sgl_gptq_marlin_repack,
)

# Custom Triton kernels
from gllm.layers.ops.cache_kernels import (
    concat_and_cache_mla as _triton_concat_and_cache_mla,
    concat_and_cache_mla_fp8 as _triton_concat_and_cache_mla_fp8,
    gather_and_dequant_mla_fp8 as _triton_gather_and_dequant_mla_fp8,
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


def concat_and_cache_mla_fp8(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    fp8_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Native FP8-packed MLA cache store (DeepSeek Sparse Attention).

    Writes each token's latent into FlashMLA's FP8 sparse-decode layout
    (nope FP8 + per-128-tile fp32 scales + rope bf16) at its ``slot_mapping``
    slot. See :func:`gllm.layers.ops.cache_kernels.concat_and_cache_mla_fp8`.
    """
    _triton_concat_and_cache_mla_fp8(kv_c, k_pe, fp8_cache, slot_mapping)


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


def gather_and_dequant_mla_fp8(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather + dequant the native FP8-packed MLA cache into a bf16 buffer.

    Inverse of :func:`concat_and_cache_mla_fp8` (DeepSeek Sparse Attention). See
    :func:`gllm.layers.ops.cache_kernels.gather_and_dequant_mla_fp8`.
    """
    _triton_gather_and_dequant_mla_fp8(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        kv_lora_rank,
        qk_rope_head_dim,
        seq_starts,
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
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """
    Compute top-k softmax for MoE routing.

    ``renormalize`` is plumbed through to the kernel so callers can fold the
    "divide topk_weights by their sum" pass into the same launch instead of
    chaining a ``topk_weights / topk_weights.sum(-1, keepdim=True)`` after
    the kernel call (which costs an extra reduce + elementwise per MoE
    layer; profile of Qwen3-VL-30B-A3B TP=4 H20-3e shows ~20 ms / 100
    decode forwards spent on these two kernels plus their launch overhead).
    The kernel also accepts bf16 / fp16 gating logits directly -- callers
    should drop their pre-call ``.float()`` cast, which is another wasted
    elementwise + ~6 MB scratch per layer.
    """
    _sgl_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    Compute top-k sigmoid for MoE routing (DeepSeek-V3 noaux_tc style).

    Internally applies sigmoid then adds correction_bias for ranking,
    but returns the unbiased sigmoid value as the topk weight.
    Has no num_experts limit (unlike moe_fused_gate's per-group constraint).
    Suitable when there is no group hierarchy (or num_expert_group=topk_group=1).

    Args:
        topk_weights: [num_tokens, topk], float32, pre-allocated
        topk_ids: [num_tokens, topk], int32, pre-allocated
        gating_output: [num_tokens, num_experts], float32/16/bf16
        renormalize: whether to renormalize topk weights
        correction_bias: [num_experts], float32 (kernel requirement)
    """
    _sgl_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
    )


def moe_sum(input: torch.Tensor, output: torch.Tensor):
    """Sum expert outputs across topk dim: output = sum(input, dim=1).

    sgl_kernel.moe_sum only ships specialized CUDA kernels for topk in
    {2, 3, 4}; everything else (e.g. Qwen3-MoE / DeepSeek topk=8) falls
    back to ``at::sum_out``, which on bf16 input launches a multi-kernel
    chain (cast -> reduce -> cast). On Qwen3-VL-30B-A3B-Instruct prefill
    that fallback was the dominant CPU-dispatch source of cross-rank
    skew right before each ``cross_device_reduce_2stage`` call. Route
    those non-specialized topk values through ``moe_sum_reduce``, which
    has dedicated bf16/fp16 kernels for arbitrary topk in a single
    launch.
    """
    topk = input.size(1)
    if topk in (2, 3, 4):
        _sgl_moe_sum(input, output)
    else:
        # routed_scaling_factor=1.0 -> plain sum (kernel computes
        # ``out[t,d] = scale * sum_k(input[t,k,d])``).
        _sgl_moe_sum_reduce(input, output, 1.0)


def moe_sum_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
):
    """Fused sum + scale across the topk dim: output = scale * sum(input, dim=1)."""
    _sgl_moe_sum_reduce(input, output, routed_scaling_factor)


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

    Note: sgl_kernel internally does expert_id = topk_ids[i] + 1, so we must
    pass num_experts + 1 to account for the shifted indexing. The output
    expert_ids will be in range [-1, num_experts - 1].
    """
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
        True,
    )


def grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    scoring_func: str = "sigmoid",
):
    """
    Two-stage expert selection (DeepSeek-V2/V3 style).

    Uses sgl_kernel.moe_fused_gate for the fused implementation.

    ``moe_fused_gate`` applies the score function (sigmoid) *internally*, so it
    must be fed the **raw** router logits and the **raw** correction bias -- NOT
    a pre-sigmoid'd score. Passing already-sigmoid'd scores double-applies the
    nonlinearity (``sigmoid(sigmoid(logits))``), which silently shifts expert
    selection and routing weights away from the reference. The kernel only
    supports the sigmoid scoring function.
    """
    assert scoring_func == "sigmoid", (
        "moe_fused_gate only implements sigmoid scoring; "
        f"got {scoring_func!r}."
    )
    # moe_fused_gate requires input and bias to share a dtype, and does the
    # sigmoid + group reduction internally. Route in float32 (matches the HF
    # reference, which casts router logits to float32 before sigmoid) so the
    # bias add and top-2 group sum are done at full precision.
    topk_weights, topk_ids = _sgl_moe_fused_gate(
        input_tensor=gating_output.to(torch.float32),
        bias=correction_bias.to(torch.float32),
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        routed_scaling_factor=routed_scaling_factor,
    )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack packed-quant weights into Marlin layout."""
    return _sgl_gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k,
        size_n,
        num_bits,
    )


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
) -> torch.Tensor:
    """Invoke sgl-kernel int4/int8 Marlin fused-MoE GEMM."""
    return _sgl_moe_wna16_marlin_gemm(
        a,
        c_or_none,
        b_q_weight,
        b_bias_or_none,
        b_scales,
        global_scale_or_none,
        b_zeros_or_none,
        g_idx_or_none,
        perm_or_none,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


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
