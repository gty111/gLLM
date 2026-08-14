"""
Custom ops wrapper for gllm.

This module provides the kernel operation interface used throughout gllm.
All kernel calls go through this abstraction, making it the single point
where we can swap backends (sgl-kernel, Triton, etc.).

The current backends are FlashInfer, sgl-kernel, and in-tree Triton kernels.
"""

from typing import Optional

import torch
from gllm.layers.ops.flashinfer_utils import ensure_ninja_on_path

ensure_ninja_on_path()

from flashinfer.activation import (
    gelu_and_mul as _flashinfer_gelu_and_mul,
    silu_and_mul as _flashinfer_silu_and_mul,
)
from flashinfer.fused_moe import (
    convert_to_block_layout as _flashinfer_convert_to_block_layout,
    fused_topk_deepseek as _flashinfer_fused_topk_deepseek,
    trtllm_mxint4_block_scale_moe as _flashinfer_mxint4_moe,
)
from flashinfer.norm import (
    fused_add_rmsnorm as _flashinfer_fused_add_rmsnorm,
    rmsnorm as _flashinfer_rmsnorm,
)
from flashinfer.rope import (
    apply_rope_with_cos_sin_cache_inplace as _flashinfer_rotary_embedding,
)

# sgl-kernel imports
from sgl_kernel import (
    moe_align_block_size as _sgl_moe_align_block_size,
    moe_sum as _sgl_moe_sum,
    moe_sum_reduce as _sgl_moe_sum_reduce,
    topk_softmax as _sgl_topk_softmax,
    topk_sigmoid as _sgl_topk_sigmoid,
    merge_state_v2 as _sgl_merge_state_v2,
    sgl_per_token_quant_fp8 as _sgl_per_token_quant_fp8,
)

# Custom Triton kernels
from gllm.layers.ops.cache_kernels import (
    concat_and_cache_mla as _triton_concat_and_cache_mla,
    concat_and_cache_mla_fp8 as _triton_concat_and_cache_mla_fp8,
    dequant_mla_fp8_flat as _triton_dequant_mla_fp8_flat,
    dequant_mla_fp8_slots as _triton_dequant_mla_fp8_slots,
    gather_and_dequant_mla_fp8 as _triton_gather_and_dequant_mla_fp8,
    gather_and_maybe_dequant_cache as _triton_gather_and_maybe_dequant_cache,
    reshape_and_cache_flash as _triton_reshape_and_cache_flash,
    store_index_k_fp8 as _triton_store_index_k_fp8,
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


def dequant_mla_fp8_flat(
    src_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """Dequant the whole FP8-packed MLA cache to a flat bf16 latent buffer.

    Returns ``[num_slots, kv_lora_rank + qk_rope_head_dim]`` bf16 indexed by
    absolute physical cache slot (DeepSeek Sparse Attention prefill). See
    :func:`gllm.layers.ops.cache_kernels.dequant_mla_fp8_flat`.
    """
    return _triton_dequant_mla_fp8_flat(
        src_cache, kv_lora_rank, qk_rope_head_dim
    )


def dequant_mla_fp8_slots(
    src_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """Dequant only ``slot_ids`` of the FP8-packed MLA cache -> flat bf16 buffer.

    Gather-only variant of :func:`dequant_mla_fp8_flat`: fills only the referenced
    physical slots (DSA prefill top-k's unique slots), physical-slot-indexed. See
    :func:`gllm.layers.ops.cache_kernels.dequant_mla_fp8_slots`.
    """
    return _triton_dequant_mla_fp8_slots(
        src_cache, slot_ids, kv_lora_rank, qk_rope_head_dim
    )


def store_index_k_fp8(
    idx_k: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
    index_head_dim: int,
    use_ue8m0: bool = False,
) -> None:
    """Quantize + write the DSA indexer key into the paged FP8 index cache. See
    :func:`gllm.layers.ops.cache_kernels.store_index_k_fp8`."""
    _triton_store_index_k_fp8(
        idx_k, cache, slot_mapping, page_size, index_head_dim, use_ue8m0=use_ue8m0
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

    Note: the fused kernel requires 16-byte alignment on the last dimension.
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
    _flashinfer_silu_and_mul(x, out=out)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """
    Fused GELU activation: out = gelu(x[..., :d]) * x[..., d:]

    Note: the fused kernel requires 16-byte alignment on the last dimension.
    Falls back to PyTorch native when alignment is not met.
    """
    d = x.shape[-1] // 2
    if d * x.element_size() % 16 != 0:
        x_flat = x.view(-1, x.shape[-1])
        out_flat = out.view(-1, d)
        out_flat.copy_(torch.nn.functional.gelu(x_flat[..., :d]) * x_flat[..., d:])
        return
    _flashinfer_gelu_and_mul(x, out=out)


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
    _flashinfer_rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )


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

    FlashInfer RMSNorm requires 2D input; reshape if needed.
    """
    input = input.contiguous()
    if input.ndim != 2:
        input = input.view(-1, input.shape[-1])
        out_2d = out.view(-1, out.shape[-1])
        _flashinfer_rmsnorm(input, weight, eps=epsilon, out=out_2d)
    else:
        _flashinfer_rmsnorm(input, weight, eps=epsilon, out=out)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """
    Fused residual + RMS norm (in-place):
      Step 1: residual += input
      Step 2: input = (residual / RMS(residual)) * weight

    FlashInfer requires 2D tensors; reshape if needed.
    """
    if input.ndim != 2:
        input_2d = input.view(-1, input.shape[-1])
        residual_2d = residual.view(-1, residual.shape[-1])
        _flashinfer_fused_add_rmsnorm(
            input_2d, residual_2d, weight, eps=epsilon
        )
    else:
        _flashinfer_fused_add_rmsnorm(input, residual, weight, eps=epsilon)


def gemma_rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """Gemma RMSNorm with the learned ``weight + 1`` applied in fp32.

    sglang-kernel 0.4.6's SM100 kernel accumulates differently from the fp32
    checkpoint reference, so retain the established gLLM numerical contract.
    """
    input_fp32 = input.float()
    normalized = input_fp32 * torch.rsqrt(
        input_fp32.square().mean(dim=-1, keepdim=True) + epsilon
    )
    out.copy_((normalized * (weight.float() + 1.0)).to(input.dtype))


def gemma_fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """In-place residual add followed by fp32-reference Gemma RMSNorm."""
    residual.add_(input)
    residual_fp32 = residual.float()
    normalized = residual_fp32 * torch.rsqrt(
        residual_fp32.square().mean(dim=-1, keepdim=True) + epsilon
    )
    input.copy_((normalized * (weight.float() + 1.0)).to(input.dtype))


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
    Has no grouped-routing constraints.
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

    Uses FlashInfer for normalized DeepSeek routing and a native Torch fallback
    for configurations outside FlashInfer's fused-kernel limits.

    The score function is applied internally, so this function must receive the
    raw router logits and correction bias rather than pre-sigmoid scores.
    """
    assert scoring_func == "sigmoid", (
        "grouped_topk only implements sigmoid scoring; "
        f"got {scoring_func!r}."
    )
    # The fused kernels require input and bias to share a dtype, and do the
    # sigmoid + group reduction internally. Route in float32 (matches the HF
    # reference, which casts router logits to float32 before sigmoid) so the
    # bias add and top-2 group sum are done at full precision.
    num_experts = gating_output.shape[1]
    experts_per_group = (
        num_experts // num_expert_group if num_expert_group > 0 else num_experts
    )
    flashinfer_supported = topk <= 8 and (
        (
            num_expert_group == 1
            and topk_group == 1
            and topk == 1
            and num_experts <= 384
        )
        or (
            1 < num_expert_group <= 32
            and num_experts % num_expert_group == 0
            and topk_group <= num_expert_group
            and topk_group * num_expert_group >= topk
            and experts_per_group <= 32
            and experts_per_group * topk_group <= 128
        )
    )
    if renormalize and flashinfer_supported:
        scores = gating_output.float().contiguous()
        bias = correction_bias.float().contiguous()
        topk_weights = torch.empty(
            scores.shape[0], topk, dtype=scores.dtype, device=scores.device
        )
        topk_ids = torch.empty(
            scores.shape[0], topk, dtype=torch.int32, device=scores.device
        )
        _flashinfer_fused_topk_deepseek(
            scores=scores,
            bias=bias,
            n_group=num_expert_group,
            topk_group=topk_group,
            topk=topk,
            routed_scaling_factor=routed_scaling_factor,
            topk_values=topk_weights,
            topk_indices=topk_ids,
        )
        # FlashInfer already performs normalize-then-scale in its fused kernel.
        return topk_weights, topk_ids
    # Preserve the exact noaux_tc routing semantics when the configuration is
    # outside FlashInfer's fused-kernel limits.
    scores = gating_output.float().sigmoid()
    biased_scores = scores + correction_bias.float().unsqueeze(0)
    num_tokens, num_experts = scores.shape
    grouped = biased_scores.view(num_tokens, num_expert_group, -1)
    group_score_k = min(2, grouped.shape[-1])
    group_scores = grouped.topk(group_score_k, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand_as(grouped)
        .reshape(num_tokens, num_experts)
    )
    topk_ids = biased_scores.masked_fill(~expert_mask, float("-inf")).topk(
        topk, dim=-1
    ).indices
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def _uint4b8_to_signed_int4_(packed: torch.Tensor) -> None:
    """Convert eight biased INT4 values in each int32 to signed nibbles."""
    if packed.dtype != torch.int32:
        raise TypeError(f"expected int32 packed weights, got {packed.dtype}")

    # Bound the temporary nibble tensor while processing large expert shards.
    flat = packed.view(-1)
    chunk_elems = (64 << 20) // packed.element_size()
    for start in range(0, flat.numel(), chunk_elems):
        chunk = flat[start : start + chunk_elems]
        for lane in range(8):
            shift = 4 * lane
            nibble = (chunk >> shift) & 0xF
            chunk &= ~(0xF << shift)
            chunk |= ((nibble - 8) & 0xF) << shift


def prepare_flashinfer_mxint4_moe_weight(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    gated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare checkpoint uint4b8 weights for FlashInfer MXINT4 MoE.

    The checkpoint stores eight biased INT4 values per int32. FlashInfer uses
    signed INT4 bytes, TensorRT-LLM row shuffles, interleaved block scales, and
    a BlockMajorK weight layout.
    """
    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    if packed_weight.ndim != 3 or scales.ndim != 3:
        raise ValueError("MXINT4 MoE weights and scales must both be 3D")
    if packed_weight.dtype == torch.int32:
        _uint4b8_to_signed_int4_(packed_weight)
    elif packed_weight.dtype != torch.uint8:
        raise TypeError(
            "MXINT4 MoE weights must be checkpoint int32 or packed uint8, "
            f"got {packed_weight.dtype}"
        )
    if scales.dtype != torch.bfloat16:
        raise TypeError(f"MXINT4 MoE scales must be bfloat16, got {scales.dtype}")

    weights_u8 = packed_weight.view(torch.uint8)
    num_experts, rows, _ = weights_u8.shape
    if gated and rows % 2:
        raise ValueError(f"gated MXINT4 weight row count must be even, got {rows}")

    row_swap = None
    if gated:
        half = rows // 2
        # Checkpoints use [gate, up], while TRT-LLM's fused SwiGLU consumes
        # [up, gate] before applying its gated-row interleave.
        row_swap = torch.cat((torch.arange(half, rows), torch.arange(half))).to(
            weights_u8.device
        )

    permutation_cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
    prepared_weights = None
    prepared_scales = []
    epilogue_tile_m = 128
    block_k_bytes = 128

    for expert in range(num_experts):
        weight = weights_u8[expert]
        scale = scales[expert]
        if gated:
            weight_perm = _maybe_get_cached_w3_w1_permute_indices(
                permutation_cache, weight, epilogue_tile_m
            )
            scale_perm = _maybe_get_cached_w3_w1_permute_indices(
                permutation_cache,
                scale,
                epilogue_tile_m,
                num_elts_per_sf=32,
            )
            weight_perm = row_swap[weight_perm].to(weight.device)
            scale_perm = row_swap[scale_perm].to(scale.device)
        else:
            weight_perm = get_w2_permute_indices_with_cache(
                permutation_cache, weight, epilogue_tile_m
            )
            scale_perm = get_w2_permute_indices_with_cache(
                permutation_cache,
                scale,
                epilogue_tile_m,
                num_elts_per_sf=16,
            )

        block_weight = _flashinfer_convert_to_block_layout(
            weight[weight_perm].contiguous(), block_k_bytes
        )
        if prepared_weights is None:
            prepared_weights = torch.empty(
                (num_experts, *block_weight.shape),
                dtype=block_weight.dtype,
                device=block_weight.device,
            )
        prepared_weights[expert].copy_(block_weight)
        prepared_scales.append(
            block_scale_interleave(scale[scale_perm].contiguous())
        )

    assert prepared_weights is not None
    return prepared_weights, torch.stack(prepared_scales).view(torch.bfloat16)


def flashinfer_mxint4_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    *,
    global_num_experts: int,
    local_num_experts: int,
    local_expert_offset: int,
    top_k: int,
    intermediate_size: int,
    renormalize: bool,
    use_grouped_topk: bool,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
    scoring_func: str,
    correction_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the monolithic FlashInfer TensorRT-LLM MXINT4 MoE kernel."""
    from flashinfer import RoutingMethodType

    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            f"FlashInfer MXINT4 MoE requires bfloat16 activations, got "
            f"{hidden_states.dtype}"
        )

    routing_bias = None
    routing_logits = router_logits
    if use_grouped_topk:
        if scoring_func != "sigmoid" or correction_bias is None:
            raise ValueError(
                "FlashInfer MXINT4 grouped routing requires sigmoid scoring "
                "and a correction bias"
            )
        routing_method = RoutingMethodType.DeepSeekV3
        routing_logits = router_logits.float()
        routing_bias = correction_bias.to(torch.bfloat16)
        n_group = num_expert_group
        selected_groups = topk_group
    else:
        if scoring_func != "softmax":
            raise ValueError(
                "FlashInfer MXINT4 ungrouped routing requires softmax scoring"
            )
        routing_method = (
            RoutingMethodType.RenormalizeNaive
            if renormalize
            else RoutingMethodType.Default
        )
        n_group = None
        selected_groups = None

    output = _flashinfer_mxint4_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_scales,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_scales,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=selected_groups,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method,
        do_finalize=True,
        enable_pdl=None,
        output=None,
        tune_max_num_tokens=8192,
        norm_topk_prob=renormalize,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.to(hidden_states.dtype)


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
