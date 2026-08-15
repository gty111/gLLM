from __future__ import annotations

from typing import Optional

import torch


_EPILOGUE_TILE_M = 128
_BLOCK_K = 128


def bf16_moe_support_reason(layer: torch.nn.Module) -> Optional[str]:
    """Return why the TRT-LLM BF16 MoE backend cannot serve ``layer``."""
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.get_device_capability()[0] != 10:
        return "TRT-LLM BF16 MoE requires an SM100-family GPU"
    try:
        from flashinfer.fused_moe import trtllm_bf16_moe
    except (ImportError, AttributeError) as exc:
        return f"FlashInfer TRT-LLM BF16 MoE is unavailable: {exc}"
    if not callable(trtllm_bf16_moe):
        return "flashinfer.fused_moe.trtllm_bf16_moe is not callable"

    w13 = layer.w13_weight
    w2 = layer.w2_weight
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "TRT-LLM BF16 MoE requires bfloat16 expert weights"
    if layer.global_num_experts > 2048:
        return "TRT-LLM routing supports at most 2048 experts"
    if layer.activation != "silu":
        return f"TRT-LLM BF16 MoE does not support activation={layer.activation!r}"
    if layer.apply_router_weight_on_input:
        return "router weights on the GEMM1 input are unsupported"
    if layer.scoring_func != "softmax":
        return f"routing scoring_func={layer.scoring_func!r} is unsupported"
    if layer.use_grouped_topk or layer.e_score_correction_bias is not None:
        return "grouped or biased routing is unsupported by this dispatch"
    if w13.shape[-1] * w13.element_size() % _BLOCK_K:
        return "the GEMM1 K dimension is not aligned to 128 bytes"
    if w2.shape[-1] * w2.element_size() % _BLOCK_K:
        return "the GEMM2 K dimension is not aligned to 128 bytes"
    if layer.intermediate_size_per_partition % 128:
        return "the local intermediate size is not aligned to 128 elements"
    return None


def convert_bf16_moe_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ordinary expert matrices to TRT-LLM BlockMajorK layout."""
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError("TRT-LLM BF16 MoE requires bfloat16 weights")

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
    num_experts = w13_weight.shape[0]

    def allocate_block_layout(weight: torch.Tensor) -> torch.Tensor:
        rows, byte_columns = weight[0].view(torch.uint8).shape
        if byte_columns % _BLOCK_K:
            raise ValueError(
                f"weight byte columns ({byte_columns}) must be divisible by {_BLOCK_K}"
            )
        return torch.empty(
            (num_experts, byte_columns // _BLOCK_K, rows, _BLOCK_K),
            dtype=torch.uint8,
            device=weight.device,
        )

    w13_block = allocate_block_layout(w13_weight)
    w2_block = allocate_block_layout(w2_weight)

    def copy_expert(
        output: torch.Tensor,
        expert: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> None:
        rows = expert.shape[0]
        source = expert.view(torch.uint8).view(
            rows, output.shape[0], _BLOCK_K
        ).permute(1, 0, 2)
        torch.index_select(
            source,
            1,
            row_indices.to(expert.device),
            out=output,
        )

    for expert_id in range(num_experts):
        w13_expert = w13_weight[expert_id].view(torch.uint8)
        w13_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w13_expert,
            _EPILOGUE_TILE_M,
            is_gated_act_gemm=True,
        )
        # gLLM stores gate then up; TRT-LLM's fused epilogue consumes the
        # opposite half ordering after its gated-activation row permutation.
        w13_indices = (w13_indices + w13_expert.shape[0] // 2) % w13_expert.shape[0]
        copy_expert(w13_block[expert_id], w13_expert, w13_indices)

        w2_expert = w2_weight[expert_id].view(torch.uint8)
        w2_indices = get_w2_permute_indices_with_cache(
            cache,
            w2_expert,
            _EPILOGUE_TILE_M,
        )
        copy_expert(w2_block[expert_id], w2_expert, w2_indices)

    return w13_block.view(torch.bfloat16), w2_block.view(torch.bfloat16)


def trtllm_bf16_moe(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Run one finalized, unquantized TRT-LLM MoE forward."""
    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import trtllm_bf16_moe as flashinfer_moe
    from flashinfer.fused_moe.core import ActivationType

    routing_method = (
        RoutingMethodType.RenormalizeNaive
        if layer.renormalize
        else RoutingMethodType.Default
    )
    local_expert_offset = layer.ep_rank * layer.local_num_experts
    output = torch.empty_like(hidden_states)
    result = flashinfer_moe(
        routing_logits=router_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=layer.w13_weight,
        gemm2_weights=layer.w2_weight,
        num_experts=layer.global_num_experts,
        top_k=layer.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=local_expert_offset,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method,
        use_shuffled_weight=True,
        do_finalize=True,
        activation_type=ActivationType.Swiglu.value,
        norm_topk_prob=layer.renormalize,
        output=output,
    )
    if isinstance(result, (list, tuple)):
        return result[0]
    return result
