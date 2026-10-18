"""DeepSeek-V4 mixture-of-experts building blocks.

The V4 checkpoint mixes three numerical formats in one MoE block: the router
is BF16, routed experts remain packed MXFP4, and the shared expert is block
FP8.  Keeping the routing and activation order here explicit makes this module
the correctness path that optimized grouped kernels must reproduce.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from gllm.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from gllm.layers.moe.mxfp4_experts import NativeMXFP4Experts
from gllm.layers.moe.topk import deepseek_v4_topk
from gllm.distributed.parallel_state import (
    ep_all_reduce,
    get_ep_size,
    get_tp_size,
    is_use_ep,
    tensor_model_parallel_all_reduce,
)


def deepseek_v4_swiglu(
    gate_up: torch.Tensor,
    *,
    limit: float = 10.0,
) -> torch.Tensor:
    """Apply the checkpoint's clamped SwiGLU and return BF16 activations.

    The reference computes the activation in FP32, then casts it back to the
    model dtype before the dynamically quantized down projection.
    """
    if gate_up.shape[-1] % 2:
        raise ValueError("DeepSeek-V4 gate/up width must be even")
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(gate) * up).to(gate_up.dtype)


class DeepseekV4SharedExpert(nn.Module):
    """Tensor-parallel block-FP8 shared expert from the native checkpoint."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        quant_config: dict[str, Any],
        swiglu_limit: float = 10.0,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        intermediate = deepseek_v4_swiglu(
            gate_up,
            limit=self.swiglu_limit,
        )
        return self.down_proj(intermediate)



class DeepseekV4MoE(nn.Module):
    """Native-precision DeepSeek-V4 router, experts, and shared expert."""

    def __init__(self, layer_id: int, config: Any) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.swiglu_limit = getattr(config, "swiglu_limit", 10.0)
        mlp_layer_types = getattr(config, "mlp_layer_types", None)
        # ``mlp_layer_types`` only describes the 43 target-model layers.  The
        # three DSpark stages live under ``mtp.*`` and use ordinary score
        # routing, so their synthetic layer ids deliberately fall past this
        # list.
        self.hash_routing = (
            mlp_layer_types[layer_id] == "hash_moe"
            if mlp_layer_types is not None and layer_id < len(mlp_layer_types)
            else layer_id < getattr(config, "num_hash_layers", 0)
        )

        if getattr(config, "n_shared_experts", 1) != 1:
            raise ValueError("DeepSeek-V4 currently requires exactly one shared expert")

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        if self.hash_routing:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    self.topk,
                    dtype=torch.int32,
                    device="cuda",
                ),
                requires_grad=False,
            )
            self.register_parameter("e_score_correction_bias", None)
        else:
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(
                    self.num_experts,
                    dtype=torch.float32,
                    device="cuda",
                ),
                requires_grad=False,
            )

        self.experts = NativeMXFP4Experts(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            swiglu_limit=self.swiglu_limit,
        )
        self.shared_experts = DeepseekV4SharedExpert(
            self.hidden_size,
            config.moe_intermediate_size,
            quant_config=config.quantization_config,
            swiglu_limit=self.swiglu_limit,
            # The routed and shared branches are both partial on each rank.
            # Their sum is reduced once below instead of issuing two
            # collectives per transformer layer.
            reduce_results=False,
        )


    def route(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The checkpoint deliberately promotes both operands to FP32 here.
        router_logits = torch.nn.functional.linear(
            hidden_states.float(),
            self.gate.weight.float(),
        )
        return deepseek_v4_topk(
            router_logits,
            self.topk,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            correction_bias=self.e_score_correction_bias,
            input_ids=input_ids,
            hash_indices_table=self.tid2eid,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if self.hash_routing:
            if input_ids is None:
                raise ValueError(
                    "DeepSeek-V4 hash-routed layers require the current input ids"
                )
            input_ids = input_ids.reshape(-1)

        topk_weights, topk_ids = self.route(hidden_states, input_ids)
        routed_output = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            reduce_results=False,
        )
        shared_output = self.shared_experts(hidden_states)
        output = routed_output + shared_output
        if is_use_ep() and get_ep_size() > 1:
            output = ep_all_reduce(output)
        elif not is_use_ep() and get_tp_size() > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output.to(hidden_states.dtype).view(original_shape)


__all__ = [
    "DeepseekV4MoE",
    "DeepseekV4SharedExpert",
    "deepseek_v4_swiglu",
]
