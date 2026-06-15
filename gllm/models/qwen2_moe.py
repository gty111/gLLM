from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from gllm.dist_utils import (
    get_ep_rank,
    get_ep_size,
    get_tp_size,
    tensor_model_parallel_all_reduce,
)

from gllm.input_data import InputData
from gllm.layers.layernorm import RMSNorm
from gllm.layers.moe import FusedMoE, determine_expert_map

from .qwen2 import Qwen2Attention as Qwen2MoeAttention
from .qwen2 import Qwen2ForCausalLM
from .qwen2 import Qwen2MLP as Qwen2MoeMLP
from .qwen2 import Qwen2Model
from .weight_loader import (
    WeightRule,
    contains,
    make_w13_loader,
    make_w2_loader,
)


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        quant_config = getattr(config, "quantization_config", None)
        self.tp_size = get_tp_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        if getattr(config, "shared_expert_intermediate_size", 0) > 0:
            self.shared_expert = Qwen2MoeMLP(config, shared_expert=True)
            # ``Qwen2MoeMLP.down_proj`` is a ``RowParallelLinear`` that
            # all-reduces by default. We defer the all-reduce to this
            # block's tail (after summing the FusedMoE and shared-expert
            # contributions) so the shared output is not reduced twice
            # under TP > 1 — without this each TP rank's shared
            # contribution would be summed ``tp_size`` times in the final
            # output.
            if self.tp_size > 1:
                self.shared_expert.down_proj.reduce_results = False
        else:
            self.shared_expert = None
        if hasattr(config, "shared_expert_intermediate_size"):
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                )

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        layer_id,
        config,
        moe_block_type=Qwen2MoeSparseMoeBlock,
        mlp_type=Qwen2MoeMLP,
        attn_type=Qwen2MoeAttention,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attn_type(layer_id, config)

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_id not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = moe_block_type(config=config)
        else:
            self.mlp = mlp_type(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            input_data=input_data,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2MoeModel(Qwen2Model):
    def __init__(self, config, decoder_layer_type=Qwen2MoeDecoderLayer):
        super().__init__(config, decoder_layer_type)


class Qwen2MoeForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, model_type=Qwen2MoeModel):
        super().__init__(config, model_type)

    def _resolve_num_experts(self):
        n = getattr(
            self.config, "num_experts", getattr(self.config, "n_routed_experts", None)
        )
        assert n is not None
        return n

    def _make_load_context(self, weights):
        ctx = super()._make_load_context(weights)
        ctx.num_experts = self._resolve_num_experts()
        _, ctx.expert_map = determine_expert_map(
            get_ep_size(), get_ep_rank(), ctx.num_experts
        )
        return ctx

    def expert_weight_rules(self):
        """Routed-expert rules for the standard ``gate_proj``/``up_proj``/
        ``down_proj`` per-expert checkpoint layout (Qwen2/3-MoE, DeepSeek bf16).

        Subclasses with other expert key conventions (Mixtral's ``w1``/``w3``/
        ``w2``, int4 packed) override or extend this.
        """
        return [
            WeightRule(
                contains("w13_weight"),
                make_w13_loader("w13_weight", "gate_proj.weight", "up_proj.weight"),
                "w13_expert",
            ),
            WeightRule(
                contains("w2_weight"),
                make_w2_loader("w2_weight", "down_proj.weight"),
                "w2_expert",
            ),
        ]

    def weight_rules(self):
        # Expert rules first; their FusedMoE param names (``w13_weight`` /
        # ``w2_weight``) are disjoint from the dense base patterns, so order
        # between the two groups is not load-bearing, but keeping experts up
        # front documents that the MoE block is the specialization.
        return self.expert_weight_rules() + super().weight_rules()

