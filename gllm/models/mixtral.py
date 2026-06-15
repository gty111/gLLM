from typing import Optional

import torch
from torch import nn

from gllm.dist_utils import (
    get_ep_rank,
    get_ep_size,
)
from gllm.input_data import InputData
from gllm.layers.layernorm import RMSNorm
from gllm.layers.moe import FusedMoE, determine_expert_map

from .qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2Model
from .weight_loader import (
    WeightRule,
    contains,
    make_w13_loader,
    make_w2_loader,
)


class MixtralMoE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=True,
            renormalize=True,
        )
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(orig_shape)


class MixtralAttention(Qwen2Attention):
    def __init__(self, layer_id, config):
        super().__init__(layer_id, config, qkv_bias=False)


class MixtralDecoderLayer(nn.Module):

    def __init__(self, layer_id, config):
        super().__init__()

        self.self_attn = MixtralAttention(layer_id, config)
        self.block_sparse_moe = MixtralMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(input_data, hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config, MixtralDecoderLayer)


class MixtralForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config, MixtralModel)

    def _make_load_context(self, weights):
        ctx = super()._make_load_context(weights)
        ctx.num_experts = self.config.num_local_experts
        _, ctx.expert_map = determine_expert_map(
            get_ep_size(), get_ep_rank(), ctx.num_experts
        )
        return ctx

    def weight_rules(self):
        # Mixtral stores experts as w1 (gate) / w3 (up) / w2 (down) and has no
        # shared expert or dense MLP, so the only specialization vs the dense
        # base is the routed-expert mapping. ``qkv_proj`` (weight + bias),
        # ``o_proj``, ``embed``/``lm_head`` and norms come from the base rules.
        return [
            WeightRule(
                contains("w13_weight"),
                make_w13_loader("w13_weight", "w1.weight", "w3.weight"),
                "w13_expert",
            ),
            WeightRule(
                contains("w2_weight"),
                make_w2_loader("w2_weight", "w2.weight"),
                "w2_expert",
            ),
        ] + super().weight_rules()
