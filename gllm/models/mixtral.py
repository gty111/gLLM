from typing import Optional

import torch
from torch import nn

from gllm.dist_utils import (
    get_ep_rank,
    get_ep_size,
    get_local_rank,
    resolve_pp_layer_idx,
)
from gllm.input_data import InputData
from gllm.layers.layernorm import RMSNorm
from gllm.layers.moe import FusedMoE, determine_expert_map
from gllm.utils import get_model_load_pbar

from .qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2Model
from .weight_utils import (
    copy_qkv_proj,
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    load_fused_w13_per_expert,
    load_w2_per_expert,
    moe_expert_load_pool,
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

    def load_weights(self, weights, mp_load_progress=None):
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank() * 2] = len(parameters)
            mp_load_progress[get_local_rank() * 2 + 1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))

        attn = self.model.layers[0].self_attn
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim

        num_experts = self.config.num_local_experts

        _, expert_map = determine_expert_map(get_ep_size(), get_ep_rank(), num_experts)

        with moe_expert_load_pool(num_experts) as expert_pool:
            for k, v in parameters.items():
                k = resolve_pp_layer_idx(k, 2, self.model.start_layer)
                if k.find("self_attn.qkv_proj.weight") != -1:
                    copy_qkv_proj(
                        v.data,
                        weights[k.replace("qkv_proj", "q_proj")],
                        weights[k.replace("qkv_proj", "k_proj")],
                        weights[k.replace("qkv_proj", "v_proj")],
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    )
                elif k.find("self_attn.qkv_proj.bias") != -1:
                    copy_qkv_proj(
                        v.data,
                        weights[k.replace("qkv_proj", "q_proj")],
                        weights[k.replace("qkv_proj", "k_proj")],
                        weights[k.replace("qkv_proj", "v_proj")],
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    )
                elif k.find("w13_weight") != -1:  # expert
                    # Mixtral stores experts as w1 (gate) / w3 (up) / w2 (down).
                    load_fused_w13_per_expert(
                        v.data,
                        weights,
                        key_for_gate=lambda i, k=k: k.replace(
                            "w13_weight", f"{i}.w1.weight"
                        ),
                        key_for_up=lambda i, k=k: k.replace(
                            "w13_weight", f"{i}.w3.weight"
                        ),
                        expert_map=expert_map,
                        num_experts=num_experts,
                        pool=expert_pool,
                    )
                elif k.find("w2_weight") != -1:  # expert
                    load_w2_per_expert(
                        v.data,
                        weights,
                        key_for_down=lambda i, k=k: k.replace(
                            "w2_weight", f"{i}.w2.weight"
                        ),
                        expert_map=expert_map,
                        num_experts=num_experts,
                        pool=expert_pool,
                    )
                elif k.find("self_attn.o_proj") != -1:
                    copy_single_proj_dim1(v.data, weights[k])
                elif k.find("embed_tokens") != -1 or k.find("lm_head") != -1:
                    copy_single_proj_dim0(v.data, weights[k])
                else:
                    v.data.copy_(weights[k])
                if mp_load_progress is not None:
                    mp_load_progress[get_local_rank() * 2 + 1] += 1
                else:
                    pbar.update(1)
