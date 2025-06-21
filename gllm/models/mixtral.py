import torch

from typing import Optional
from torch import nn

from gllm.layers.layernorm import RMSNorm
from gllm.layers.moe.fused_moe_triton.layer import FusedMoE, determine_expert_map
from gllm.input_data import InputData
from gllm.dist_utils import (get_local_rank, resolve_pp_layer_idx, get_tp_size,
                             get_ep_size, get_ep_rank, resolve_ep_expert_idx,
                             is_use_ep)
from gllm.utils import get_model_load_pbar

from .qwen2 import Qwen2Attention
from .qwen2 import Qwen2Model
from .qwen2 import Qwen2ForCausalLM

from .weight_utils import (copy_qkv_proj_weight, copy_qkv_proj_bias, 
                           copy_gate_up_proj_weight, copy_single_proj_col,
                           copy_single_proj_row)

class MixtralMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.experts = FusedMoE(num_experts=config.num_local_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=self.hidden_size,
                                intermediate_size=config.intermediate_size,
                                reduce_results=True,
                                renormalize=True)
        self.gate = nn.Linear(config.hidden_size,
                              config.num_local_experts,
                              bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)

        return final_hidden_states.view(orig_shape)

class MixtralAttention(Qwen2Attention):
    def __init__(self, layer_id, config):
        super().__init__(layer_id, config, qkv_bias=False)

class MixtralDecoderLayer(nn.Module):
    
    def __init__(self, layer_id, config):
        super().__init__()
        
        self.self_attn = MixtralAttention(layer_id, config)
        self.block_sparse_moe = MixtralMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)
    
    def forward(self, input_data: InputData, 
                hidden_states: torch.Tensor, 
                residual: Optional[torch.Tensor]):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(input_data, hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
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
            mp_load_progress[get_local_rank()*2] = len(parameters)
            mp_load_progress[get_local_rank()*2+1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))
            
        attn = self.model.layers[0].self_attn
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim

        intermediate_size = self.config.intermediate_size
        intermediate_size_partition = intermediate_size // get_tp_size()
        
        vocab_size_partition = self.config.vocab_size // get_tp_size()
        
        num_experts = self.config.num_local_experts
        
        _, expert_map = determine_expert_map(get_ep_size(), get_ep_rank(), num_experts)
        
        for k, v in parameters.items():
            k = resolve_pp_layer_idx(k, 2, self.model.start_layer)
            if k.find('self_attn.qkv_proj.weight') != -1:
                copy_qkv_proj_weight(v.data, 
                                     weights[k.replace('qkv_proj', 'q_proj')], 
                                     weights[k.replace('qkv_proj', 'k_proj')], 
                                     weights[k.replace('qkv_proj', 'v_proj')],
                                     num_heads, num_kv_heads, head_dim)
            elif k.find('self_attn.qkv_proj.bias') != -1:
                copy_qkv_proj_bias(v.data, 
                                   weights[k.replace('qkv_proj', 'q_proj')], 
                                   weights[k.replace('qkv_proj', 'k_proj')], 
                                   weights[k.replace('qkv_proj', 'v_proj')],
                                   num_heads, num_kv_heads, head_dim)
            elif k.find('w13_weight') != -1: # expert
                for expert_idx in range(num_experts):
                    local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
                    if local_expert_idx == -1:
                        continue
                    copy_gate_up_proj_weight(v.data[local_expert_idx],
                                             weights[k.replace('w13_weight', f'{expert_idx}.w1.weight')],
                                             weights[k.replace('w13_weight', f'{expert_idx}.w3.weight')],
                                             intermediate_size_partition if not is_use_ep() else intermediate_size,
                                             not is_use_ep())
            elif k.find('w2_weight') != -1: # expert
                for expert_idx in range(num_experts):
                    local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
                    if local_expert_idx == -1:
                        continue
                    copy_single_proj_col(v.data[local_expert_idx],
                                         weights[k.replace('w2_weight', f'{expert_idx}.w2.weight')],
                                         intermediate_size_partition if not is_use_ep() else intermediate_size,
                                         not is_use_ep())
            elif k.find('self_attn.o_proj') != -1:
                copy_single_proj_col(v.data, weights[k], num_heads*head_dim)
            elif k.find('embed_tokens') != -1 or k.find('lm_head') != -1:
                copy_single_proj_row(v.data, weights[k], vocab_size_partition)
            else:
                v.data.copy_(weights[k])
            if mp_load_progress is not None:
                mp_load_progress[get_local_rank()*2+1] += 1
            else:
                pbar.update(1)