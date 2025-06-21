import torch
import torch.nn.functional as F

from typing import Optional
from torch import nn

from gllm.layers.layernorm import RMSNorm
from gllm.layers.moe.fused_moe_triton.layer import FusedMoE, determine_expert_map
from gllm.input_data import InputData
from gllm.dist_utils import (get_local_rank, resolve_pp_layer_idx, get_tp_size, 
                             tensor_model_parallel_all_reduce, is_use_ep,
                             get_ep_rank, resolve_ep_expert_idx, get_ep_size)
from gllm.utils import get_model_load_pbar

from .qwen2 import Qwen2MLP as Qwen2MoeMLP
from .qwen2 import Qwen2Attention as Qwen2MoeAttention
from .qwen2 import Qwen2Model
from .qwen2 import Qwen2ForCausalLM

from .weight_utils import (copy_qkv_proj_weight, copy_qkv_proj_bias, 
                           copy_gate_up_proj_weight, copy_single_proj_col,
                           copy_single_proj_row)

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tp_size = get_tp_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")
        
        self.experts = FusedMoE(num_experts=config.num_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=False,
                                renormalize=config.norm_topk_prob)
        self.gate = nn.Linear(config.hidden_size,
                              config.num_experts,
                              bias=False)
        
        if getattr(config, 'shared_expert_intermediate_size', 0) > 0:
            self.shared_expert = Qwen2MoeMLP(config, shared_expert=True)
        else:
            self.shared_expert = None
        if hasattr(config, 'shared_expert_intermediate_size'):
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size,
                                                    1,
                                                    bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)
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
        attn_type=Qwen2MoeAttention
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attn_type(layer_id, config)

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_id not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_id + 1) % config.decoder_sparse_step == 0):
            self.mlp = moe_block_type(config=config)
        else:
            self.mlp = mlp_type(config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            input_data=input_data,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
class Qwen2MoeModel(Qwen2Model):
    def __init__(self, config, decoder_layer_type=Qwen2MoeDecoderLayer):
        super().__init__(config, decoder_layer_type)

class Qwen2MoeForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, model_type=Qwen2MoeModel):
        super().__init__(config, model_type)
        
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

        intermediate_size = self.config.moe_intermediate_size
        intermediate_size_partition = intermediate_size // get_tp_size()
        shared_expert_intermediate_size_partition = getattr(self.config, 'shared_expert_intermediate_size', None)
        if shared_expert_intermediate_size_partition:
            shared_expert_intermediate_size_partition = shared_expert_intermediate_size_partition // get_tp_size()
        vocab_size_partition = self.config.vocab_size // get_tp_size()
        _, expert_map = determine_expert_map(get_ep_size(), get_ep_rank(), self.config.num_experts)
        
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
                for expert_idx in range(self.config.num_experts):
                    local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
                    if local_expert_idx == -1:
                        continue
                    copy_gate_up_proj_weight(v.data[local_expert_idx],
                                             weights[k.replace('w13_weight', f'{expert_idx}.gate_proj.weight')],
                                             weights[k.replace('w13_weight', f'{expert_idx}.up_proj.weight')],
                                             intermediate_size_partition if not is_use_ep() else intermediate_size,
                                             not is_use_ep())
            elif k.find('w2_weight') != -1: # expert
                for expert_idx in range(self.config.num_experts):
                    local_expert_idx = resolve_ep_expert_idx(expert_idx, expert_map)
                    if local_expert_idx == -1:
                        continue
                    copy_single_proj_col(v.data[local_expert_idx],
                                         weights[k.replace('w2_weight', f'{expert_idx}.down_proj.weight')],
                                         intermediate_size_partition if not is_use_ep() else intermediate_size,
                                         not is_use_ep())
            elif k.find('gate_up_proj.weight') != -1: # shared_expert
                copy_gate_up_proj_weight(v.data,
                                         weights[k.replace('gate_up_proj', 'gate_proj')],
                                         weights[k.replace('gate_up_proj', 'up_proj')],
                                         shared_expert_intermediate_size_partition)
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