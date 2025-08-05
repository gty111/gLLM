import torch
import torch.nn as nn

from typing import Optional

from gllm.layers.moe import FusedMoE
from gllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, ColumnParallelLinear
from gllm.layers.vocab_parallel_embedding import VocabParallelEmbedding
from gllm.layers.activation import SiluAndMul
from gllm.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
from gllm.layers.attention import FlashAttention, MLAAttention
from gllm.layers.layernorm import RMSNorm
from gllm.input_data import InputData
from gllm.modules.attention import Attention
from gllm.utils import yarn_get_mscale
from gllm.dist_utils import (tensor_model_parallel_all_reduce, 
                             get_tp_size, is_first_pp_rank,
                             get_pp_layers, is_last_pp_rank)

from .qwen2_moe import Qwen2MoeForCausalLM

class DeepseekV2MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size, reduce_results: bool = True):
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, 
                                                       [intermediate_size]*2, 
                                                       bias=False)
        
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           reduce_results=reduce_results)

        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))
        
class DeepseekV2MOE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts: int = config.n_shared_experts
        self.gate = nn.Linear(config.hidden_size,
                              config.n_routed_experts,
                              bias=False)
        if config.topk_method == 'noaux_tc':
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None
        
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(config.hidden_size,
                                                intermediate_size,
                                                reduce_results=False)
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)

        if hidden_states.dtype != torch.float16:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits) * self.routed_scaling_factor
        else:
            # Fix FP16 overflow
            # See DeepseekV2DecoderLayer for more details.
            final_hidden_states = self.experts(hidden_states=hidden_states,
                                               router_logits=router_logits)
        if shared_output is not None:
            if hidden_states.dtype != torch.float16:
                final_hidden_states = final_hidden_states + shared_output
            else:
                # Fix FP16 overflow
                # See DeepseekV2DecoderLayer for more details.
                final_hidden_states = final_hidden_states + shared_output \
                    * (1. / self.routed_scaling_factor)

        if get_tp_size() > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

class DeepseekV2Attention(Attention):
    
    def __init__(self, layer_id: int, config):
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        super().__init__(config.num_attention_heads, 
                         config.num_attention_heads,
                         self.hidden_size,
                         self.qk_head_dim)
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank if hasattr(
            config, 'q_lora_rank') else None
        self.kv_lora_rank = config.kv_lora_rank
        self.rope_theta = getattr(config, 'rope_theta', 10000)
        self.max_poistion_embeddings = getattr(
            config, 'max_position_embeddings', 8192)
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size,
                                      self.q_lora_rank,
                                      bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,
                                                 self.total_num_heads * 
                                                 self.qk_head_dim,
                                                 bias=False)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.total_num_heads *
                                               self.qk_head_dim,
                                               bias=False)
        
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False)
        # O projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False)
        
        if self.rope_scaling:
            self.rope_scaling['rope_type'] = 'deepseek_yarn'
            mscale_all_dim = self.rope_scaling.get('mscale_all_dim', False)
            scaling_factor = self.rope_scaling['factor']
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        
        extra_kwargs = {
            k: v
            for k, v in self.rope_scaling.items()
            if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
        }
        
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=self.rope_scaling[
                'original_max_position_embeddings'],
            base=self.rope_theta,
            is_neox_style=False,
            scaling_factor=self.rope_scaling['factor'],
            **extra_kwargs
        )
        
        self.attn = FlashAttention(
            layer_id,
            self.scaling,
            self.num_heads,
            self.num_heads,
            self.qk_head_dim
        )
        
    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor
    ):
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q).view(-1, self.num_heads,
                                         self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states).view(-1, self.num_heads,
                                                   self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)
        kv = kv.view(-1, self.num_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        
        q_pe, k_pe = self.rotary_emb(input_data.positions, q_pe, k_pe)
        
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim],
            value=0).view(-1, self.num_heads * self.qk_head_dim)
        attn_output = self.attn.forward(q, k, v, input_data)
        attn_output = attn_output.view(
            -1, self.num_heads,
            self.qk_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_heads * self.v_head_dim)
        output = self.o_proj(attn_output)
        return output
    
class DeepseekV2MLAAttention(Attention):
    def __init__(self, layer_id: int, config):
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        super().__init__(config.num_attention_heads, 
                         config.num_attention_heads,
                         self.hidden_size,
                         self.qk_head_dim)
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank if hasattr(
            config, 'q_lora_rank') else None
        self.kv_lora_rank = config.kv_lora_rank
        self.rope_theta = getattr(config, 'rope_theta', 10000)
        self.max_poistion_embeddings = getattr(
            config, 'max_position_embeddings', 8192)
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        self.layer_id = layer_id

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = nn.Linear(
                self.hidden_size,
                sum([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim]),
                bias=False)
        else:
            self.kv_a_proj_with_mqa = nn.Linear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False)

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,
                                                 self.total_num_heads *
                                                 self.qk_head_dim,
                                                 bias=False)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.total_num_heads *
                                               self.qk_head_dim,
                                               bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False)
        
        if self.rope_scaling:
            self.rope_scaling['rope_type'] = 'deepseek_yarn'
            mscale_all_dim = self.rope_scaling.get('mscale_all_dim', False)
            scaling_factor = self.rope_scaling['factor']
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        extra_kwargs = {
            k: v
            for k, v in self.rope_scaling.items()
            if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
        }
        
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=self.rope_scaling[
                'original_max_position_embeddings'],
            base=self.rope_theta,
            is_neox_style=False,
            scaling_factor=self.rope_scaling['factor'],
            **extra_kwargs
        )

        self.mla_attn = MLAAttention(
            layer_id=layer_id,
            scale=self.scaling,
            num_heads=self.num_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_key_value_heads=1,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=self.kv_b_proj,
        )
    
    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
    ):
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)
            q = self.q_proj(hidden_states)

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            input_data.positions, q[..., self.qk_nope_head_dim:], k_pe)

        output_shape=(hidden_states.shape[0],
                    self.num_heads * self.v_head_dim)
        output = torch.zeros(output_shape,
                            dtype=q.dtype)

        kv_cache = input_data.memory_manager.segment.kv_cache[self.layer_id]

        attn_out = self.mla_attn.forward(
            q,
            kv_c_normed,
            k_pe,
            kv_cache=kv_cache,
            input_data=input_data,
            output=output,
            )
        return self.o_proj(attn_out)
        

class DeepseekV2DecoderLayer(nn.Module):
    
    def __init__(self, glb_layer_id:int, layer_id:int, config):
        super().__init__()
        if not config.use_mla:
            self.self_attn = DeepseekV2Attention(layer_id, config)
        else:
            self.self_attn = DeepseekV2MLAAttention(layer_id, config)
        
        if(config.n_routed_experts is not None
            and glb_layer_id >= config.first_k_dense_replace
            and glb_layer_id % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MOE(config)
        else:
            self.mlp = DeepseekV2MLP(config.hidden_size,
                                     config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor
        
    def forward(self, input_data:InputData,
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
        hidden_states = self.self_attn(input_data, hidden_states)
        
        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor
        
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp,
                      DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor

        return hidden_states, residual
    
class DeepseekV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        if is_first_pp_rank():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size)
        self.start_layer, self.end_layer = get_pp_layers(config.num_hidden_layers)
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(i, i-self.start_layer, config)
            for i in range(self.start_layer, self.end_layer)
        ])
        
        if is_last_pp_rank():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
    
    def forward(self, input_data:InputData, hidden_states=None, residual=None):
        if is_first_pp_rank():
            hidden_states = self.embed_tokens(input_data.tokens)
        for layer in self.layers:
            hidden_states, residual = layer(
                input_data, hidden_states, residual)
        if is_last_pp_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states, residual
        

class DeepseekV2ForCausalLM(Qwen2MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config, model_type=DeepseekV2Model)
        attn = self.model.layers[0].self_attn
        if not config.use_mla:
            self.head_dim = attn.qk_head_dim
        else:
            self.head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim