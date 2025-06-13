import torch

from typing import Optional
from torch import nn

from gllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from gllm.layers.activation import SiluAndMul
from gllm.layers.rotary_embedding import RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.layers.layernorm import RMSNorm
from gllm.input_data import InputData
from gllm.dist_utils import (get_pp_layers, get_pp_rank, get_local_rank, is_last_pp_rank, 
                             resolve_pp_layer, get_tp_size)
from gllm.utils import get_model_load_pbar

from .weight_utils import (copy_qkv_proj_weight, copy_qkv_proj_bias, 
                           copy_gate_up_proj_weight, copy_single_proj)

class Qwen2MLP(nn.Module):

    def __init__(self, config, shared_expert=False):
        super().__init__()
        if not shared_expert:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = config.shared_expert_intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, 
                                                       [intermediate_size]*2, 
                                                       bias=False)
        
        self.down_proj = RowParallelLinear(intermediate_size,
                                           config.hidden_size,
                                           bias=False)

        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2Attention(nn.Module):
    def __init__(self, layer_id: int, config, qkv_bias=True):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        
        tp_size = get_tp_size()
        
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        
        self.head_dim = self.hidden_size // self.total_num_heads
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = getattr(config,'rope_theta',10000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, self.max_position_embeddings, self.rope_theta, True)
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size)

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(input_data.positions, q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, config, attention_type=Qwen2Attention, mlp_type=Qwen2MLP):
        super().__init__()
        self.self_attn = attention_type(layer_id, config)
        self.mlp = mlp_type(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps)

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
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    def __init__(self, config, decoder_layer_type=Qwen2DecoderLayer):
        super().__init__()
        if get_pp_rank() == 0 or config.tie_word_embeddings and is_last_pp_rank():
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size)
        self.start_layer, self.end_layer = get_pp_layers(
            config.num_hidden_layers)
        
        self.layers = nn.ModuleList([
            decoder_layer_type(i-self.start_layer, config)
            for i in range(self.start_layer, self.end_layer)
        ])
        if is_last_pp_rank():
            self.norm = RMSNorm(
                config.hidden_size, config.rms_norm_eps)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        if get_pp_rank() == 0:
            hidden_states = self.embed_tokens(input_data.tokens)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                input_data, hidden_states, residual)
        if is_last_pp_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states, residual


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config, model_type=Qwen2Model):
        super().__init__()
        self.config = config
        self.max_model_len = config.max_position_embeddings
        self.dtype = config.torch_dtype
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.model = model_type(config)
        self.num_layers = len(self.model.layers)
        self.ret_residual = True
        if is_last_pp_rank():
            if config.tie_word_embeddings:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                self.lm_head.weight = self.model.embed_tokens.weight
            else:
                self.lm_head = nn.Linear(
                    config.hidden_size, config.vocab_size, 
                    bias=False)
    
    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.model(input_data, hidden_states, residual)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        # fetch hidden_states of last token in each seq
        idx_list = input_data.query_start_loc - 1
        return self.lm_head(hidden_states[idx_list[1:]])

    def load_weights(self, weights, mp_load_progress=None):
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank()*2] = len(parameters)
            mp_load_progress[get_local_rank()*2+1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))

        tp_size = get_tp_size()
        
        total_num_heads = self.config.num_attention_heads
        head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // total_num_heads)
        assert total_num_heads % tp_size == 0
        num_heads = total_num_heads // tp_size
        
        total_num_kv_heads = self.config.num_key_value_heads

        assert total_num_kv_heads % tp_size == 0
        
        num_kv_heads = total_num_kv_heads // tp_size
        
        intermediate_size_partition = self.config.intermediate_size // tp_size
        
        for k, v in parameters.items():
            k = resolve_pp_layer(k, 2, self.model.start_layer)
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
            elif k.find('self_attn.o_proj') != -1:
                copy_single_proj(v.data, weights[k], num_heads*head_dim)
            elif k.find('gate_up_proj') != -1:
                copy_gate_up_proj_weight(v.data,
                                         weights[k.replace('gate_up_proj', 'gate_proj')],
                                         weights[k.replace('gate_up_proj', 'up_proj')],
                                         intermediate_size_partition)
            elif k.find('down_proj') != -1:
                copy_single_proj(v.data, weights[k], intermediate_size_partition)
            else:
                v.data.copy_(weights[k])
            if mp_load_progress is not None:
                mp_load_progress[get_local_rank()*2+1] += 1
            else:
                pbar.update(1)
