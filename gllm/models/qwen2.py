import torch

from typing import Optional
from torch import nn

from gllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from gllm.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from gllm.layers.activation import SiluAndMul
from gllm.layers.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.layers.layernorm import RMSNorm
from gllm.input_data import InputData
from gllm.dist_utils import (get_pp_layers, get_local_rank, is_last_pp_rank, 
                             resolve_pp_layer_idx, is_first_pp_rank)
from gllm.utils import get_model_load_pbar
from gllm.modules.attention import Attention

from .weight_utils import (copy_qkv_proj, copy_gate_up_proj, 
                           copy_single_proj_dim1, copy_single_proj_dim0)

class Qwen2MLP(nn.Module):

    def __init__(self, config, shared_expert=False):
        super().__init__()
        quant_config = getattr(config, 'quantization_config', None)
        if not shared_expert:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = config.shared_expert_intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, 
                                                       [intermediate_size]*2, 
                                                       bias=False,
                                                       quant_config=quant_config)
        
        self.down_proj = RowParallelLinear(intermediate_size,
                                           config.hidden_size,
                                           bias=False,
                                           quant_config=quant_config)

        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen2Attention(Attention):
    def __init__(self, layer_id: int, config, qkv_bias=True):
        super().__init__(config.num_attention_heads,
                         config.num_key_value_heads,
                         config.hidden_size)

        self.rope_theta = getattr(config,'rope_theta',10000)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        quant_config = getattr(config, 'quantization_config', None)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, self.head_dim, self.max_position_embeddings, 
                self.rope_theta, True)
        else:
            assert 'mrope_section' in rope_scaling
            self.rotary_emb = MRotaryEmbedding(
                self.head_dim, self.head_dim, self.max_position_embeddings,
                self.rope_theta, True, rope_scaling['mrope_section']
            )
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim)

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
        if is_first_pp_rank() or (config.tie_word_embeddings and is_last_pp_rank()):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
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
        if is_first_pp_rank() and hidden_states is None:
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
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
            )
            if config.tie_word_embeddings:
                self.lm_head.tie_weights(self.model.embed_tokens)
    
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

        attn = self.model.layers[0].self_attn
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        
        for k, v in parameters.items():
            k = resolve_pp_layer_idx(k, 2, self.model.start_layer)
            if k.find('self_attn.qkv_proj') != -1:
                head_dim_patch = head_dim if k.find('scale') == -1 or k.find('weight') == -1 else 1
                copy_qkv_proj(v.data, 
                                     weights[k.replace('qkv_proj', 'q_proj')], 
                                     weights[k.replace('qkv_proj', 'k_proj')], 
                                     weights[k.replace('qkv_proj', 'v_proj')],
                                     num_heads, num_kv_heads, head_dim_patch)
            elif k.find('self_attn.o_proj') != -1:
                copy_single_proj_dim1(v.data, weights[k])
            elif k.find('gate_up_proj') != -1:
                copy_gate_up_proj(v.data,
                                         weights[k.replace('gate_up_proj', 'gate_proj')],
                                         weights[k.replace('gate_up_proj', 'up_proj')])
            elif k.find('down_proj') != -1:
                copy_single_proj_dim1(v.data, weights[k])
            elif k.find('embed_tokens') != -1 or k.find('lm_head') != -1:
                copy_single_proj_dim0(v.data, weights[k])
            else:
                v.data.copy_(weights[k])
            if mp_load_progress is not None:
                mp_load_progress[get_local_rank()*2+1] += 1
            else:
                pbar.update(1)
