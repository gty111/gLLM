import torch

from typing import Optional
from torch import nn

from gllm.layers.activation import SiluAndMul
from gllm.layers.layernorm import RMSNorm
from gllm.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding, Llama3RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.input_data import InputData
from gllm.dist_utils import get_pp_layers, get_pp_rank, get_local_rank, is_pp_last_rank, resolve_pp_layer
from gllm.utils import get_model_load_pbar


class LlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size*2, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LlamaAttention(nn.Module):

    def __init__(self, layer_id: int, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads+self.num_key_value_heads*2)
                                  * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim,
                                self.hidden_size, bias=False)
        
        self.rope_theta = getattr(config,'rope_theta',10000)
        
        rope_scaling = config.rope_scaling
        if rope_scaling is not None:
            scaling_type = rope_scaling['type'] if 'type' in rope_scaling else rope_scaling['rope_type']
            if scaling_type == 'llama3':
                low_freq_factor = rope_scaling["low_freq_factor"]
                high_freq_factor = rope_scaling["high_freq_factor"]
                original_max_position = rope_scaling[
                    "original_max_position_embeddings"]
                self.rotary_emb = Llama3RotaryEmbedding(
                    self.head_dim, self.head_dim, original_max_position,
                    self.rope_theta, True, rope_scaling['factor'], low_freq_factor, 
                    high_freq_factor, original_max_position)
            elif rope_scaling['type'] == 'linear':
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim, self.head_dim, config.max_position_embeddings,
                    self.rope_theta, True, rope_scaling['factor'])
            else:
                assert 0
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, self.head_dim, config.max_position_embeddings,
                self.rope_theta, True)

        self.scaling = self.head_dim**-0.5

        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_key_value_heads, self.head_dim, self.hidden_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=1)
        q, k = self.rotary_emb(input_data.positions, q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, layer_id: int, config):
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(layer_id, config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, input_data: InputData, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]):
        # residual connection and input layernorm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # self attention
        hidden_states = self.self_attn(input_data, hidden_states)

        # post attention layernorm
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        # mlp
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.start_layer, self.end_layer = get_pp_layers(
            config.num_hidden_layers)
        self.layers = nn.ModuleList([LlamaDecoderLayer(
            layer_id-self.start_layer, config) for layer_id in range(self.start_layer, self.end_layer)])
        if get_pp_rank() == 0:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size)
        if is_pp_last_rank():
            self.norm = RMSNorm(
                config.hidden_size, config.rms_norm_eps)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        if get_pp_rank() == 0:
            hidden_states = self.embed_tokens(input_data.tokens)
        for layer in self.layers:
            hidden_states, residual = layer(
                input_data, hidden_states, residual)
        if is_pp_last_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states, residual


class LlamaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_model_len = config.max_position_embeddings
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.model = LlamaModel(config)
        self.num_layers = len(self.model.layers)
        self.config = config
        self.ret_residual = True
        if is_pp_last_rank():
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.model(input_data, hidden_states, residual)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        # fetch hidden_states of last token in each seq
        idx_list = input_data.query_start_loc - 1
        return self.lm_head(hidden_states[idx_list[1:]])

    def load_weights(self, weights, mp_load_progress):
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank()*2] = len(parameters)
            mp_load_progress[get_local_rank()*2+1] = 0
        else:
            pbar = get_model_load_pbar(len(parameters))

        # assert len(parameters) == len(weights)
        num_attn_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attn_heads
        num_kv_heads = self.config.num_key_value_heads
        intermediate_size = self.config.intermediate_size
        for k, v in parameters.items():
            k = resolve_pp_layer(k, 2, self.model.start_layer)
            if k.find('self_attn.qkv_proj') != -1:
                v.data[:num_attn_heads*head_dim, :] = weights[k.replace(
                    'qkv_proj', 'q_proj')]
                v.data[num_attn_heads*head_dim:(num_attn_heads +
                       num_kv_heads)*head_dim, :] = weights[k.replace('qkv_proj', 'k_proj')]
                v.data[(num_attn_heads +
                       num_kv_heads)*head_dim:, :] = weights[k.replace('qkv_proj', 'v_proj')]
            elif k.find('gate_up_proj') != -1:
                v.data[:intermediate_size, :] = weights[k.replace(
                    'gate_up_proj', 'gate_proj')]
                v.data[intermediate_size:, :] = weights[k.replace(
                    'gate_up_proj', 'up_proj')]
            else:
                v.data.copy_(weights[k])
            if mp_load_progress is not None:
                mp_load_progress[get_local_rank()*2+1] += 1
            else:
                pbar.update(1)
