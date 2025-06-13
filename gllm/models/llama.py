import torch

from typing import Optional
from torch import nn

from gllm.layers.linear import RowParallelLinear, QKVParallelLinear
from gllm.layers.layernorm import RMSNorm
from gllm.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding, Llama3RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.input_data import InputData
from gllm.dist_utils import get_pp_layers, get_pp_rank, is_last_pp_rank, get_tp_size

from .qwen2 import Qwen2MLP, Qwen2ForCausalLM

class LlamaMLP(Qwen2MLP):

    def __init__(self, config):
        super().__init__(config, False)


class LlamaAttention(nn.Module):

    def __init__(self, layer_id: int, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        tp_size = get_tp_size()
        
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = self.hidden_size // self.total_num_heads

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
        
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
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

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
        if is_last_pp_rank():
            self.norm = RMSNorm(
                config.hidden_size, config.rms_norm_eps)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        if get_pp_rank() == 0:
            hidden_states = self.embed_tokens(input_data.tokens)
        for layer in self.layers:
            hidden_states, residual = layer(
                input_data, hidden_states, residual)
        if is_last_pp_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            return hidden_states, residual


class LlamaForCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config, LlamaModel)