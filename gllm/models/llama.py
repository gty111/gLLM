import torch

from typing import Optional
from torch import nn

from gllm.layers.linear import RowParallelLinear, QKVParallelLinear
from gllm.layers.layernorm import RMSNorm
from gllm.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding, Llama3RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.input_data import InputData
from gllm.dist_utils import get_pp_layers, get_pp_rank, is_last_pp_rank
from gllm.modules.attention import Attention

from .qwen2 import Qwen2MLP, Qwen2ForCausalLM, Qwen2Model

class LlamaMLP(Qwen2MLP):

    def __init__(self, config):
        super().__init__(config, False)


class LlamaAttention(Attention):

    def __init__(self, layer_id: int, config):
        super().__init__(config.num_attention_heads,
                         config.num_key_value_heads,
                         config.hidden_size)

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

        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size)

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


class LlamaModel(Qwen2Model):

    def __init__(self, config):
        super().__init__(config, LlamaDecoderLayer)


class LlamaForCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config, LlamaModel)