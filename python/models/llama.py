import torch

from gllm.runtime.input_data import InputData
from gllm.layers.attention.base import AttentionLayerBase
from gllm.layers.attention.qkv import QKVAttention
from gllm.layers.linear import QKVParallelLinear, RowParallelLinear
from gllm.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    RotaryEmbedding,
)

from .qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model
from .utils import extract_rope_config


class LlamaAttention(AttentionLayerBase):

    def __init__(self, layer_id: int, config):
        super().__init__(
            config.num_attention_heads, config.num_key_value_heads, config.hidden_size
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope_theta, rope_scaling = extract_rope_config(
            config, default_theta=10000.0
        )
        if rope_scaling is not None:
            scaling_type = (
                rope_scaling["type"]
                if "type" in rope_scaling
                else rope_scaling["rope_type"]
            )
            if scaling_type == "llama3":
                low_freq_factor = rope_scaling["low_freq_factor"]
                high_freq_factor = rope_scaling["high_freq_factor"]
                original_max_position = rope_scaling["original_max_position_embeddings"]
                self.rotary_emb = Llama3RotaryEmbedding(
                    self.head_dim,
                    self.head_dim,
                    getattr(config, "model_max_length", config.max_position_embeddings),
                    self.rope_theta,
                    True,
                    rope_scaling["factor"],
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position,
                )
            elif rope_scaling["type"] == "linear":
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim,
                    self.head_dim,
                    config.max_position_embeddings,
                    self.rope_theta,
                    True,
                    rope_scaling["factor"],
                )
            else:
                assert 0
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                self.head_dim,
                config.max_position_embeddings,
                self.rope_theta,
                True,
            )

        self.attn = QKVAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim
        )

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=1)
        q, k = self.rotary_emb(input_data.get_position(), q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(Qwen2DecoderLayer):
    # Qwen2DecoderLayer's default mlp_type is Qwen2MLP(config) ==
    # Qwen2MLP(config, shared_expert=False), which is exactly what Llama wants;
    # only the attention (rope_scaling branches) differs.
    def __init__(self, layer_id: int, config):
        super().__init__(layer_id, config, attention_type=LlamaAttention)


class LlamaModel(Qwen2Model):

    def __init__(self, config):
        super().__init__(config, LlamaDecoderLayer)


class LlamaForCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config, LlamaModel)
