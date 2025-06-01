import torch

from torch import nn

from gllm.layers.rotary_embedding import RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.layers.layernorm import RMSNorm
from gllm.input_data import InputData

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .qwen2 import Qwen2DecoderLayer
from .qwen2 import Qwen2ForCausalLM

class Qwen3Attention(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = getattr(config, "rope_theta", 1000000)
        self.qkv_bias = getattr(config, 'attention_bias', False)
        
        self.qkv_proj = nn.Linear(
            self.hidden_size, (self.num_heads+self.num_kv_heads*2)*self.head_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, config.max_position_embeddings, self.rope_theta, True)
        self.attn = FlashAttention(
            layer_id, self.scaling, self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        
    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(input_data.positions, q, k)
        attn_output = self.attn.forward(q, k, v, input_data)
        output = self.o_proj(attn_output)
        return output

class Qwen3DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, layer_id, config):
        super().__init__(layer_id, config, attention_type=Qwen3Attention, mlp_type=Qwen3MLP)
    
class Qwen3Model(Qwen2Model):
    def __init__(self, config):
        super().__init__(config, decoder_layer_type=Qwen3DecoderLayer)
        
class Qwen3ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config, model_type=Qwen3Model)