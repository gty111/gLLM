import torch
import tqdm
from typing import Optional
from torch import nn

from gllm.layers.activation import SiluAndMul
from gllm.layers.layernorm import RMSNorm
from gllm.layers.rotary_embedding import RotaryEmbedding, LinearScalingRotaryEmbedding, Llama3RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.input_data import InputData
from gllm.layers.sampler import Sampler


class LlamaMLP(nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        self.hidden_size = model_config['hidden_size']
        self.intermediate_size = model_config['intermediate_size']
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size*2, bias=False, dtype=model_config['torch_dtype'], device='cuda')
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=model_config['torch_dtype'], device='cuda')
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LlamaAttention(nn.Module):

    def __init__(self, layer_id: int, model_config: dict):
        super().__init__()
        self.hidden_size = model_config['hidden_size']
        self.num_heads = model_config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = model_config['num_key_value_heads']

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads+self.num_key_value_heads*2)
                                  * self.head_dim, bias=False, dtype=model_config['torch_dtype'], device='cuda')
        self.o_proj = nn.Linear(self.num_heads*self.head_dim,
                                self.hidden_size, bias=False, dtype=model_config['torch_dtype'], device='cuda')

        rope_scaling = model_config['rope_scaling']
        if rope_scaling is not None:
            scaling_type = rope_scaling['type'] if 'type' in rope_scaling else rope_scaling['rope_type']
            if scaling_type == 'llama3':
                low_freq_factor = rope_scaling["low_freq_factor"]
                high_freq_factor = rope_scaling["high_freq_factor"]
                original_max_position = rope_scaling[
                    "original_max_position_embeddings"]
                self.rotary_emb = Llama3RotaryEmbedding(
                    self.head_dim, self.head_dim, original_max_position,
                    model_config['rope_theta'], True, model_config['torch_dtype'],
                    rope_scaling['factor'], low_freq_factor, high_freq_factor, original_max_position)
            elif rope_scaling['type'] == 'linear':
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim, self.head_dim, model_config['max_position_embeddings'],
                    model_config['rope_theta'], True, rope_scaling['factor'], model_config['torch_dtype'])
            else:
                assert 0
        else:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, self.head_dim, model_config['max_position_embeddings'],
                model_config['rope_theta'], True, model_config['torch_dtype'])

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

    def __init__(self, layer_id: int, model_config: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(
            model_config['hidden_size'], model_config['rms_norm_eps'], model_config['torch_dtype'])
        self.self_attn = LlamaAttention(layer_id, model_config)
        self.post_attention_layernorm = RMSNorm(
            model_config['hidden_size'], model_config['rms_norm_eps'], model_config['torch_dtype'])
        self.mlp = LlamaMLP(model_config)

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

    def __init__(self, model_config: dict):
        super().__init__()
        self.layers = nn.ModuleList([LlamaDecoderLayer(layer_id, model_config) for layer_id in range(
            model_config['num_hidden_layers'])])
        self.embed_tokens = nn.Embedding(
            model_config['vocab_size'], model_config['hidden_size'], dtype=model_config['torch_dtype'], device='cuda')
        self.norm = RMSNorm(
            model_config['hidden_size'], model_config['rms_norm_eps'], model_config['torch_dtype'])

    def forward(self, input_data: InputData):
        hidden_states = self.embed_tokens(input_data.tokens)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                input_data, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        self.max_model_len = model_config['max_position_embeddings']
        self.num_layers = model_config['num_hidden_layers']
        self.dtype = model_config['torch_dtype']
        self.num_kv_heads = model_config['num_key_value_heads']
        self.head_dim = model_config['hidden_size'] // model_config['num_attention_heads']
        if model_config['eos_token_id'] == 128001:
            # Llama3-8b-chat
            self.finish_tokens = [model_config['eos_token_id'], 128009]
        else:
            if type(model_config['eos_token_id']) == int:
                self.finish_tokens = [model_config['eos_token_id']]
            elif type(model_config['eos_token_id']) == list:
                self.finish_tokens = model_config['eos_token_id']
            else:
                assert 0
        self.model = LlamaModel(model_config)
        self.model_config = model_config
        self.lm_head = nn.Linear(
            model_config['hidden_size'], model_config['vocab_size'], bias=False, dtype=model_config['torch_dtype'], device='cuda')
        self.sampler = Sampler()

    def forward(self, input_data: InputData):
        return self.model(input_data)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        if input_data.computed_prompt:
            return self.lm_head(hidden_states)
        else:
            # fetch hidden_states of last token in each seq
            if input_data.prefix_prefill:
                idx_list = input_data.query_start_loc - 1
            else:
                idx_list = input_data.seq_start_loc - 1
            return self.lm_head(hidden_states[idx_list[1:]])

    def sample(self, input_data: InputData, logits: torch.Tensor):
        return self.sampler.forward(logits, input_data)

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())

        # assert len(parameters) == len(weights)
        num_attn_heads = self.model_config['num_attention_heads']
        head_dim = self.model_config['hidden_size'] // num_attn_heads
        num_kv_heads = self.model_config['num_key_value_heads']
        intermediate_size = self.model_config['intermediate_size']
        for k, v in tqdm.tqdm(parameters.items()):
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
