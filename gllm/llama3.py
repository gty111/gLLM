import torch
import random
from typing import List, Optional
from torch import nn

from gllm.layers.activation import SiluAndMul
from gllm.layers.layernorm import RMSNorm
from gllm.layers.rotary_embedding import RotaryEmbedding
from gllm.layers.attention import FlashAttention
from gllm.sequence import Sequence
from gllm.input_data import InputData



class LlamaMLP(nn.Module):

    def __init__(self, model_config: dict):
        super().__init__()
        self.hidden_size = model_config['hidden_size']
        self.intermediate_size = model_config['intermediate_size']
        # self.gate_proj = nn.Linear(
        #     self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16, device='cuda')
        # self.up_proj = nn.Linear(
        #     self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16, device='cuda')
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size*2, bias=False, dtype=torch.bfloat16, device='cuda')
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=torch.bfloat16, device='cuda')
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        # y1 = self.gate_proj(x)
        # y2 = self.up_proj(x)
        # return self.down_proj(self.act_fn(torch.concat((y1,y2),dim=1)))
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LlamaAttention(nn.Module):

    def __init__(self, layer_id: int, model_config: dict):
        super().__init__()
        self.hidden_size = model_config['hidden_size']
        self.num_heads = model_config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = model_config['num_key_value_heads']

        # self.q_proj = nn.Linear(
        #     self.hidden_size, self.num_heads*self.head_dim, bias=False, dtype=torch.bfloat16, device='cuda')
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads *
        #                         self.head_dim, bias=False, dtype=torch.bfloat16, device='cuda')
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads *
        #                         self.head_dim, bias=False, dtype=torch.bfloat16, device='cuda')
        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads+self.num_key_value_heads*2)
                                  * self.head_dim, bias=False, dtype=torch.bfloat16, device='cuda')
        self.o_proj = nn.Linear(self.num_heads*self.head_dim,
                                self.hidden_size, bias=False, dtype=torch.bfloat16, device='cuda')

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, model_config['max_position_embeddings'], model_config['rope_theta'])

        self.scaling = self.head_dim**-0.5

        self.attn = FlashAttention(layer_id, self.scaling)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        # q = self.q_proj(hidden_states)
        # k = self.k_proj(hidden_states)
        # v = self.v_proj(hidden_states)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=1)
        q = qkv[:, :self.num_heads*self.head_dim]
        k = qkv[:, self.num_heads *
                self.head_dim:(self.num_heads+self.num_key_value_heads)*self.head_dim]
        v = qkv[:, (self.num_heads+self.num_key_value_heads)*self.head_dim:]
        q, k = self.rotary_emb(input_data.positions, q, k)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        attn_output = self.attn.forward(q, k, v, input_data)
        attn_output = attn_output.view(-1, self.hidden_size)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, layer_id: int, model_config: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(
            model_config['hidden_size'], model_config['rms_norm_eps'])
        self.self_attn = LlamaAttention(layer_id, model_config)
        self.post_attention_layernorm = RMSNorm(
            model_config['hidden_size'], model_config['rms_norm_eps'])
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
            model_config['vocab_size'], model_config['hidden_size'], dtype=torch.bfloat16, device='cuda')
        self.norm = RMSNorm(
            model_config['hidden_size'], model_config['rms_norm_eps'])

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
        self.model = LlamaModel(model_config)
        self.model_config = model_config
        self.lm_head = nn.Linear(
            model_config['hidden_size'], model_config['vocab_size'], bias=False, dtype=torch.bfloat16, device='cuda')

    def free_kv_cache(self, seq: Sequence):
        for i in self.model.layers:
            i.self_attn.attn.free(seq)

    def forward(self, input_data: InputData):
        return self.model(input_data)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        if input_data.computed_prompt:
            return self.lm_head(hidden_states)
        else:
            # fetch hidden_states of last token in each seq
            idx_list = input_data.cu_seqs_len - 1
            return self.lm_head(hidden_states[idx_list[1:]])

    def sample(self, logits: torch.Tensor, temperature=0.6, top_p=0.9, top_k=5):
        logits.div_(temperature)
        probs = torch.softmax(logits, dim=1)
        next_tokens = []
        for i in range(probs.shape[0]):
            accum_probs = 0
            candi_tokens = []
            for token in torch.argsort(probs[i], descending=True)[:top_k]:
                candi_tokens.append(token.item())
                accum_probs += probs[i, token]
                if accum_probs >= top_p:
                    break
            next_token = random.sample(candi_tokens, 1)[0]
            next_tokens.append(next_token)
        return next_tokens
