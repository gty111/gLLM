from typing import List
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import torch

from gllm.sequence import Sequence
from gllm.input_data import InputData


class FlashAttention():

    def __init__(self,
                 layer_id: int,
                 scaling: float,
                 num_heads: int,
                 num_key_value_heads: int,
                 head_dim: int,
                 hidden_size: int):
        self.scaling = scaling
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                input_data: InputData):

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        input_data.memory_manager.store(
            self.layer_id, k, v, input_data.seqs, input_data.computed_prompt)
        if not input_data.computed_prompt:
            out = flash_attn_varlen_func(q,
                                         k,
                                         v,
                                         cu_seqlens_q=input_data.cu_seqs_len,
                                         cu_seqlens_k=input_data.cu_seqs_len,
                                         max_seqlen_q=input_data.max_seqlen,
                                         max_seqlen_k=input_data.max_seqlen,
                                         softmax_scale=self.scaling,
                                         causal=True)
        else:
            out = flash_attn_with_kvcache(q.unsqueeze(1),
                                          input_data.memory_manager.segments[
                                              input_data.segment_id].k_cache[self.layer_id],
                                          input_data.memory_manager.segments[
                                              input_data.segment_id].v_cache[self.layer_id],
                                          block_table=input_data.block_table,
                                          softmax_scale=self.scaling,
                                          cache_seqlens=input_data.cache_seqs_len,
                                          causal=True).squeeze(1)
        return out.view(-1, self.hidden_size)
