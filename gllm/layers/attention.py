from typing import List
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import torch

from gllm.memory_manager import MemoryManager
from gllm.sequence import Sequence
from gllm.input_data import InputData


class FlashAttention():

    def __init__(self, layer_id: int, scaling: float):
        self.scaling = scaling
        self.memory_manager = MemoryManager(layer_id)
        self.layer_id = layer_id

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                input_data: InputData):

        self.memory_manager.store(k, v, input_data.seqs)
        if not input_data.computed_prompt:
            # self.prev_k = k
            # self.prev_v = v
            out = flash_attn_varlen_func(q,
                                         k,
                                         v,
                                         cu_seqlens_q=input_data.cu_seqs_len,
                                         cu_seqlens_k=input_data.cu_seqs_len,
                                         max_seqlen_q=input_data.max_seqlen,
                                         max_seqlen_k=input_data.max_seqlen,
                                         softmax_scale=self.scaling,
                                         causal=True)
            return out
        else:
            # self.prev_k = torch.cat((self.prev_k, k), dim=0)
            # self.prev_v = torch.cat((self.prev_v, v), dim=0)
            # out = flash_attn_with_kvcache(q.unsqueeze(1),
            #                               self.prev_k.unsqueeze(0),
            #                               self.prev_v.unsqueeze(0),
            #                               softmax_scale=self.scaling,
            #                               causal=True)
            block_table = input_data.get_block_table(self.layer_id)
            cache_seqlens = input_data.get_cache_seqlens()
            out = flash_attn_with_kvcache(q.unsqueeze(1),
                                          self.memory_manager.segments[input_data.segment_id].k_cache,
                                          self.memory_manager.segments[input_data.segment_id].v_cache,
                                          block_table=block_table,
                                          softmax_scale=self.scaling,
                                          cache_seqlens=cache_seqlens,
                                          causal=True)
            return out.squeeze(1)

    def free(self, seq: Sequence):
        self.memory_manager.free(seq)
