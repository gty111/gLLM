from gllm.vllm_flash_attn import flash_attn_varlen_func
import torch

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

        input_data.memory_manager.batch_store(
            self.layer_id, k, v, input_data.slot_mapping_tensor)

        k_cache = input_data.memory_manager.segment.k_cache[self.layer_id]
        v_cache = input_data.memory_manager.segment.v_cache[self.layer_id]

        out = flash_attn_varlen_func(q,
                                     k_cache,
                                     v_cache,
                                     cu_seqlens_q=input_data.query_start_loc,
                                     max_seqlen_q=input_data.max_query_len,
                                     seqused_k=input_data.seq_start_loc,
                                     max_seqlen_k=input_data.max_seq_len,
                                     softmax_scale=self.scaling,
                                     causal=True,
                                     block_table=input_data.block_table)
        return out.view(-1, out.shape[-2]*out.shape[-1])
