import torch
from typing import List

from gllm.allocatorID import AllocatorID
from gllm.sequence import Sequence
from gllm.input_data import InputData


class MemoryManager():
    def __init__(self, num_layers: int, page_num_segment=1024, token_num_page=16, kv_head_num=8, kv_head_dim=128):
        '''
        num_layers: number of hidden layers
        page_num_segment: number of pages in a segment
        token_num_page: number of tokens in a page
        kv_head_num: number of k/v heads
        kv_head_dim: dimension of k/v head
        '''
        self.num_layers = num_layers
        self.page_num_segment = page_num_segment
        self.token_num_page = token_num_page
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim
        # free_mem_size, _ = torch.cuda.mem_get_info()
        # num_max_pages = free_mem_size // (
        #     token_num_page*kv_head_num*kv_head_dim*2*2*32)
        # print(num_max_pages)
        self.segments = [
            Segment(num_layers, page_num_segment, token_num_page, kv_head_num, kv_head_dim, torch.bfloat16)]

    def store(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, input_data: InputData):
        cu_seqs_len = 0
        for seq in input_data.seqs:
            # prompt KV cache
            if len(seq.page_table) == 0:
                for i in range(0, seq.prompt_len, self.token_num_page):
                    page_num = i // self.token_num_page
                    idx_right = min(seq.prompt_len, i+self.token_num_page)
                    self.segments[seq.segment_id].k_cache[layer_idx][page_num][0:idx_right-i].copy_(
                        k_cache[i+cu_seqs_len:idx_right+cu_seqs_len])
                    self.segments[seq.segment_id].v_cache[layer_idx][page_num][0:idx_right-i].copy_(
                        v_cache[i+cu_seqs_len:idx_right+cu_seqs_len])
                cu_seqs_len += seq.prompt_len
            # decode KV cache
            else:
                if (len(seq.token_ids)-1) % self.token_num_page == 0:
                    page_num = seq.page_table[-1]
                    self.segments[seq.segment_id].k_cache[layer_idx][page_num][0].copy_(
                        k_cache[cu_seqs_len])
                    self.segments[seq.segment_id].v_cache[layer_idx][page_num][0].copy_(
                        v_cache[cu_seqs_len])
                else:
                    offset = len(seq.token_ids) % self.token_num_page - 1
                    page_num = seq.page_table[-1]
                    self.segments[seq.segment_id].k_cache[layer_idx][page_num][offset].copy_(
                        k_cache[cu_seqs_len])
                    self.segments[seq.segment_id].v_cache[layer_idx][page_num][offset].copy_(
                        v_cache[cu_seqs_len])
                cu_seqs_len += 1

    def pre_allocate_page(self, input_data: InputData):
        for seq in input_data.seqs:
            if not input_data.computed_prompt:
                assert len(seq.page_table) == 0
                num_page = (seq.prompt_len + self.token_num_page -
                            1) // self.token_num_page
                for _ in range(num_page):
                    seq.page_table.append(
                        self.segments[seq.segment_id].allocate())
            else:
                assert len(seq.page_table) != 0
                if (len(seq.token_ids)-1) % self.token_num_page == 0:
                    seq.page_table.append(
                        self.segments[seq.segment_id].allocate())

    def free(self, seq: Sequence):
        for page_num in seq.page_table:
            self.segments[seq.segment_id].free(page_num)


class Segment():
    def __init__(self,
                 num_layers: int,
                 page_num_segment: int,
                 token_num_page: int,
                 kv_head_num: int,
                 kv_head_dim: int,
                 dtype: torch.dtype):
        self.num_layers = num_layers
        # We don't need zero initialization here
        self.k_cache = [torch.ones(
            (page_num_segment, token_num_page, kv_head_num, kv_head_dim), dtype=dtype, device='cuda') for _ in range(num_layers)]
        self.v_cache = [torch.ones(
            (page_num_segment, token_num_page, kv_head_num, kv_head_dim), dtype=dtype, device='cuda') for _ in range(num_layers)]
        self.allocatorID = AllocatorID(0, page_num_segment-1)

    def allocate(self):
        pagenum = self.allocatorID.allocate()
        # if self.layer_id == 0:
        #     print(
        #         f'{self.layer_id}: #pages({self.k_cache.shape[0]}) #used_pages({self.allocatorID.get_num_used_ids()}))')
        return pagenum

    def free(self, page_num: int):
        self.allocatorID.free(page_num)
