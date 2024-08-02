import torch
from typing import List
from logger import logger

from gllm.allocatorID import AllocatorID
from gllm.sequence import Sequence


class MemoryManager():
    def __init__(self, gpu_memory_utilization: float, num_layers: int, dtype: torch.dtype, page_size: int, kv_head_num: int, kv_head_dim: int):
        '''
        num_layers: number of hidden layers
        page_size: number of tokens in a page
        kv_head_num: number of k/v heads
        kv_head_dim: dimension of k/v head
        '''
        self.num_layers = num_layers
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim
        free_mem_size, _ = torch.cuda.mem_get_info()
        num_max_pages = free_mem_size // (
            2*num_layers*page_size*kv_head_num*kv_head_dim*2)
        self.page_num_segment = int(num_max_pages * gpu_memory_utilization)
        logger.info(f'Allocate {self.page_num_segment} pages')
        self.segments = [
            Segment(num_layers, self.page_num_segment, page_size, kv_head_num, kv_head_dim, dtype)]

    def batch_store(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, seqs: List[Sequence], computed_prompt: bool):
        slot_mapping = []
        for seq in seqs:
            # prompt KV cache
            if not computed_prompt:
                for i in range(0, seq.prompt_len, self.page_size):
                    page_num = seq.page_table[i // self.page_size]
                    idx_right = min(seq.prompt_len, i+self.page_size)
                    slot_mapping.extend(list(
                        range(page_num*self.page_size, page_num*self.page_size+idx_right-i)))
            # decode KV cache
            else:
                if (len(seq.token_ids)-1) % self.page_size == 0:
                    page_num = seq.page_table[-1]
                    slot_mapping.append(page_num*self.page_size)
                else:
                    offset = (len(seq.token_ids) - 1) % self.page_size
                    page_num = seq.page_table[-1]
                    slot_mapping.append(page_num*self.page_size+offset)

        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.int64, device='cuda')

        from gllm import _custom_ops as ops
        ops.reshape_and_cache_flash(k_cache,
                                    v_cache,
                                    self.segments[0].k_cache[layer_idx],
                                    self.segments[0].v_cache[layer_idx],
                                    slot_mapping_tensor)

    def store(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, seqs: List[Sequence], computed_prompt: bool):
        cu_seqs_len = 0
        for seq in seqs:
            # prompt KV cache
            if not computed_prompt:
                for i in range(0, seq.prompt_len, self.page_size):
                    page_num = seq.page_table[i // self.page_size]
                    idx_right = min(seq.prompt_len, i+self.page_size)
                    self.segments[seq.segment_id].k_cache[layer_idx][page_num][0:idx_right-i].copy_(
                        k_cache[i+cu_seqs_len:idx_right+cu_seqs_len])
                    self.segments[seq.segment_id].v_cache[layer_idx][page_num][0:idx_right-i].copy_(
                        v_cache[i+cu_seqs_len:idx_right+cu_seqs_len])
                cu_seqs_len += seq.prompt_len
            # decode KV cache
            else:
                if (len(seq.token_ids)-1) % self.page_size == 0:
                    page_num = seq.page_table[-1]
                    self.segments[seq.segment_id].k_cache[layer_idx][page_num][0].copy_(
                        k_cache[cu_seqs_len])
                    self.segments[seq.segment_id].v_cache[layer_idx][page_num][0].copy_(
                        v_cache[cu_seqs_len])
                else:
                    offset = len(seq.token_ids) % self.page_size - 1
                    page_num = seq.page_table[-1]
                    self.segments[seq.segment_id].k_cache[layer_idx][page_num][offset].copy_(
                        k_cache[cu_seqs_len])
                    self.segments[seq.segment_id].v_cache[layer_idx][page_num][offset].copy_(
                        v_cache[cu_seqs_len])
                cu_seqs_len += 1

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            if not seqs[0].computed_prompt:
                assert len(seq.page_table) == 0
                num_page = (seq.prompt_len + self.page_size -
                            1) // self.page_size
                for _ in range(num_page):
                    seq.page_table.append(
                        self.segments[seq.segment_id].allocate())
            else:
                assert len(seq.page_table) != 0
                if (len(seq.token_ids)-1) % self.page_size == 0:
                    seq.page_table.append(
                        self.segments[seq.segment_id].allocate())

    def free(self, seq: Sequence):
        for page_num in seq.page_table:
            self.segments[seq.segment_id].free(page_num)

    def get_num_free_pages(self):
        return self.segments[0].get_num_free_pages()

    def get_memory_util(self):
        return self.segments[0].get_memory_util()


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
        return pagenum

    def free(self, page_num: int):
        self.allocatorID.free(page_num)

    def get_num_free_pages(self):
        return self.allocatorID.get_num_free_ids()

    # return percent of used memory
    def get_memory_util(self):
        return round(100 * self.allocatorID.get_num_used_ids()/self.allocatorID.get_num_ids(), 2)
