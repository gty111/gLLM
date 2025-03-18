import torch
import torch.distributed as dist

from typing import List, Set
from logger import logger

from gllm.allocatorID import AllocatorID
from gllm.sequence import Sequence
from gllm.dist_utils import get_pp_rank


class MemoryManager():
    def __init__(self, gpu_memory_util: float, num_layers: int, dtype: torch.dtype,
                 page_size: int, kv_head_num: int, kv_head_dim: int, vocab_size: int):
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
        self.dtype = dtype
        self.vocab_size = vocab_size

        if not dist.is_initialized():
            free_mem_size, _ = torch.cuda.mem_get_info()
            num_max_pages = free_mem_size // (
                2*num_layers*page_size*kv_head_num*kv_head_dim*2)
            self.num_pages = int(num_max_pages * gpu_memory_util)
        else:
            free_mem_size, _ = torch.cuda.mem_get_info()
            num_max_pages = free_mem_size // (
                2*num_layers*page_size*kv_head_num*kv_head_dim*2)
            num_pages = int(num_max_pages * gpu_memory_util)
            num_pages_all = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(num_pages_all, num_pages)
            self.num_pages = min(num_pages_all)

        if get_pp_rank() == 0:
            logger.info(f'Allocate {self.num_pages} pages ({self.page_size} tokens/page)')

        self.segment = Segment(self.num_layers, self.num_pages,
                               self.page_size, self.kv_head_num, self.kv_head_dim, self.dtype)

    def batch_store(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping_tensor: torch.Tensor):
        from gllm import _custom_ops as ops
        ops.reshape_and_cache_flash(k_cache,
                                    v_cache,
                                    self.segment.k_cache[layer_idx],
                                    self.segment.v_cache[layer_idx],
                                    slot_mapping_tensor)

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            num_page = (len(seq.token_ids) + self.page_size - 1) // self.page_size - len(seq.page_table)
            for _ in range(num_page):
                seq.page_table.append(
                    self.segment.allocate())

    def free(self, seq: Sequence):
        for page_num in seq.page_table:
            self.segment.free(page_num)

    def get_num_free_pages(self):
        return self.segment.get_num_free_pages()

    def get_memory_util(self):
        return self.segment.get_memory_util()
    
    def get_memory_free(self):
        return self.get_num_free_pages() / self.num_pages


class Segment():
    def __init__(self,
                 num_layers: int,
                 num_pages: int,
                 page_size: int,
                 kv_head_num: int,
                 kv_head_dim: int,
                 dtype: torch.dtype):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim
        # We don't need zero initialization here
        self.k_cache = [torch.ones(
            (num_pages, page_size, kv_head_num, kv_head_dim), dtype=dtype, device='cuda') for _ in range(num_layers)]
        self.v_cache = [torch.ones(
            (num_pages, page_size, kv_head_num, kv_head_dim), dtype=dtype, device='cuda') for _ in range(num_layers)]
        self.allocatorID = AllocatorID(0, num_pages-1)

    def allocate(self):
        pagenum = self.allocatorID.allocate()
        return pagenum

    def free(self, page_num: int):
        self.allocatorID.free(page_num)

    def get_num_free_pages(self):
        return self.allocatorID.get_num_free_ids()

    # return percent of used memory
    def get_memory_util(self):
        return round(100 * self.allocatorID.get_num_used_ids()/self.allocatorID.size, 2)


class PrefixMemoryManager(MemoryManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.segment
        self.segment = PrefixSegment(self.num_layers, self.num_pages, self.page_size, self.kv_head_num, self.kv_head_dim, self.dtype)
        
        # for prefill stage
        self.num_allocated_pages = 0
        self.num_hit_pages = 0

    def batch_store(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping_tensor: torch.Tensor):
        from gllm import _custom_ops as ops
        ops.reshape_and_cache_flash(k_cache,
                                    v_cache,
                                    self.segment.k_cache[layer_idx],
                                    self.segment.v_cache[layer_idx],
                                    slot_mapping_tensor)

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            len_page_table = len(seq.page_table)
            num_page = (len(seq.token_ids) + self.page_size - 1) // self.page_size - len_page_table
            if not seq.computed_prompt():
                self.num_allocated_pages += num_page
            computed_prefix = True
            for i in range(len_page_table,len_page_table+num_page):
                if computed_prefix and (i+1)*self.page_size <= len(seq.token_ids):
                    computed_prefix, page_num = self.segment.allocate(
                            (*seq.token_ids[:(i+1)*self.page_size],))
                    seq.computed_token_num += int(computed_prefix) * self.page_size
                    if not seq.computed_prompt():
                        self.num_hit_pages += int(computed_prefix)
                elif (i+1)*self.page_size <= len(seq.token_ids):
                    _, page_num = self.segment.allocate(
                            (*seq.token_ids[:(i+1)*self.page_size],))
                else:
                    _, page_num = self.segment.allocate()
                seq.page_table.append(page_num)
    
    def get_cache_hit_rate(self):
        return round(100 * self.num_hit_pages/self.num_allocated_pages, 2)


class PrefixSegment(Segment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash2page = {}
        self.page_ref_num = [0 for _ in range(self.num_pages)]
        self.page2hash = [0 for _ in range(self.num_pages)]

    def update(self, token_ids: Set[int], page_num: int):
        '''update page hash
        '''
        page_hash = hash(token_ids)
        if page_hash not in self.hash2page:
            self.page2hash[page_num] = page_hash
            self.hash2page[page_hash] = page_num

    def allocate(self, token_ids: Set[int] = None):
        page_hash = hash(token_ids) if token_ids is not None else None
        if page_hash is not None and page_hash in self.hash2page:
            page_num = self.hash2page[page_hash]
            # print(f'reuse {page_num}')
            self.allocatorID.allocate(page_num)
            computed = True
        else:
            page_num = self.allocatorID.allocate()
            # print(f'allocate {page_num}')
            if self.page2hash[page_num] != 0 and self.page2hash[page_num] in self.hash2page:
                del self.hash2page[self.page2hash[page_num]]
            if page_hash is not None:
                self.page2hash[page_num] = page_hash
                self.hash2page[page_hash] = page_num
            computed = False
        self.page_ref_num[page_num] += 1
        return computed, page_num

    def free(self, page_num: int):
        assert self.page_ref_num[page_num] > 0
        self.page_ref_num[page_num] -= 1
        if self.page_ref_num[page_num] == 0:
            # print(f'free {page_num}')
            self.allocatorID.free(page_num)
