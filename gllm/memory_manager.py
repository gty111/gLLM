import torch
import torch.distributed as dist

from typing import List, Set
from logger import logger

from gllm.id_allocator import IDAllocator
from gllm.sequence import Sequence
from gllm.utils import get_dtype_bytes


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

        free_mem_size, _ = torch.cuda.mem_get_info()
        num_max_pages = free_mem_size // self.get_sizeof_KV_per_page()
        num_pages = int(num_max_pages * gpu_memory_util)
        
        if not dist.is_initialized():
            self.num_pages = num_pages
        else:
            num_pages_all = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(num_pages_all, num_pages)
            self.num_pages = min(num_pages_all)

        logger.info(f'KV cache: {self.num_pages} pages ({self.page_size} tokens/page), '
                    f'{round(self.get_sizeof_KV_per_page()/(2**10*self.page_size),2)} KB (per token), '
                    f'{round(self.num_pages*self.get_sizeof_KV_per_page()/(2**30),2)} GB (total)')

        self.segment = Segment(self.num_layers, self.num_pages,
                               self.page_size, self.kv_head_num, self.kv_head_dim)

    def get_sizeof_KV_per_page(self): # Bytes
        # 2: K cache and V cache 
        return  2 * self.num_layers * self.page_size * self.kv_head_num * self.kv_head_dim * get_dtype_bytes(self.dtype)  
    
    def batch_store(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping_tensor: torch.Tensor):
        from gllm import _custom_ops as ops
        ops.reshape_and_cache_flash(k_cache,
                                    v_cache,
                                    self.segment.k_cache[layer_idx],
                                    self.segment.v_cache[layer_idx],
                                    slot_mapping_tensor)

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            num_page = (seq.seq_len + self.page_size - 1) // self.page_size - len(seq.page_table)
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
                 kv_head_dim: int):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim
        # We don't need zero initialization here
        self.k_cache = [torch.ones(
            (num_pages, page_size, kv_head_num, kv_head_dim)) for _ in range(num_layers)]
        self.v_cache = [torch.ones(
            (num_pages, page_size, kv_head_num, kv_head_dim)) for _ in range(num_layers)]
        self.id_allocator = IDAllocator(0, num_pages-1)

    def allocate(self):
        pagenum = self.id_allocator.allocate()
        return pagenum

    def free(self, page_num: int):
        self.id_allocator.free(page_num)

    def get_num_free_pages(self):
        return self.id_allocator.get_num_free_ids()

    # return percent of used memory
    def get_memory_util(self):
        return round(100 * self.id_allocator.get_num_used_ids()/self.id_allocator.size, 2)


class PrefixMemoryManager(MemoryManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        logger.info('Enable prefix caching')

        del self.segment
        self.segment = PrefixSegment(self.num_layers, self.num_pages, self.page_size, self.kv_head_num, self.kv_head_dim)
        
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

    def pre_allocate_computed_page(self, seqs: List[Sequence]):
        for seq in seqs:
            assert len(seq.page_table) == 0
            num_page = (len(seq) + self.page_size - 1) // self.page_size 
            if not seq.computed_prompt:
                self.num_allocated_pages += num_page
            for i in range(num_page):
                if (i+1)*self.page_size <= len(seq):
                    page_num = self.segment.has_computed((*seq[:(i+1)*self.page_size],))
                    if page_num is not None:
                        seq.page_table.append(page_num)
                        seq.computed_token_num += self.page_size
                        self.num_hit_pages += 1
                    else:
                        break
                else:
                    break

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            # update hash of newly generated page in decode stage
            if seq.computed_prompt and len(seq) % self.page_size == 0:
                self.segment.update((*seq[:],), seq.page_table[-1])
            len_page_table = len(seq.page_table)
            num_page = (seq.seq_len + self.page_size - 1) // self.page_size - len_page_table
            for i in range(len_page_table,len_page_table+num_page):
                if (i+1)*self.page_size <= len(seq):
                    page_num = self.segment.allocate(
                            (*seq[:(i+1)*self.page_size],))
                else:
                    page_num = self.segment.allocate()
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
            
    def has_computed(self, token_ids):
        page_hash = hash(token_ids)
        if page_hash in self.hash2page:
            page_num = self.hash2page[page_hash]
            self.id_allocator.allocate(page_num)
            # print(f'reuse {page_num}')
            self.page_ref_num[page_num] += 1
            return page_num
        else:
            return None

    def allocate(self, token_ids: Set[int] = None):
        page_hash = hash(token_ids) if token_ids is not None else None
        page_num = self.id_allocator.allocate()
        # print(f'allocate {page_num}')
        if self.page2hash[page_num] != 0 and self.page2hash[page_num] in self.hash2page:
            del self.hash2page[self.page2hash[page_num]]
        if page_hash is not None:
            self.page2hash[page_num] = page_hash
            self.hash2page[page_hash] = page_num
        self.page_ref_num[page_num] += 1
        return page_num

    def free(self, page_num: int):
        assert self.page_ref_num[page_num] > 0
        self.page_ref_num[page_num] -= 1
        if self.page_ref_num[page_num] == 0:
            # print(f'free {page_num}')
            self.id_allocator.free(page_num)
