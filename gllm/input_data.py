import torch
import numpy as np

from typing import List

from gllm.dist_utils import get_pp_size, get_pp_rank
from gllm.utils import async_tensor_h2d
from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager, PrefixMemoryManager


class InputData():
    def __init__(self, seqs: List[Sequence], memory_manager: MemoryManager):
        if len(seqs) == 0:
            return
        if get_pp_rank() == get_pp_size() - 1:
            self.temperature = async_tensor_h2d([seq.temperature if seq.temperature > 1e-5 else 1 for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.top_p = async_tensor_h2d([seq.top_p for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.top_k = async_tensor_h2d([seq.top_k if seq.top_k != -1 else memory_manager.vocab_size for seq in seqs], memory_manager.dtype, 'cuda', True)
        if get_pp_rank() == 0:
            memory_manager.pre_allocate_page(seqs)
        self.seqs = seqs
        self.memory_manager = memory_manager
        self.page_size = memory_manager.page_size
        # we assume all seqs have the same computed_prompt and segment_id
        self.computed_prompt = seqs[0].computed_prompt
        self.prefix_prefill = False
        self.segment_id = seqs[0].segment_id
        self.slot_mapping_tensor = self.get_slot_mapping()
        if not self.computed_prompt:
            tokens_list = []
            for seq in seqs:
                tokens_list.extend(
                    seq.token_ids[seq.computed_page_num*self.memory_manager.page_size:])
                self.prefix_prefill |= seq.computed_page_num != 0
            self.tokens = async_tensor_h2d(
                tokens_list, torch.long, 'cuda', True)
            positions_list = []
            for seq in seqs:
                positions_list.extend(
                    list(range(seq.computed_page_num*self.memory_manager.page_size, seq.prompt_len)))
            self.positions = async_tensor_h2d(
                positions_list, torch.long, 'cuda', True)
            self.max_seq_len, self.seq_start_loc = self.get_seq_len_loc()
            if self.prefix_prefill:
                self.block_table = self.get_block_table()
                self.max_query_len, self.query_start_loc = self.get_query_len_loc()
        else:
            self.tokens = async_tensor_h2d(
                [seq.token_ids[-1] for seq in seqs], torch.long, 'cuda', True)
            self.positions = async_tensor_h2d(
                [len(seq.token_ids) for seq in seqs], torch.long, 'cuda', True)
            self.cache_seqs_len = async_tensor_h2d(
                [len(seq.token_ids) for seq in self.seqs], torch.int32, 'cuda', True)
            self.block_table = self.get_block_table()

        assert self.tokens.shape == self.positions.shape

    def get_seq_len_loc(self):
        max_seqlen = 0
        cu_seqs_len_num = 0
        seq_start_loc = [0]
        for seq in self.seqs:
            cu_seqs_len_num += seq.prompt_len
            seq_start_loc.append(cu_seqs_len_num)
            max_seqlen = max(seq.prompt_len, max_seqlen)
        return max_seqlen, async_tensor_h2d(seq_start_loc, torch.int32, 'cuda', True)

    def get_query_len_loc(self):
        max_query_len = 0
        cu_query_len = 0
        query_start_loc = [0]
        for seq in self.seqs:
            query_len = seq.prompt_len - \
                seq.computed_page_num * self.memory_manager.page_size
            cu_query_len += query_len
            query_start_loc.append(cu_query_len)
            max_query_len = max(query_len, max_query_len)
        return max_query_len, async_tensor_h2d(query_start_loc, torch.int32, 'cuda', True)

    def get_block_table(self):
        block_tables_list = [seq.page_table for seq in self.seqs]
        max_num_block = max(map(len, block_tables_list))
        block_tables = np.full(
            (len(block_tables_list), max_num_block), 0, dtype=np.int32)
        for idx,block_table in enumerate(block_tables_list):
            block_tables[idx, :len(block_table)] = block_table
        return torch.from_numpy(block_tables).to('cuda')

    def get_slot_mapping(self):
        slot_mapping = []
        if isinstance(self.memory_manager, PrefixMemoryManager):
            for seq in self.seqs:
                # prompt KV cache
                if not self.computed_prompt:
                    for i in range(seq.computed_page_num*self.page_size, seq.prompt_len, self.page_size):
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
                        if offset == self.page_size - 1:
                            self.memory_manager.segments[seq.segment_id].update(
                                (*seq.token_ids[-self.page_size:],), page_num)
                        slot_mapping.append(page_num*self.page_size+offset)
        
        else:
            for seq in self.seqs:
                # prompt KV cache
                if not self.computed_prompt:
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

        return async_tensor_h2d(
            slot_mapping, torch.int64, 'cuda', True)
