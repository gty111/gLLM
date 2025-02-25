import torch
import numpy as np

from typing import List

from gllm.utils import async_tensor_h2d
from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager


class InputData():
    def __init__(self, seqs: List[Sequence], memory_manager: MemoryManager):
        memory_manager.pre_allocate_page(seqs)
        self.seqs = seqs
        self.memory_manager = memory_manager
        # we assume all seqs have the same computed_prompt and segment_id
        self.computed_prompt = seqs[0].computed_prompt
        self.segment_id = seqs[0].segment_id
        if not self.computed_prompt:
            tokens_list = []
            self.prefix_prefill = False
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
    
        
