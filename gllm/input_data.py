import torch
import numpy as np

from typing import List

from gllm.dist_utils import get_pp_size, get_pp_rank
from gllm.utils import async_tensor_h2d
from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager, PrefixMemoryManager


class InputData():
    def __init__(self, seqs: List[Sequence], memory_manager: MemoryManager):
        assert len(seqs) != 0
        if get_pp_rank() == get_pp_size() - 1:
            self.temperature = async_tensor_h2d(
                [seq.temperature if seq.temperature > 1e-5 else 1 for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.top_p = async_tensor_h2d(
                [seq.top_p for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.top_k = async_tensor_h2d(
                [seq.top_k if seq.top_k != -1 else memory_manager.vocab_size for seq in seqs], memory_manager.dtype, 'cuda', True)
        if get_pp_rank() == 0:
            memory_manager.pre_allocate_page(seqs)
        
        # workaround for setting to_compute_token_num in prefill stage
        # considering prefix caching
        for seq in seqs:
            if seq.to_compute_token_num == 0:
                seq.to_compute_token_num = len(seq.token_ids) - seq.computed_token_num
        
        self.seqs = seqs
        self.memory_manager = memory_manager
        self.page_size = memory_manager.page_size
        self.slot_mapping_tensor = self.get_slot_mapping()
        self.tokens = self.get_tokens()
        self.positions = self.get_position()
        self.max_seq_len, self.seq_start_loc = self.get_seq_len_loc()
        self.block_table = self.get_block_table()
        self.max_query_len, self.query_start_loc = self.get_query_len_loc()

        assert self.tokens.shape == self.positions.shape

    def get_tokens(self):
        tokens_list = []
        for seq in self.seqs:
            tokens_list.extend(
                seq.token_ids[seq.computed_token_num:seq.computed_token_num+seq.to_compute_token_num])
        return async_tensor_h2d(
            tokens_list, torch.long, 'cuda', True)

    def get_position(self):
        positions_list = []
        for seq in self.seqs:
            positions_list.extend(
                range(seq.computed_token_num, seq.computed_token_num+seq.to_compute_token_num))
        return async_tensor_h2d(
            positions_list, torch.long, 'cuda', True)

    def get_seq_len_loc(self):
        max_seqlen = 0
        cu_seqs_len_num = 0
        seq_start_loc = [0]
        for seq in self.seqs:
            seq_len = seq.computed_token_num + seq.to_compute_token_num
            cu_seqs_len_num += seq_len
            seq_start_loc.append(cu_seqs_len_num)
            max_seqlen = max(seq_len, max_seqlen)
        return max_seqlen, async_tensor_h2d(seq_start_loc, torch.int32, 'cuda', True)

    def get_query_len_loc(self):
        max_query_len = 0
        cu_query_len = 0
        query_start_loc = [0]
        for seq in self.seqs:
            query_len = seq.to_compute_token_num
            cu_query_len += query_len
            query_start_loc.append(cu_query_len)
            max_query_len = max(query_len, max_query_len)
        return max_query_len, async_tensor_h2d(query_start_loc, torch.int32, 'cuda', True)

    def get_block_table(self):
        block_tables_list = [seq.page_table for seq in self.seqs]
        max_num_block = max(map(len, block_tables_list))
        block_tables = np.full(
            (len(block_tables_list), max_num_block), 0, dtype=np.int32)
        for idx, block_table in enumerate(block_tables_list):
            block_tables[idx, :len(block_table)] = block_table
        return torch.from_numpy(block_tables).to('cuda')

    def get_slot_mapping(self):
        slot_mapping = []
        for seq in self.seqs:
            for i in range(seq.computed_token_num,seq.computed_token_num+seq.to_compute_token_num):
                page_idx = i // self.page_size
                slot_idx = i % self.page_size
                slot_mapping.append(seq.page_table[page_idx]*self.page_size+slot_idx)
                # for now we only update decode page
                if isinstance(self.memory_manager, PrefixMemoryManager) and seq.to_compute_token_num == 1 and slot_idx==self.page_size-1:
                    self.memory_manager.segment.update((*seq.token_ids[-self.page_size:],),seq.page_table[page_idx])

        return async_tensor_h2d(
            slot_mapping, torch.int64, 'cuda', True)
