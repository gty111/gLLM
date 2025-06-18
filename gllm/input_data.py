import torch
import numpy as np

from typing import List

from gllm.dist_utils import is_last_pp_rank
from gllm.utils import async_tensor_h2d
from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager


class InputData():
    def __init__(self, seqs: List[Sequence], memory_manager: MemoryManager):
        assert len(seqs) != 0
        if is_last_pp_rank():
            self.temperature = async_tensor_h2d(
                [seq.temperature if seq.temperature > 1e-5 else 1 for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.top_p = async_tensor_h2d(
                [seq.top_p for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.top_k = async_tensor_h2d(
                [seq.top_k if seq.top_k != -1 else memory_manager.vocab_size for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.repetition_penalty = async_tensor_h2d(
                [seq.repetition_penalty for seq in seqs], memory_manager.dtype, 'cuda', True)
            self.repetition_penalty = self.repetition_penalty.unsqueeze(dim=1).repeat(1,memory_manager.vocab_size)
        
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
            tokens_list.extend(seq[seq.computed_token_num:seq.seq_len])
        return async_tensor_h2d(
            tokens_list, torch.long, 'cuda', True)

    def get_position(self):
        positions_list = []
        for seq in self.seqs:
            positions_list.extend(
                range(seq.computed_token_num, seq.seq_len))
        return async_tensor_h2d(
            positions_list, torch.long, 'cuda', True)

    def get_seq_len_loc(self):
        seq_start_loc = [seq.seq_len for seq in self.seqs]
        max_seqlen = max(seq_start_loc)
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
        return torch.from_numpy(block_tables).to(device='cuda',non_blocking=True)

    def get_slot_mapping(self):
        slot_mapping = []
        for seq in self.seqs:
            for i in range(seq.computed_token_num,seq.seq_len):
                page_idx = i // self.page_size
                slot_idx = i % self.page_size
                slot_mapping.append(seq.page_table[page_idx]*self.page_size+slot_idx)

        return async_tensor_h2d(
            slot_mapping, torch.int64, 'cuda', True)
