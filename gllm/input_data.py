import torch
from typing import List

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
                self.prefix_prefill |= seq.computed_page_num!=0
            self.tokens = torch.tensor(tokens_list, device='cuda')
            positions_list = []
            for seq in seqs:
                positions_list.extend(
                    list(range(seq.computed_page_num*self.memory_manager.page_size, seq.prompt_len)))
            self.positions = torch.tensor(positions_list, device='cuda')
            self.seq_start_loc = self.get_seq_start_loc()
            self.max_seq_len = self.get_max_seq_len()
            if self.prefix_prefill:
                self.block_table = self.get_block_table()
                self.query_start_loc = self.get_query_start_loc()
                self.max_query_len = self.get_max_query_len()
        else:
            self.tokens = torch.tensor(
                [seq.token_ids[-1] for seq in seqs], device='cuda')
            self.positions = torch.tensor(
                [len(seq.token_ids) for seq in seqs], device='cuda')
            self.cache_seqs_len = torch.tensor(
                [len(seq.token_ids) for seq in self.seqs], dtype=torch.int32, device='cuda')
            self.block_table = self.get_block_table()

        assert self.tokens.shape == self.positions.shape

    def get_seq_start_loc(self):
        cu_seqs_len_num = 0
        seq_start_loc = [0]
        for seq in self.seqs:
            cu_seqs_len_num += seq.prompt_len
            seq_start_loc.append(cu_seqs_len_num)
        return torch.tensor(seq_start_loc, device='cuda', dtype=torch.int32)

    def get_max_seq_len(self):
        max_seqlen = 0
        for seq in self.seqs:
            max_seqlen = max(seq.prompt_len, max_seqlen)
        return max_seqlen

    def get_query_start_loc(self):
        cu_query_len_num = 0
        query_start_loc = [0]
        for seq in self.seqs:
            cu_query_len_num += seq.prompt_len - \
                seq.computed_page_num * self.memory_manager.page_size
            query_start_loc.append(cu_query_len_num)
        return torch.tensor(query_start_loc, device='cuda', dtype=torch.int32)

    def get_max_query_len(self):
        max_query_len = 0
        for seq in self.seqs:
            max_query_len = max(seq.prompt_len-seq.computed_page_num *
                                self.memory_manager.page_size, max_query_len)
        return max_query_len

    def get_block_table(self):
        max_num_block = torch.max(
            torch.tensor([len(seq.page_table) for seq in self.seqs]))
        block_tables = [seq.page_table for seq in self.seqs]
        block_tables = [block_table+[0]*(max_num_block-len(block_table))
                        for block_table in block_tables]

        return torch.tensor(block_tables, dtype=torch.int32, device='cuda')
