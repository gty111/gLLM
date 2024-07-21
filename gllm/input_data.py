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
            for seq in seqs:
                tokens_list.extend(seq.token_ids)
            self.tokens = torch.tensor(tokens_list, device='cuda')
            positions_list = []
            for seq in seqs:
                positions_list.extend(list(range(seq.prompt_len)))
            self.positions = torch.tensor(positions_list, device='cuda')
            self.cu_seqs_len = self.get_cu_seqs_len()
            self.max_seqlen = self.get_max_seqlen()
        else:
            self.tokens = torch.tensor(
                [seq.token_ids[-1] for seq in seqs], device='cuda')
            self.positions = torch.tensor(
                [len(seq.token_ids) for seq in seqs], device='cuda')
            self.cache_seqs_len = torch.tensor(
                [len(seq.token_ids) for seq in self.seqs], dtype=torch.int32, device='cuda')
            self.block_table = self.get_block_table()

        assert self.tokens.shape == self.positions.shape

    def get_cu_seqs_len(self):
        cu_seqs_len_num = 0
        cu_seqs_len_list = [0]
        for seq in self.seqs:
            cu_seqs_len_num += seq.prompt_len
            cu_seqs_len_list.append(cu_seqs_len_num)
        return torch.tensor(cu_seqs_len_list, device='cuda', dtype=torch.int32)

    def get_max_seqlen(self):
        max_seqlen = 0
        for seq in self.seqs:
            max_seqlen = max(seq.prompt_len, max_seqlen)
        return max_seqlen

    def get_block_table(self):
        max_num_block = torch.max(
            torch.tensor([len(seq.page_table) for seq in self.seqs]))
        block_tables = [seq.page_table for seq in self.seqs]
        block_tables = [block_table+[0]*(max_num_block-len(block_table))
                        for block_table in block_tables]

        return torch.tensor(block_tables, dtype=torch.int32, device='cuda')
