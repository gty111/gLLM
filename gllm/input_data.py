import torch
import numpy as np

from typing import List, Optional
from dataclasses import dataclass

from gllm.dist_utils import is_last_pp_rank
from gllm.utils import async_tensor_h2d, ceil_div, round_down
from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager


class InputData():
    def __init__(
        self, 
        seqs: List[Sequence], 
        memory_manager: MemoryManager, 
        use_mla: bool = False,
    ):
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
        self.tokens = self.get_tokens(seqs)
        self.positions = self.get_position(seqs)
        self.slot_mapping_tensor = self.get_slot_mapping(seqs)
        self.block_table = self.get_block_table(seqs)
        self.max_seq_len, self.seq_lens = self.get_seq_lens(seqs)
        self.max_query_len, self.query_start_loc = self.get_query_start_loc(seqs)

        if use_mla:
            # Construct MLA-related metadata
            num_actual_tokens = self.tokens.shape[0]
            slot_mapping = self.slot_mapping_tensor

            num_decodes = len(seqs)
            num_decode_tokens = num_decodes
            num_prefills = 0
            for idx,seq in enumerate(seqs):
                if not seq.computed_prompt:
                    num_decodes = idx 
                    num_decode_tokens = idx
                    num_prefills = len(seqs) - num_decodes
                    break
            
            query_seq_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
            num_computed_tokens = self.seq_lens - query_seq_lens
            
            # MLACommonDecodeMetadata
            decode_metatdata = None
            if num_decodes > 0:
                decode_seqs = seqs[:num_decode_tokens]
                _, seq_lens = self.get_seq_lens(decode_seqs)
                decode_metatdata = MLACommonDecodeMetadata(
                    block_table=self.block_table[:num_decode_tokens], 
                    seq_lens=seq_lens,
                )

            # MLACommonPrefillMetadata
            prefill_metadata = None
            if num_prefills > 0:
                prefill_seqs = seqs[num_decode_tokens:]
                max_query_len, query_start_loc = self.get_query_start_loc(prefill_seqs)

                # ChunkedContextMetadata
                chunked_context_metadata = None
                chunked_prefill_workspace_size = 128 * 1024
                context_lens = num_computed_tokens[num_decode_tokens:]
                max_context_len = context_lens.max().item()
                if max_context_len > 0:

                    workspace = torch.empty(
                        (chunked_prefill_workspace_size, 
                        memory_manager.kv_head_dim)
                    )

                    num_prefills_with_context = (context_lens > 0).sum().item()

                    max_context_chunk = (chunked_prefill_workspace_size //
                                         num_prefills_with_context)
                    max_context_chunk = round_down(max_context_chunk, self.page_size)

                    num_chunks = ceil_div(max_context_len, max_context_chunk)

                    chunk_starts = torch.arange(num_chunks, dtype=torch.int32) \
                                    .unsqueeze(1).expand(-1, num_prefills) \
                                    * max_context_chunk
                    chunk_ends = torch.min(context_lens.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                    chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                    cu_seq_lens = torch.zeros(
                        num_chunks,
                        num_prefills + 1,
                        dtype=torch.int32)
                    torch.cumsum(chunk_seq_lens,
                                 dim=1,
                                 out=cu_seq_lens[:,1:],
                                 dtype=torch.int32)
                    
                    chunked_context_metadata = MLACommonPrefillMetadata.ChunkedContextMetadata(
                        cu_seq_lens=cu_seq_lens,
                        starts=chunk_starts,
                        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        workspace=workspace,
                    )

                prefill_metadata = MLACommonPrefillMetadata(
                    block_table=self.block_table[num_decode_tokens:],
                    query_start_loc=query_start_loc,
                    max_query_len=max_query_len,
                    chunked_context=chunked_context_metadata
                )
            
            self.metadata = MLACommonMetadata(
                num_actual_tokens,
                slot_mapping,
                num_decodes,
                num_decode_tokens,
                num_prefills,
                decode_metatdata,
                prefill_metadata
            )
        
    def get_tokens(self, seqs):
        tokens_list = []
        for seq in seqs:
            tokens_list.extend(seq[seq.computed_token_num:seq.seq_len])
        return async_tensor_h2d(
            tokens_list, torch.long, 'cuda', True)

    def get_position(self, seqs):
        positions_list = []
        for seq in seqs:
            positions_list.extend(
                range(seq.computed_token_num, seq.seq_len))
        return async_tensor_h2d(
            positions_list, torch.long, 'cuda', True)

    def get_seq_lens(self, seqs):
        seq_lens = [seq.seq_len for seq in seqs]
        max_seqlen = max(seq_lens)
        return max_seqlen, async_tensor_h2d(seq_lens, torch.int32, 'cuda', True)

    def get_query_start_loc(self, seqs):
        query_lens = [0] + [seq.to_compute_token_num for seq in seqs]
        max_query_len = max(query_lens)
        query_start_loc = torch.from_numpy(np.cumsum(query_lens)).to(device='cuda', 
                                                                    dtype=torch.int32, 
                                                                    non_blocking=True)
        return max_query_len, query_start_loc

    def get_block_table(self, seqs):
        block_tables_list = [seq.page_table for seq in seqs]
        max_num_block = max(map(len, block_tables_list))
        block_tables = np.full(
            (len(block_tables_list), max_num_block), 0, dtype=np.int32)
        for idx, block_table in enumerate(block_tables_list):
            block_tables[idx, :len(block_table)] = block_table
        return torch.from_numpy(block_tables).to(device='cuda', non_blocking=True)

    def get_slot_mapping(self, seqs):
        slot_mapping = []
        for seq in seqs:
            for i in range(seq.computed_token_num,seq.seq_len):
                page_idx = i // self.page_size
                slot_idx = i % self.page_size
                slot_mapping.append(seq.page_table[page_idx]*self.page_size+slot_idx)

        return async_tensor_h2d(
            slot_mapping, torch.int64, 'cuda', True)

@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor

@dataclass
class MLACommonPrefillMetadata:
    """ Prefill Specific Metadata """

    @dataclass
    class ChunkedContextMetadata:
        # New for MLA (compared to FlashAttention)
        # For handling chunked prefill
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        workspace: torch.Tensor

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: Optional[ChunkedContextMetadata] = None

@dataclass
class MLACommonMetadata():
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    decode: Optional[MLACommonDecodeMetadata] = None
    prefill: Optional[MLACommonPrefillMetadata] = None