from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from gllm.memory_manager import MemoryManager
from gllm.sequence import Sequence
from gllm.utils import async_tensor_h2d, ceil_div, round_down


# Input of model forward
class InputData:
    def __init__(
        self,
        use_buffer: bool,
        memory_manager: MemoryManager,
        max_seq_length,
        max_running_seqs=None,
    ):

        self.page_size = memory_manager.page_size
        self.max_num_block = (max_seq_length + self.page_size - 1) // self.page_size
        self.use_mla = memory_manager.use_mla
        self.memory_manager: MemoryManager = memory_manager
        self.use_buffer = use_buffer

        if self.use_mla:
            self.chunked_prefill_workspace_size = 128 * 1024

        if use_buffer:
            assert max_running_seqs is not None and max_seq_length is not None
            self.tokens = torch.zeros(max_seq_length, dtype=torch.long)
            self.positions = torch.zeros(max_seq_length, dtype=torch.long)
            self.mrope_positions = torch.zeros((3, max_seq_length), dtype=torch.long)
            self.slot_mapping = torch.zeros(
                max_seq_length * max_running_seqs, dtype=torch.int64
            )
            self.block_table = torch.zeros(
                (max_running_seqs, self.max_num_block), dtype=torch.int32
            )
            self.seq_lens = torch.zeros(max_running_seqs, dtype=torch.int32)
            self.query_start_loc = torch.zeros(max_running_seqs + 1, dtype=torch.int32)

            if self.use_mla:
                self.workspace = torch.empty(
                    (self.chunked_prefill_workspace_size, memory_manager.kv_head_dim)
                )
                self.decode_seq_lens = torch.zeros(max_running_seqs, dtype=torch.int32)
                self.prefill_query_start_loc = torch.zeros(
                    max_running_seqs + 1, dtype=torch.int32
                )

    def prepare_sample(self):
        self.temperature = async_tensor_h2d(
            [seq.temperature if seq.temperature > 1e-5 else 1 for seq in self.seqs],
            self.memory_manager.dtype,
            "cuda",
            True,
        )
        self.top_p = async_tensor_h2d(
            [seq.top_p for seq in self.seqs], self.memory_manager.dtype, "cuda", True
        )
        self.top_k = async_tensor_h2d(
            [
                seq.top_k if seq.top_k != -1 else self.memory_manager.vocab_size
                for seq in self.seqs
            ],
            self.memory_manager.dtype,
            "cuda",
            True,
        )
        repetition_penalty = async_tensor_h2d(
            [seq.repetition_penalty for seq in self.seqs],
            self.memory_manager.dtype,
            "cuda",
            True,
        )
        self.repetition_penalty = repetition_penalty.unsqueeze(dim=1).repeat(
            1, self.memory_manager.vocab_size
        )

    def cal_input(self, seqs: List[Sequence]):
        assert len(seqs) != 0
        self.seqs = seqs
        self.embedding_size = 0

        self.tokens_cpu = self._cal_tokens(seqs)
        self.positions_cpu = self._cal_position(seqs)
        self.mrope_positions_cpu = None
        assert self.tokens_cpu.shape == self.positions_cpu.shape
        self.slot_mapping_cpu = self._cal_slot_mapping(seqs)
        self.block_table_cpu = self._cal_block_table(seqs)
        self.max_seq_len, self.seq_lens_cpu = self._cal_seq_lens(seqs)
        self.max_query_len, self.query_start_loc_cpu = self._cal_query_start_loc(seqs)

        if self.use_mla:
            self._cal_mla_metadata(seqs)

    def copy_to_input_buffer(self):
        assert self.use_buffer
        self.tokens[: self.tokens_cpu.shape[0]].copy_(
            self.tokens_cpu, non_blocking=True
        )
        if (
            hasattr(self, "mrope_positions_cpu")
            and self.mrope_positions_cpu is not None
        ):
            self.mrope_positions[:, : self.mrope_positions_cpu.shape[1]].copy_(
                self.mrope_positions_cpu, non_blocking=True
            )
        else:
            self.positions[: self.positions_cpu.shape[0]].copy_(
                self.positions_cpu, non_blocking=True
            )
        self.slot_mapping[: self.slot_mapping_cpu.shape[0]].copy_(
            self.slot_mapping_cpu, non_blocking=True
        )
        self.block_table[: self.block_table_cpu.shape[0]].copy_(
            self.block_table_cpu, non_blocking=True
        )
        self.seq_lens[: self.seq_lens_cpu.shape[0]].copy_(
            self.seq_lens_cpu, non_blocking=True
        )
        self.query_start_loc[: self.query_start_loc_cpu.shape[0]].copy_(
            self.query_start_loc_cpu, non_blocking=True
        )

        if self.use_mla:
            self._set_mla_metadata()

    def cal_and_set_input(self, seqs: List[Sequence]):
        self.cal_input(seqs)
        self.copy_to_input_buffer()

    def set_input_from_prebuilt(self, input_data):
        common_attrs_copy = [
            "seqs",
            "tokens_cpu",
            "positions_cpu",
            "mrope_positions_cpu",
            "slot_mapping_cpu",
            "block_table_cpu",
            "max_seq_len",
            "seq_lens_cpu",
            "max_query_len",
            "query_start_loc_cpu",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        ]
        mla_attrs_copy = [
            "num_actual_tokens",
            "num_decodes",
            "num_decode_tokens",
            "num_prefills",
            "max_context_len",
            "decode_seq_lens_cpu",
            "prefill_max_query_len",
            "prefill_query_start_loc_cpu",
            "chunk_starts_cpu",
            "chunk_seq_lens_cpu",
            "cu_seq_lens_cpu",
        ]
        for attr in common_attrs_copy:
            setattr(self, attr, getattr(input_data, attr, None))

        if self.use_mla:
            for attr in mla_attrs_copy:
                setattr(self, attr, getattr(input_data, attr, None))

        self.copy_to_input_buffer()

    def _cal_tokens(self, seqs: List[Sequence]):
        tokens_list = []
        for seq in seqs:
            tokens_list.extend(
                seq[seq.computed_token_num : seq.seq_len]
                if seq.to_compute_tokens is None
                else seq.to_compute_tokens
            )
        return torch.tensor(tokens_list, dtype=torch.long, device="cpu")

    def get_tokens(self):
        return self.tokens[: self.tokens_cpu.shape[0]]

    def _cal_position(self, seqs: List[Sequence]):
        positions_list = []
        for seq in seqs:
            positions_list.extend(range(seq.computed_token_num, seq.seq_len))
        return torch.tensor(positions_list, dtype=torch.long, device="cpu")

    def get_position(self):
        if self.mrope_positions_cpu is not None:
            return self.mrope_positions[:, : self.mrope_positions_cpu.shape[1]]
        else:
            return self.positions[: self.positions_cpu.shape[0]]

    def set_mrope_position(self, mrope_positions: torch.Tensor):
        self.mrope_positions_cpu = mrope_positions
        if self.use_buffer:
            self.mrope_positions[:, : self.mrope_positions_cpu.shape[1]].copy_(
                self.mrope_positions_cpu, non_blocking=True
            )

    def _cal_seq_lens(self, seqs: List[Sequence]):
        seq_lens = [seq.seq_len for seq in seqs]
        return max(seq_lens), torch.tensor(
            seq_lens, dtype=torch.int32, device="cpu", pin_memory=True
        )

    def get_seq_lens(self):
        return self.seq_lens[: self.seq_lens_cpu.shape[0]]

    def _cal_query_start_loc(self, seqs: List[Sequence]):
        query_lens = [0] + [seq.to_compute_token_num for seq in seqs]
        return max(query_lens), torch.from_numpy(np.cumsum(query_lens)).to(
            device="cpu", dtype=torch.int32
        )

    def get_query_start_loc(self):
        return self.query_start_loc[: self.query_start_loc_cpu.shape[0]]

    def _cal_block_table(self, seqs: List[Sequence]):
        block_tables_list = [seq.page_table for seq in seqs]
        block_tables = np.full(
            (len(block_tables_list), self.max_num_block), 0, dtype=np.int32
        )
        for idx, block_table in enumerate(block_tables_list):
            block_tables[idx, : len(block_table)] = block_table
        return torch.from_numpy(block_tables)

    def get_block_table(self):
        return self.block_table[: self.block_table_cpu.shape[0]]

    def _cal_slot_mapping(self, seqs: List[Sequence]):
        slot_mapping = []
        for seq in seqs:
            for i in range(seq.computed_token_num, seq.seq_len):
                page_idx = i // self.page_size
                slot_idx = i % self.page_size
                slot_mapping.append(
                    seq.page_table[page_idx] * self.page_size + slot_idx
                )
        return torch.tensor(slot_mapping, dtype=torch.int64, device="cpu")

    def get_slot_mapping(self):
        return self.slot_mapping[: self.slot_mapping_cpu.shape[0]]

    def _cal_mla_metadata(self, seqs: List[Sequence]):
        # Construct MLA-related metadata
        self.num_actual_tokens = self.tokens_cpu.shape[0]

        self.num_decodes = len(seqs)
        self.num_decode_tokens = self.num_decodes
        self.num_prefills = 0
        for idx, seq in enumerate(seqs):
            if not seq.computed_prompt:
                self.num_decodes = idx
                self.num_decode_tokens = idx
                self.num_prefills = len(seqs) - self.num_decodes
                break

        query_seq_lens = self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
        num_computed_tokens = self.seq_lens_cpu - query_seq_lens

        if self.num_decodes > 0:
            decode_seqs = seqs[: self.num_decode_tokens]
            _, self.decode_seq_lens_cpu = self._cal_seq_lens(decode_seqs)

        if self.num_prefills > 0:
            prefill_seqs = seqs[self.num_decode_tokens :]
            self.prefill_max_query_len, self.prefill_query_start_loc_cpu = (
                self._cal_query_start_loc(prefill_seqs)
            )

            context_lens = num_computed_tokens[self.num_decode_tokens :]
            self.max_context_len = max(context_lens)
            if self.max_context_len > 0:
                num_prefills_with_context = (context_lens > 0).sum().item()

                max_context_chunk = (
                    self.chunked_prefill_workspace_size // num_prefills_with_context
                )
                max_context_chunk = round_down(max_context_chunk, self.page_size)

                num_chunks = ceil_div(self.max_context_len, max_context_chunk)

                self.chunk_starts_cpu = (
                    torch.arange(
                        num_chunks, dtype=torch.int32, device="cpu", pin_memory=True
                    )
                    .unsqueeze(1)
                    .expand(-1, self.num_prefills)
                    * max_context_chunk
                )
                chunk_ends = torch.min(
                    context_lens.unsqueeze(0), self.chunk_starts_cpu + max_context_chunk
                )
                self.chunk_seq_lens_cpu = (chunk_ends - self.chunk_starts_cpu).clamp(
                    min=0
                )

                self.cu_seq_lens_cpu = torch.zeros(
                    num_chunks,
                    self.num_prefills + 1,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                torch.cumsum(
                    self.chunk_seq_lens_cpu,
                    dim=1,
                    out=self.cu_seq_lens_cpu[:, 1:],
                    dtype=torch.int32,
                )

    def _set_mla_metadata(self):
        if self.num_prefills > 0:
            self.prefill_query_start_loc[
                : self.prefill_query_start_loc_cpu.shape[0]
            ].copy_(self.prefill_query_start_loc_cpu, non_blocking=True)
        if self.num_decodes > 0:
            self.decode_seq_lens[: self.decode_seq_lens_cpu.shape[0]].copy_(
                self.decode_seq_lens_cpu, non_blocking=True
            )

        decode_metadata = (
            MLACommonDecodeMetadata(
                block_table=self.get_block_table()[: self.num_decode_tokens],
                seq_lens=self.decode_seq_lens[: self.decode_seq_lens_cpu.shape[0]],
            )
            if self.num_decodes > 0
            else None
        )

        chunked_context_metadata = (
            MLACommonPrefillMetadata.ChunkedContextMetadata(
                cu_seq_lens=self.cu_seq_lens_cpu.to("cuda", non_blocking=True),
                starts=self.chunk_starts_cpu.to("cuda", non_blocking=True),
                seq_tot=self.chunk_seq_lens_cpu.sum(dim=1).tolist(),
                max_seq_lens=self.chunk_seq_lens_cpu.max(dim=1).values.tolist(),
                workspace=self.workspace,
            )
            if self.num_prefills > 0 and self.max_context_len > 0
            else None
        )

        prefill_metadata = (
            MLACommonPrefillMetadata(
                block_table=self.get_block_table()[self.num_decode_tokens :],
                query_start_loc=self.prefill_query_start_loc[
                    : self.prefill_query_start_loc_cpu.shape[0]
                ],
                max_query_len=self.prefill_max_query_len,
                chunked_context=chunked_context_metadata,
            )
            if self.num_prefills > 0
            else None
        )
        self.metadata = MLACommonMetadata(
            self.num_actual_tokens,
            self.get_slot_mapping(),
            self.num_decodes,
            self.num_decode_tokens,
            self.num_prefills,
            decode_metadata,
            prefill_metadata,
        )


@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class MLACommonPrefillMetadata:
    """Prefill Specific Metadata"""

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
class MLACommonMetadata:
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
