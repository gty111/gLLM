from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from gllm.layers.repetition_penalty import build_repetition_penalty_mask
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
        # Hybrid models (Qwen3.5 GDN / Mamba) expose a per-request SSM state
        # slot in addition to their KV pages. ``use_ssm_cache`` mirrors how
        # ``use_mla`` gates the alternate KV layout above.
        self.use_ssm_cache = memory_manager.use_ssm_cache
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

            if self.use_ssm_cache:
                # Per-seq SSM working slot id, indexed by the row in the
                # batch. The GDN kernels read this as their ``cache_indices``
                # argument. Slot 0 is the CUDA-graph dummy slot, so padded
                # decode rows go there harmlessly.
                self.ssm_state_slot_per_seq = torch.zeros(
                    max_running_seqs, dtype=torch.int32
                )
                # 1 for sequences whose prefix-state already lives in
                # ``ssm_state[ssm_state_slot_per_seq[i]]`` (either chunked-
                # prefill continuation or prefix-cache hit). Passed to the
                # conv1d / chunk-GDN kernels as ``has_initial_state``.
                self.has_initial_state_per_seq = torch.zeros(
                    max_running_seqs, dtype=torch.bool
                )
                # Snapshot-pool slot id the GDN layer should write into AT
                # END OF THIS FORWARD for each seq, or -1 to skip. Populated
                # by ``_cal_ssm_metadata`` from ``PrefixSegment.
                # page2ssm_snapshot`` whenever the seq's prefill ends
                # exactly on a page boundary and that page owns a snapshot
                # slot. Decode steps never snapshot — they would only
                # produce duplicate states with the working slot.
                self.ssm_snapshot_write_slot_per_seq = torch.full(
                    (max_running_seqs,), -1, dtype=torch.int32
                )

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
            torch.int32,
            "cuda",
            True,
        )
        # Build the per-seq ``[batch, vocab]`` repetition-penalty mask via
        # the shared helper in ``gllm/layers/repetition_penalty.py`` (see
        # that module's header for the sglang reference + the rationale
        # behind our HF-style "seed with prompt + generated" semantics).
        self.repetition_penalty = build_repetition_penalty_mask(
            self.seqs,
            batch_size=len(self.seqs),
            vocab_size=self.memory_manager.vocab_size,
            dtype=self.memory_manager.dtype,
        )
        self.needs_repetition_penalty = self.repetition_penalty is not None

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

        if self.use_ssm_cache:
            self._cal_ssm_metadata(seqs)

        if self.use_mla:
            self._cal_mla_metadata(seqs)

    def _cal_ssm_metadata(self, seqs: List[Sequence]):
        """Build per-seq SSM slot id + initial-state flag + snapshot target.

        * ``ssm_state_slot_per_seq[i]`` = ``seqs[i].ssm_state_slot`` (or 0 = the
          dummy slot if the seq somehow has no slot yet, which should never
          happen because the scheduler allocates one before this method is
          called).
        * ``has_initial_state_per_seq[i]`` = True when ``computed_token_num >
          0``. This covers both chunked-prefill continuation and prefix-cache
          hits where ``PrefixMemoryManager`` already copied the snapshot back
          into the working slot.
        * ``ssm_snapshot_write_slot_per_seq[i]`` = snapshot-pool slot id to
          which the GDN layer should copy this seq's working state at the
          end of this forward, or -1 if no snapshot is wanted. Populated
          only for prefill rows whose chunk ends exactly on a page boundary
          and whose ``PrefixSegment`` reserved a snapshot slot for that
          page (i.e. enable_prefix_caching=True + the page is cacheable).
        """
        bs = len(seqs)
        slots = np.empty(bs, dtype=np.int32)
        has_init = np.empty(bs, dtype=np.bool_)
        snap_targets = np.full(bs, -1, dtype=np.int32)

        # Pull the snapshot pointer table once; ``PrefixSegment`` is the
        # only Segment subclass that owns ``page2ssm_snapshot``. The
        # ``segment`` attribute is created lazily by
        # :meth:`MemoryManager.init`, so during the pre-init profile run
        # we may not have it yet — fall back to ``None`` (no snapshots).
        segment = getattr(self.memory_manager, "segment", None)
        page2snap = getattr(segment, "page2ssm_snapshot", None) if segment is not None else None
        page_size = self.memory_manager.page_size

        for i, seq in enumerate(seqs):
            slots[i] = seq.ssm_state_slot if seq.ssm_state_slot is not None else 0
            has_init[i] = seq.computed_token_num > 0

            # Snapshot timing: prefill chunk landing on a page boundary.
            # Decode rows skip this branch since ``computed_prompt`` flips
            # True before any decode token is emitted (see Sequence) and
            # ``page2snap is None`` for non-prefix-cache runs.
            if page2snap is None or seq.computed_prompt:
                continue
            end_tokens = seq.computed_token_num + seq.to_compute_token_num
            if end_tokens == 0 or end_tokens % page_size != 0:
                continue
            page_idx = (end_tokens // page_size) - 1
            if page_idx < 0 or page_idx >= len(seq.page_table):
                continue
            snap_slot = page2snap[seq.page_table[page_idx]]
            if snap_slot is not None:
                snap_targets[i] = snap_slot

        self.ssm_state_slot_per_seq_cpu = torch.from_numpy(slots).pin_memory()
        self.has_initial_state_per_seq_cpu = torch.from_numpy(has_init).pin_memory()
        self.ssm_snapshot_write_slot_per_seq_cpu = (
            torch.from_numpy(snap_targets).pin_memory()
        )

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
        # H2D only the columns we actually filled in ``_cal_block_table``
        # (see its docstring for the bandwidth motivation). The persistent
        # ``self.block_table`` device buffer keeps the full width so the
        # captured-graph kernel signature is stable; only the bytes that are
        # read by FlashAttention (the first ``max_blocks_used`` cols of each
        # row, where ``max_blocks_used >= ceil(cache_seqlens[i]/page_size)``
        # for every row in the current batch) get overwritten this step.
        bs_, used_cols = self.block_table_cpu.shape
        self.block_table[:bs_, :used_cols].copy_(
            self.block_table_cpu, non_blocking=True
        )
        self.seq_lens[: self.seq_lens_cpu.shape[0]].copy_(
            self.seq_lens_cpu, non_blocking=True
        )
        self.query_start_loc[: self.query_start_loc_cpu.shape[0]].copy_(
            self.query_start_loc_cpu, non_blocking=True
        )

        if self.use_ssm_cache:
            n_seqs = self.ssm_state_slot_per_seq_cpu.shape[0]
            self.ssm_state_slot_per_seq[:n_seqs].copy_(
                self.ssm_state_slot_per_seq_cpu, non_blocking=True
            )
            self.has_initial_state_per_seq[:n_seqs].copy_(
                self.has_initial_state_per_seq_cpu, non_blocking=True
            )
            self.ssm_snapshot_write_slot_per_seq[:n_seqs].copy_(
                self.ssm_snapshot_write_slot_per_seq_cpu, non_blocking=True
            )

        if self.use_mla:
            self._set_mla_metadata()

    def cal_and_set_input(self, seqs: List[Sequence]):
        self.cal_input(seqs)
        self.copy_to_input_buffer()

    # Attributes copied verbatim from a prebuilt InputData.
    _PREBUILT_COMMON_ATTRS = (
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
        "needs_repetition_penalty",
    )
    _PREBUILT_SSM_ATTRS = (
        "ssm_state_slot_per_seq_cpu",
        "has_initial_state_per_seq_cpu",
        "ssm_snapshot_write_slot_per_seq_cpu",
    )
    _PREBUILT_MLA_ATTRS = (
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
    )

    def set_input_from_prebuilt_cpu(self, input_data):
        """CPU-only portion of ``set_input_from_prebuilt``.

        Copies the prebuilt CPU attributes onto ``self`` without touching the
        shared GPU input buffers. Callers that want to overlap CPU prep with
        an in-flight GPU forward should pair this with a later
        :meth:`copy_to_input_buffer` call once the previous forward has
        released the input buffers (see ``OverlapModelRunner``).
        """
        for attr in self._PREBUILT_COMMON_ATTRS:
            setattr(self, attr, getattr(input_data, attr, None))

        if self.use_ssm_cache:
            for attr in self._PREBUILT_SSM_ATTRS:
                setattr(self, attr, getattr(input_data, attr, None))

        if self.use_mla:
            for attr in self._PREBUILT_MLA_ATTRS:
                setattr(self, attr, getattr(input_data, attr, None))

    def set_input_from_prebuilt(self, input_data):
        self.set_input_from_prebuilt_cpu(input_data)
        self.copy_to_input_buffer()

    def _cal_tokens(self, seqs: List[Sequence]):
        tokens_list = []
        for seq in seqs:
            tokens_list.extend(
                seq[seq.computed_token_num : seq.seq_len]
                if seq.to_compute_tokens is None
                else seq.to_compute_tokens
            )
        # Pin the CPU staging buffer so the H2D into ``self.tokens`` issued
        # from ``copy_to_input_buffer`` with ``non_blocking=True`` is truly
        # async on the prep stream. Pageable sources cause CUDA to fall back
        # to a synchronous staging copy, which would defeat the host/GPU
        # overlap the rest of the pipeline relies on.
        return torch.tensor(
            tokens_list, dtype=torch.long, device="cpu", pin_memory=True
        )

    def get_tokens(self):
        return self.tokens[: self.tokens_cpu.shape[0]]

    def _cal_position(self, seqs: List[Sequence]):
        # Position ids are just consecutive integers per seq, so we write
        # them straight into a pinned tensor via its numpy view instead of
        # building a Python list and going through ``torch.tensor(list,
        # pin_memory=True)``. The old path was fine for tiny decode batches
        # but cost ~200 us per prefill batch (1024 tokens -> 1024 Python int
        # boxings + list growth + tensor conversion); microbench shows the
        # vectorized form is ~22x faster on a single 1024-token prefill and
        # within noise for pure decode.
        total = sum(seq.seq_len - seq.computed_token_num for seq in seqs)
        out = torch.empty(
            total, dtype=torch.long, device="cpu", pin_memory=True
        )
        out_np = out.numpy()
        offset = 0
        for seq in seqs:
            start = seq.computed_token_num
            n = seq.seq_len - start
            if n == 1:
                out_np[offset] = start
            elif n > 1:
                out_np[offset : offset + n] = np.arange(
                    start, start + n, dtype=np.int64
                )
            offset += n
        return out

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
        # Materialize directly into a pinned tensor so downstream non_blocking
        # H2D doesn't have to fall back to a synchronous staging copy. We
        # write the cumulative sum straight into the pinned buffer's numpy
        # view to avoid the prior intermediate ``np.cumsum`` allocation +
        # ``torch.from_numpy`` + ``copy_`` round-trip. ``device="cpu"`` is
        # required because ``ModelLoader`` sets the default torch device to
        # CUDA, and ``pin_memory=True`` is only legal on dense CPU tensors.
        out = torch.empty(
            len(query_lens), dtype=torch.int32, device="cpu", pin_memory=True
        )
        np.cumsum(query_lens, dtype=np.int32, out=out.numpy())
        return max(query_lens), out

    def get_query_start_loc(self):
        view = self.query_start_loc[: self.query_start_loc_cpu.shape[0]]
        # Attach a pinned-CPU mirror so downstream kernels (e.g. fla's
        # ``prepare_chunk_indices``) can do the integer index math on the host
        # without a ``cudaStreamSynchronize``. The mirror always lags the GPU
        # tensor by 0 steps because they were filled in the same prepare-input
        # phase from the same source values.
        view._cpu_view = self.query_start_loc_cpu
        return view

    def get_ssm_state_slot_per_seq(self):
        """Per-seq SSM working-slot ids (int32). Use as ``cache_indices``
        for the conv1d/GDN kernels. Raises ``AttributeError`` if the model
        does not use the SSM cache (programmer error: a layer should not
        call this on a non-hybrid model)."""
        return self.ssm_state_slot_per_seq[: self.ssm_state_slot_per_seq_cpu.shape[0]]

    def get_has_initial_state_per_seq(self):
        return self.has_initial_state_per_seq[: self.has_initial_state_per_seq_cpu.shape[0]]

    def get_ssm_snapshot_write_slot_per_seq(self):
        """Per-seq snapshot-pool slot id to write at end of forward (-1=skip).

        Returns ``None`` when SSM cache is disabled or when there are no
        rows to write (no enable_prefix_caching, no hybrid model). Layers
        treat ``None`` as "no snapshotting to do".
        """
        if not self.use_ssm_cache:
            return None
        if not hasattr(self, "ssm_snapshot_write_slot_per_seq_cpu"):
            return None
        n = self.ssm_snapshot_write_slot_per_seq_cpu.shape[0]
        return self.ssm_snapshot_write_slot_per_seq[:n]

    def _cal_block_table(self, seqs: List[Sequence]):
        block_tables_list = [seq.page_table for seq in seqs]
        bs = len(block_tables_list)
        # Previously we (1) allocated a temporary ``np.full((bs, max_num_block),
        # 0)`` (~1 MB on Qwen3-0.6B with model_max_length=131072 / page_size=16
        # -> max_num_block=8192) and zero-filled it, (2) sparsely filled the
        # ragged page-table rows into it, then (3) allocated a same-shape
        # pinned tensor and copied the numpy buffer into it (a 1 MB host-to-
        # pinned memcpy). Profiler showed that final ``out.copy_(...)`` taking
        # 4-9 ms per batch -- it dominated cal_input. The host-side ``copy_``
        # itself is normally <30us for 1 MB; the inflated wall-clock comes
        # from (a) writing 1 MB twice (zero the numpy buffer, then memcpy
        # it into the pinned buffer) which is bandwidth-bound and contends
        # with the prior batch's still-in-flight H2D issued from the same
        # caching-host-allocator pool, and (b) cold-page touches on freshly
        # handed-out pinned slabs.
        #
        # Skip the intermediate numpy buffer: get a numpy view onto the pinned
        # tensor directly, zero only that view, then sparsely fill. This drops
        # one 1 MB CPU write and the from_numpy bookkeeping. Microbench shows
        # ~2.9x speedup, real workload sees _cal_block_table mean drop from
        # ~3.2ms to <1ms. See ``_cal_query_start_loc`` for why
        # ``device="cpu"`` is required (default device is CUDA under
        # ``ModelLoader``).
        #
        # Width: only allocate / fill / H2D ``max_blocks_used`` columns
        # instead of the full ``self.max_num_block`` (= ceil(model_max_length /
        # page_size); e.g. 16384 for Qwen3-30B-A3B's 256K context). At bs=64
        # that's 64 * 16384 * 4 = 4 MiB of int32 per forward, of which only
        # the first ``ceil(max(seq_len)/page_size)`` columns are non-zero (the
        # rest is dead padding). Torch-profiler tracing on Qwen3-30B-A3B
        # TP=4 with conc=32 showed this single copy accounting for ~80 ms of
        # ``Memcpy HtoD (Pinned -> Device)`` per 64-prompt run; SGLang at the
        # same config does ~0 such copies. FlashAttention only reads up to
        # ``cache_seqlens[i] / page_size`` columns per row in the persistent
        # device-side ``block_table`` buffer, so leaving stale data beyond
        # ``max_blocks_used`` is safe. ``copy_to_input_buffer`` H2Ds only
        # ``[:bs, :max_blocks_used]`` to match; the kernel-facing
        # ``get_block_table`` view stays wide so the captured CUDA graph
        # signature is unaffected.
        max_blocks_used = max((len(t) for t in block_tables_list), default=1)
        if max_blocks_used == 0:
            max_blocks_used = 1
        out = torch.empty(
            (bs, max_blocks_used),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        out_np = out.numpy()
        out_np.fill(0)
        for idx, block_table in enumerate(block_tables_list):
            out_np[idx, : len(block_table)] = block_table
        return out

    def get_block_table(self):
        return self.block_table[: self.block_table_cpu.shape[0]]

    def _cal_slot_mapping(self, seqs: List[Sequence]):
        # Same motivation as ``_cal_position``: write straight into a pinned
        # tensor's numpy view. The original double-Python-loop
        # ("for seq -> for i in range(...)") plus ``slot_mapping.append(...)``
        # was the largest remaining ``cal_input`` sub-op after the
        # ``_cal_block_table`` fix -- profiler showed ~226 us mean per batch
        # because every prefill iter does 1024 Python int boxings (and the
        # ``seq.page_table[i // page_size]`` lookup walks a Python list each
        # time). For prefill seqs we vectorize via numpy: precompute the
        # token index range, derive ``(page_idx, slot_idx)`` with integer
        # ops, then ``page_table_np[page_idx] * page_size + slot_idx`` in a
        # single numpy expression. Decode (n == 1) keeps a fast scalar path
        # because the numpy overhead would dominate one-element batches.
        # Microbench: ~8x faster on a 4x1024 prefill batch, ~1.3x on a 32x1
        # decode batch.
        page_size = self.page_size
        total = sum(seq.seq_len - seq.computed_token_num for seq in seqs)
        out = torch.empty(
            total, dtype=torch.int64, device="cpu", pin_memory=True
        )
        out_np = out.numpy()
        offset = 0
        for seq in seqs:
            start = seq.computed_token_num
            n = seq.seq_len - start
            if n == 1:
                out_np[offset] = (
                    seq.page_table[start // page_size] * page_size
                    + (start % page_size)
                )
            elif n > 1:
                token_indices = np.arange(start, start + n, dtype=np.int64)
                page_idx = token_indices // page_size
                slot_idx = token_indices - page_idx * page_size
                page_table_np = np.asarray(seq.page_table, dtype=np.int64)
                out_np[offset : offset + n] = (
                    page_table_np[page_idx] * page_size + slot_idx
                )
            offset += n
        return out

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

    def pad_for_cuda_graph(self, padded_size: int):
        """Pad input buffers to padded_size using dummy values.

        This enables CUDA graph replay for a fixed batch size (a power-of-two
        bucket) even when the actual number of decode tokens is smaller.

        The dummy tokens write their KV entries to memory_manager.dummy_page,
        which is permanently reserved and never used by real sequences.

        Returns:
            num_real_tokens (int): the actual (unpadded) token count, so that
            the caller can slice output_hidden_states[:num_real_tokens] when
            computing logits after graph replay.
        """
        assert self.use_buffer, "pad_for_cuda_graph requires use_buffer=True"
        num_real_tokens = self.tokens_cpu.shape[0]
        if num_real_tokens >= padded_size:
            return num_real_tokens

        dummy_page = self.memory_manager.dummy_page
        dummy_slot = dummy_page * self.page_size  # slot index within dummy page

        num_pad = padded_size - num_real_tokens

        # tokens: pad with 0
        self.tokens[num_real_tokens:padded_size].zero_()
        # positions: pad with 0
        self.positions[num_real_tokens:padded_size].zero_()
        # mrope_positions: pad with 0
        self.mrope_positions[:, num_real_tokens:padded_size].zero_()
        # slot_mapping: pad with dummy slot so writes go to the reserved page
        self.slot_mapping[num_real_tokens:padded_size].fill_(dummy_slot)
        # seq_lens: pad with 1 (avoid division-by-zero in attention kernels)
        self.seq_lens[len(self.seqs):len(self.seqs) + num_pad].fill_(1)
        # block_table: pad rows with dummy_page
        self.block_table[len(self.seqs):len(self.seqs) + num_pad].fill_(dummy_page)
        # query_start_loc: continue the cumulative sum — each dummy token counts
        # as 1 query token, so the padded entries are last_loc+1, last_loc+2, ...
        last_loc = self.query_start_loc[len(self.seqs)]
        self.query_start_loc[len(self.seqs) + 1:len(self.seqs) + num_pad + 1].copy_(
            last_loc + torch.arange(1, num_pad + 1, dtype=torch.int32)
        )

        if self.use_mla:
            # Pad decode_seq_lens for the dummy sequences so that MLA decode
            # kernels see a valid (non-zero) sequence length for every row.
            self.decode_seq_lens[len(self.seqs):len(self.seqs) + num_pad].fill_(1)

        if self.use_ssm_cache:
            # Padded rows write into the SSM dummy slot (slot 0) and report
            # ``has_initial_state=False`` so the GDN kernels treat them as a
            # fresh prefill that quietly writes scratch into a slot nobody
            # else reads.
            self.ssm_state_slot_per_seq[len(self.seqs):len(self.seqs) + num_pad].fill_(0)
            self.has_initial_state_per_seq[len(self.seqs):len(self.seqs) + num_pad].fill_(False)
            # Padded rows must never trigger a snapshot copy: -1 = skip.
            self.ssm_snapshot_write_slot_per_seq[
                len(self.seqs) : len(self.seqs) + num_pad
            ].fill_(-1)

        return num_real_tokens

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
