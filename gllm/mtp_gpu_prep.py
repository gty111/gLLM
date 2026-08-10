"""GPU-native input preparation for the MTP draft / verify forwards.

Motivation (measured on Qwen3.5-0.8B, 1×H200, 64 concurrent greedy decodes with
host-side phase profiling): one fused MTP step cost ~12.4 ms, of which ~4.3 ms was
**host** time spent rebuilding input arrays in Python -- ``cal_input`` for the
draft batch (0.5 ms) and for the verify batch (1.5 ms), the per-seq GPU
context-length bookkeeping (2.1 ms) and the dummy pad-``Sequence`` objects the
padded buckets needed. All of that is pure overhead: the arrays are simple
affine functions of *per-sequence* facts the engine already knows, and the
verify batch's token ids already live on the GPU (they are the draft chain's
output). Only the sequences' page tables genuinely have to cross the PCIe bus.

This module removes that host work by keeping a persistent batch representation:

* keeps the per-sequence facts in **persistent pinned staging buffers** (no
  per-step allocation, no ``pin_memory()`` call, no ``torch.tensor(list)``),
* ships them to the GPU in **three small H2D copies** (metadata, page tables,
  SSM block tables),
* derives every **per-token** array (positions, slot_mapping, seq_lens,
  query_start_loc, token ids, mrope positions) with vectorized CUDA ops written
  **directly into the static input buffers** that the captured draft / verify
  CUDA graphs read, and
* pads a bucket by writing dummy rows on the GPU instead of building throwaway
  ``Sequence`` objects on the host.

The formulas mirror ``InputData._cal_*`` exactly for the shapes MTP uses (one
token per seq for a draft step, a uniform ``1+k`` query over a cached context
for a verify step).
"""

from typing import List, Optional

import numpy as np
import torch

from gllm.sequence import Sequence

# Columns of the per-sequence metadata staging buffer. One row per batch row;
# a single [bucket, META_W] int64 H2D per prepared batch.
META_CTX = 0          # cached context length (== position of the first new token)
META_DELTA = 1        # mrope position delta (0 when the model has no mrope)
META_X1 = 2           # the seq's x1 token id (verify column 0 / draft input)
META_NUM_ACCEPTED = 3  # SSM resume column selector (``num_accepted``)
META_W = 4


class MtpGpuPrep:
    """Persistent staging buffers + GPU builders for MTP input preparation.

    One instance per :class:`ModelRunner`. ``max_bs`` is the largest captured
    graph bucket (batch rows), ``max_blocks`` the page-table width of the
    engine's static ``block_table`` buffer.
    """

    def __init__(
        self,
        *,
        max_bs: int,
        max_blocks: int,
        bt_width: int,
        page_size: int,
        uses_mrope: bool,
        device: torch.device,
    ):
        self.max_bs = max_bs
        self.max_blocks = max_blocks
        self.bt_width = bt_width
        self.page_size = page_size
        self.uses_mrope = uses_mrope
        self.device = device

        # --- persistent host staging (pinned; filled through numpy views) ---
        self._h_meta = torch.zeros((max_bs, META_W), dtype=torch.int64, device="cpu", pin_memory=True)
        self._h_meta_np = self._h_meta.numpy()
        self._h_pt = torch.zeros((max_bs, max_blocks), dtype=torch.int32, device="cpu", pin_memory=True)
        self._h_pt_np = self._h_pt.numpy()
        self._h_bt = torch.zeros((max_bs, max(bt_width, 1)), dtype=torch.int32, device="cpu", pin_memory=True)
        self._h_bt_np = self._h_bt.numpy()

        # --- persistent device mirrors ---
        self._d_meta = torch.zeros((max_bs, META_W), dtype=torch.int64, device=device)
        self._d_pt = torch.zeros((max_bs, max_blocks), dtype=torch.int32, device=device)
        self._d_bt = torch.zeros(
            (max_bs, max(bt_width, 1)), dtype=torch.int32, device=device
        )

        # Cached ``arange`` helpers (avoid re-allocating them every step).
        self._ar_cache = {}
        # Number of page-table columns pushed by the last ``push_meta`` (the
        # GPU builders only touch that prefix -- everything past it is stale
        # padding that no kernel reads, exactly like ``_cal_block_table``).
        self.pt_cols = 1
        # Batch geometry of the last ``push_meta`` + the (epoch, bucket) it was
        # staged for, so the draft and verify phases of one MTP step share a
        # single staging pass (their per-seq facts are identical: the page
        # tables are pre-allocated for the whole speculative window up front).
        self.nd = 0
        self.bucket = 0
        self._staged = (-1, -1)

    # ------------------------------------------------------------------
    # host -> device: per-sequence facts
    # ------------------------------------------------------------------
    def _arange(self, n: int, dtype=torch.int64) -> torch.Tensor:
        key = (n, dtype)
        ar = self._ar_cache.get(key)
        if ar is None:
            ar = torch.arange(n, dtype=dtype, device=self.device)
            self._ar_cache[key] = ar
        return ar

    def push_meta(
        self,
        seqs: List[Sequence],
        bucket: int,
        *,
        epoch: int,
        ctx_lens: List[int],
        x1: List[int],
        dummy_page: int,
        mrope_deltas: Optional[List[int]] = None,
        ctx_lens_gpu: Optional[torch.Tensor] = None,
        x1_gpu: Optional[torch.Tensor] = None,
        num_accepted_gpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Stage + H2D this step's per-sequence facts for rows ``[0, bucket)``.

        ``ctx_lens[i]`` is the number of already-cached tokens of ``seqs[i]``
        (the position its first new token lands on) and ``x1[i]`` its first new
        token id. Rows ``[len(seqs), bucket)`` are the CUDA-graph padding: they
        get context length 0 and a page table full of ``dummy_page``, so every
        derived index lands in the reserved dummy page.

        ``epoch`` identifies the MTP step; re-staging the same (epoch, bucket) is
        a no-op so the draft and verify phases only pay for it once.
        """
        nd = len(seqs)
        assert bucket <= self.max_bs, (bucket, self.max_bs)
        if self._staged == (epoch, bucket):
            return
        self._staged = (epoch, bucket)
        self.nd = nd
        self.bucket = bucket

        meta = self._h_meta_np
        meta[:nd, META_CTX] = ctx_lens
        meta[:nd, META_X1] = x1
        if mrope_deltas is not None:
            meta[:nd, META_DELTA] = mrope_deltas
        else:
            meta[:nd, META_DELTA] = 0
        meta[:nd, META_NUM_ACCEPTED] = [
            getattr(s, "ssm_num_accepted", 1) or 1 for s in seqs
        ]
        if bucket > nd:
            # Padding rows mirror the throwaway dummy ``Sequence`` objects the
            # CPU path used (``_create_dummy_verify_seqs`` / ``create_dummy_seqs``):
            # context length 1, token id 1, SSM block 0, page table all
            # ``dummy_page`` -- value-identical to what the CPU path produced.
            meta[nd:bucket, :] = 0
            meta[nd:bucket, META_CTX] = 1
            meta[nd:bucket, META_X1] = 1
            meta[nd:bucket, META_NUM_ACCEPTED] = 1

        # Page tables: ragged Python lists, so this is the one place a per-row
        # loop is unavoidable. It writes into the persistent pinned buffer, so
        # unlike ``_cal_block_table`` there is no allocation and no host->pinned
        # bounce; only ``pt_cols`` columns are shipped. The used window is zeroed
        # first so a short row's tail reads 0 rather than a previous step's page
        # id -- the attention kernels never look past ``ceil(seq_len/P)`` columns,
        # but matching ``_cal_block_table`` byte for byte keeps the assert honest
        # (and a stale page id in a debug dump is deeply confusing).
        pt = self._h_pt_np
        lens = [len(s.page_table) for s in seqs]
        cols = max(max(lens, default=1), 1)
        pt[:bucket, :cols] = 0
        for i, seq in enumerate(seqs):
            if lens[i]:
                pt[i, : lens[i]] = seq.page_table
        if bucket > nd:
            # Dummy rows: one page (the reserved dummy page), like the throwaway
            # pad ``Sequence`` objects had.
            pt[nd:bucket, 0] = dummy_page
        self.pt_cols = cols

        # SSM state block tables (``1+k`` per seq; column 0 == committed state).
        if self.bt_width:
            bt = self._h_bt_np
            w = self.bt_width
            for i, seq in enumerate(seqs):
                row = seq.ssm_block_table
                if row is not None:
                    bt[i, :w] = row
                else:
                    bt[i, :w] = 0
                    bt[i, 0] = seq.ssm_state_slot or 0
            if bucket > nd:
                bt[nd:bucket, :w] = 0

        self._d_meta[:bucket].copy_(self._h_meta[:bucket], non_blocking=True)
        self._d_pt[:bucket, :cols].copy_(self._h_pt[:bucket, :cols], non_blocking=True)
        if self.bt_width:
            self._d_bt[:bucket].copy_(self._h_bt[:bucket], non_blocking=True)

        # A chained overlap-MTP step is launched before the predecessor's CPU
        # completion is finalized.  In that case the host values above are
        # intentionally optimistic placeholders; overwrite the authoritative
        # fields from the predecessor's GPU-resident state on the same stream.
        # Page/block tables still come from the CPU allocator, whose optimistic
        # reservation is a safe upper bound for the real accepted length.
        if ctx_lens_gpu is not None:
            self._d_meta[:nd, META_CTX].copy_(ctx_lens_gpu[:nd])
        if x1_gpu is not None:
            self._d_meta[:nd, META_X1].copy_(x1_gpu[:nd])
        if num_accepted_gpu is not None:
            self._d_meta[:nd, META_NUM_ACCEPTED].copy_(
                num_accepted_gpu[:nd]
            )

    def x1_gpu(self, nd: int) -> torch.Tensor:
        """The staged ``x1`` token ids, already on the device (int64 ``[nd]``)."""
        return self._d_meta[:nd, META_X1]

    # ------------------------------------------------------------------
    # device: per-token arrays straight into the static input buffers
    # ------------------------------------------------------------------
    def _fill_common(self, input_data, qlen: int, positions_2d: torch.Tensor):
        """Shared writes for a uniform ``qlen``-per-row batch.

        ``positions_2d`` is ``[bucket, qlen]`` absolute token positions.
        Writes positions (+ mrope), slot_mapping, seq_lens, query_start_loc and
        block_table. Everything is a slice of a persistent buffer, so the
        captured graphs keep reading the same addresses.
        """
        bucket = self.bucket
        ntok = bucket * qlen
        page_size = self.page_size
        pos_flat = positions_2d.reshape(-1)

        input_data.positions[:ntok].copy_(pos_flat)
        if self.uses_mrope:
            # Decode-side mrope positions are the plain position plus the
            # prefill-time delta, identical on all three rows (see
            # ``MRotaryEmbedding.get_next_input_positions``).
            mp = (positions_2d + self._d_meta[:bucket, META_DELTA].unsqueeze(1)).reshape(
                1, -1
            )
            input_data.mrope_positions[:, :ntok].copy_(mp.expand(3, ntok))

        # slot_mapping[t] = page_table[row, pos // P] * P + pos % P
        pidx = positions_2d // page_size
        sidx = positions_2d - pidx * page_size
        phys = torch.gather(self._d_pt[:bucket, : self.pt_cols].to(torch.int64), 1, pidx)
        input_data.slot_mapping[:ntok].copy_((phys * page_size + sidx).reshape(-1))

        input_data.seq_lens[:bucket].copy_(
            (positions_2d[:, -1] + 1).to(input_data.seq_lens.dtype)
        )
        input_data.query_start_loc[: bucket + 1].copy_(
            self._arange(bucket + 1, torch.int32) * qlen
        )
        input_data.block_table[:bucket, : self.pt_cols].copy_(
            self._d_pt[:bucket, : self.pt_cols]
        )

    def _fill_ssm(self, input_data):
        """SSM (hybrid GDN) per-row metadata: block table + resume column."""
        if not input_data.use_ssm_cache:
            return
        bucket = self.bucket
        bt = self._d_bt[:bucket]
        input_data.ssm_state_slot_per_seq[:bucket].copy_(bt[:, 0])
        input_data.has_initial_state_per_seq[:bucket].fill_(True)
        input_data.ssm_snapshot_write_slot_per_seq[:bucket].fill_(-1)
        if self.bt_width and hasattr(input_data, "ssm_block_table_2d"):
            w = self.bt_width
            input_data.ssm_block_table_2d[:bucket, :w].copy_(bt[:, :w])
            input_data.ssm_num_accepted[:bucket].copy_(
                self._d_meta[:bucket, META_NUM_ACCEPTED].to(torch.int32)
            )

    def fill_draft(self, input_data) -> None:
        """Prepare a draft-chain step: one query token per row at ``ctx``.

        Mirrors the CPU build for ``token_ids = committed + [x1]`` with
        ``computed_token_num = ctx`` and ``to_compute_token_num = 1``. The token
        ids themselves are not written here -- the captured draft graph feeds
        the MTP head from ``ModelRunner._d_tok`` instead. The SSM metadata is
        also skipped: the MTP head is a single *full-attention* block (Qwen3.5)
        or an MLA block (DeepSeek), so no GDN layer reads it during a draft.
        """
        bucket = self.bucket
        pos = self._d_meta[:bucket, META_CTX].unsqueeze(1)  # [bucket, 1]
        self._fill_common(input_data, 1, pos)

    def fill_verify(
        self,
        input_data,
        qlen: int,
        drafts_gpu: torch.Tensor,
    ) -> None:
        """Prepare the uniform ``1+k`` verify batch.

        Row ``i`` gets query tokens ``[x1_i, d_i1 .. d_ik]`` at absolute
        positions ``ctx_i .. ctx_i + qlen - 1`` over its cached context.
        ``drafts_gpu`` is the ``[nd, qlen-1]`` GPU draft matrix produced by the
        draft chain -- the token ids never round-trip through the host.
        """
        bucket, nd = self.bucket, self.nd
        ntok = bucket * qlen
        pos = self._d_meta[:bucket, META_CTX].unsqueeze(1) + self._arange(qlen)
        self._fill_common(input_data, qlen, pos)
        self._fill_ssm(input_data)

        tok = input_data.tokens[:ntok].view(bucket, qlen)
        tok[:nd, 0] = self._d_meta[:nd, META_X1]
        if qlen > 1:
            tok[:nd, 1:].copy_(drafts_gpu[:nd, : qlen - 1])
        if bucket > nd:
            # Same filler the dummy pad seqs used (``[1] * (ctx + qlen)``).
            tok[nd:].fill_(1)

    def correct_mixed_prefix(
        self,
        input_data,
        qlen: int,
        drafts_gpu: torch.Tensor,
    ) -> None:
        """Correct the MTP prefix of a ragged mixed target batch on the GPU.

        ``InputData.cal_input`` deliberately builds the successor from the
        scheduler's optimistic fixed-width placeholders.  Query widths and
        page reservations are already valid, but the predecessor's *accepted*
        context length, relay x1, GDN resume column and derived per-token
        metadata only become authoritative on the GPU.  Rewrite just the
        leading uniform verify rows after the ordinary mixed H2D preparation;
        the following ragged prefill rows remain untouched.
        """
        nd = self.nd
        if nd == 0:
            return
        ntok = nd * qlen
        pos = self._d_meta[:nd, META_CTX].unsqueeze(1) + self._arange(qlen)
        pos_flat = pos.reshape(-1)
        input_data.positions[:ntok].copy_(pos_flat)
        if self.uses_mrope:
            mp = (
                pos
                + self._d_meta[:nd, META_DELTA].unsqueeze(1)
            ).reshape(1, -1)
            input_data.mrope_positions[:, :ntok].copy_(mp.expand(3, ntok))

        page_size = self.page_size
        pidx = pos // page_size
        sidx = pos - pidx * page_size
        phys = torch.gather(
            self._d_pt[:nd, : self.pt_cols].to(torch.int64), 1, pidx
        )
        input_data.slot_mapping[:ntok].copy_(
            (phys * page_size + sidx).reshape(-1)
        )
        input_data.seq_lens[:nd].copy_(
            (pos[:, -1] + 1).to(input_data.seq_lens.dtype)
        )
        input_data.block_table[:nd, : self.pt_cols].copy_(
            self._d_pt[:nd, : self.pt_cols]
        )

        tok = input_data.tokens[:ntok].view(nd, qlen)
        tok[:, 0].copy_(self._d_meta[:nd, META_X1])
        if qlen > 1:
            tok[:, 1:].copy_(drafts_gpu[:nd, : qlen - 1])

        if input_data.use_ssm_cache:
            input_data.ssm_state_slot_per_seq[:nd].copy_(self._d_bt[:nd, 0])
            input_data.has_initial_state_per_seq[:nd].fill_(True)
            input_data.ssm_snapshot_write_slot_per_seq[:nd].fill_(-1)
            if self.bt_width and hasattr(input_data, "ssm_block_table_2d"):
                input_data.ssm_block_table_2d[:nd, : self.bt_width].copy_(
                    self._d_bt[:nd, : self.bt_width]
                )
                input_data.ssm_num_accepted[:nd].copy_(
                    self._d_meta[:nd, META_NUM_ACCEPTED].to(torch.int32)
                )
