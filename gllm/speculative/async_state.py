"""GPU-resident state for overlapped MTP speculative decoding.

The ordinary overlap pipeline can represent one future token with
``FutureMap``.  MTP produces a variable-width committed prefix, so keeping the
same pipeline depth requires two additional pieces of state:

* the real context length and next relay seed stay in persistent GPU buffers;
  and
* a fixed-width, ``-1`` padded completion record is copied to pinned host
  memory asynchronously for deferred scheduler finalization.

This module owns those buffers only. The worker launches a successor directly
from the producing step's GPU state, then collects the predecessor while that
successor runs. Optimistic placeholders and GDN state-column selection remain
explicit in the worker so the synchronization boundary is easy to audit.
"""

from dataclasses import dataclass
from typing import List, Sequence as SequenceType, Tuple

import torch


@dataclass
class MtpAsyncCompletion:
    """One in-flight MTP completion copied into a pinned ring slot."""

    owner: "MtpAsyncBatchState"
    slot: int
    seq_ids: Tuple[int, ...]
    batch_size: int
    event: torch.cuda.Event
    output_seq_ids: Tuple[int, ...] = ()
    extra_batch_size: int = 0

    def collect(self) -> Tuple[List[int], List[List[int]]]:
        """Wait for D2H and return ``(valid_counts, committed)``.

        ``valid_counts`` includes the always-committed x1. ``committed`` has
        already had its ``-1`` padding removed and is therefore ready for the
        scheduler's variable-length commit path.
        """
        self.event.synchronize()
        host = self.owner._host[self.slot][: self.batch_size]
        valid = host[:, 0].tolist()
        grid = host[:, 1:].tolist()
        committed = [row[:n] for row, n in zip(grid, valid)]
        if self.extra_batch_size:
            extra = self.owner._extra_host[self.slot][: self.extra_batch_size]
            committed.extend([[int(token)] for token in extra.tolist()])
        self.owner._release(self.slot)
        return valid, committed


class MtpAsyncBatchState:
    """Double-buffered GPU state for an overlapped MTP request cohort.

    Rows are keyed by ``seq_id`` rather than by their transient batch index.
    When requests retire or the scheduler compacts/reorders a batch,
    :meth:`remap` gathers the surviving GPU state into the successor's row
    order.  The completion ring is independent of these live state buffers, so
    an older completion can still be copied/finalized using the row ids it
    captured at publication time without draining the MTP pipeline.
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        k: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.k = k
        self.width = 1 + k
        self.device = device
        self.seq_ids: Tuple[int, ...] = ()
        self.batch_size = 0

        self.context_lens = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self.relay_tokens = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self.relay_hidden = torch.empty(
            (max_batch_size, hidden_size), dtype=hidden_dtype, device=device
        )
        # Number of tokens committed by the latest verify step (x1 plus the
        # accepted draft prefix).  During a stable async cohort the CPU block
        # table deliberately stays unchanged; the next GDN verify reads this
        # tensor and resumes directly from column ``valid_count - 1``.
        self.resume_num_accepted = torch.ones(
            max_batch_size, dtype=torch.int64, device=device
        )
        # Read-only draft-column ids are reused by every acceptance pack.
        self._draft_cols = torch.arange(
            k, dtype=torch.int64, device=device
        ).unsqueeze(0)

        # Persistent row-remap storage. Cohort compaction is infrequent, but it
        # sits exactly on the async launch boundary; keeping indices and gather
        # destinations stable avoids allocator work and lets the remap enqueue
        # on the producer stream behind the predecessor's state publication.
        self._remap_idx_host = torch.empty(
            max_batch_size, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._remap_idx_host_np = self._remap_idx_host.numpy()
        self._remap_idx_device = torch.empty(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._scratch_context_lens = torch.empty_like(self.context_lens)
        self._scratch_relay_tokens = torch.empty_like(self.relay_tokens)
        self._scratch_relay_hidden = torch.empty_like(self.relay_hidden)
        self._scratch_resume_num_accepted = torch.empty_like(
            self.resume_num_accepted
        )

        # Completion layout per row:
        #   [valid_count, current_x1, accepted_drafts..., -1 padding]
        # The relay bonus remains in the authoritative GPU state and is copied
        # only at a real drain, so it does not belong in every host completion.
        shape = (max_batch_size, 1 + self.width)
        self._device = [
            torch.empty(shape, dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._host = [
            torch.empty(
                shape, dtype=torch.int64, device="cpu", pin_memory=True
            )
            for _ in range(2)
        ]
        # Mixed target forwards also sample one token for every prefill row.
        # Keep those tokens in the same completion slot/event as the decode
        # acceptance record so deferred scheduler finalization has one atomic
        # batch result in the original [decode | prefill] row order.
        self._extra_device = [
            torch.empty(max_batch_size, dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._extra_host = [
            torch.empty(
                max_batch_size, dtype=torch.int64, device="cpu", pin_memory=True
            )
            for _ in range(2)
        ]
        self._events = [torch.cuda.Event() for _ in range(2)]
        self._busy = [False, False]
        self._next_slot = 0
        self.copy_stream = torch.cuda.Stream(device=device)

    def install(
        self,
        seq_ids: SequenceType[int],
        context_lens: torch.Tensor,
        relay_tokens: torch.Tensor,
        relay_hidden: torch.Tensor,
    ) -> None:
        """Install a synchronized CPU/GPU boundary as the async batch base."""
        n = len(seq_ids)
        if n > self.max_batch_size:
            raise ValueError((n, self.max_batch_size))
        if any(self._busy):
            raise RuntimeError("cannot replace MTP async batch with pending completions")
        self.seq_ids = tuple(int(x) for x in seq_ids)
        self.batch_size = n
        self.context_lens[:n].copy_(context_lens[:n], non_blocking=True)
        self.relay_tokens[:n].copy_(relay_tokens[:n])
        self.relay_hidden[:n].copy_(relay_hidden[:n])
        self.resume_num_accepted[:n].fill_(1)

    def matches(self, seq_ids: SequenceType[int]) -> bool:
        return tuple(int(x) for x in seq_ids) == self.seq_ids

    def can_remap(self, seq_ids: SequenceType[int]) -> bool:
        """Whether every successor row already has authoritative GPU state."""
        ids = tuple(int(x) for x in seq_ids)
        if not ids or len(ids) != len(set(ids)):
            return False
        current = set(self.seq_ids)
        return all(seq_id in current for seq_id in ids)

    @torch.inference_mode()
    def remap(self, seq_ids: SequenceType[int]) -> bool:
        """Gather live state into ``seq_ids`` order without synchronizing.

        Returns ``True`` when a physical remap was enqueued and ``False`` for
        an already-identical cohort. This method must run on the same CUDA
        stream as :meth:`publish`; stream order then guarantees that it reads
        the predecessor's latest relay/context/GDN state.

        Busy completion slots are legal: their D2H source is the immutable
        per-slot completion record, not the live tensors rearranged here.
        """
        ids = tuple(int(x) for x in seq_ids)
        if ids == self.seq_ids:
            return False
        if not self.can_remap(ids):
            raise ValueError(
                f"cannot remap MTP async cohort {self.seq_ids} to {ids}"
            )

        old_row = {seq_id: i for i, seq_id in enumerate(self.seq_ids)}
        n = len(ids)
        self._remap_idx_host_np[:n] = [old_row[seq_id] for seq_id in ids]
        self._remap_idx_device[:n].copy_(
            self._remap_idx_host[:n], non_blocking=True
        )
        idx = self._remap_idx_device[:n]
        torch.index_select(
            self.context_lens,
            0,
            idx,
            out=self._scratch_context_lens[:n],
        )
        torch.index_select(
            self.relay_tokens,
            0,
            idx,
            out=self._scratch_relay_tokens[:n],
        )
        torch.index_select(
            self.relay_hidden,
            0,
            idx,
            out=self._scratch_relay_hidden[:n],
        )
        torch.index_select(
            self.resume_num_accepted,
            0,
            idx,
            out=self._scratch_resume_num_accepted[:n],
        )
        # The gathered destinations are already full-size persistent buffers.
        # Swap their roles instead of copying all four tensors back into the
        # previous live buffers. This halves remap kernels and bytes moved while
        # preserving stable storage for the next gather.
        self.context_lens, self._scratch_context_lens = (
            self._scratch_context_lens,
            self.context_lens,
        )
        self.relay_tokens, self._scratch_relay_tokens = (
            self._scratch_relay_tokens,
            self.relay_tokens,
        )
        self.relay_hidden, self._scratch_relay_hidden = (
            self._scratch_relay_hidden,
            self.relay_hidden,
        )
        self.resume_num_accepted, self._scratch_resume_num_accepted = (
            self._scratch_resume_num_accepted,
            self.resume_num_accepted,
        )
        self.seq_ids = ids
        self.batch_size = n
        return True

    def publish(
        self,
        *,
        current_x1: torch.Tensor,
        drafts: torch.Tensor,
        num_accepted_drafts: torch.Tensor,
        next_bonus: torch.Tensor,
        next_hidden: torch.Tensor,
        producer_stream: torch.cuda.Stream,
        extra_tokens: torch.Tensor | None = None,
        extra_seq_ids: SequenceType[int] = (),
        new_state_seq_ids: SequenceType[int] = (),
        new_state_context_lens: torch.Tensor | None = None,
        new_state_tokens: torch.Tensor | None = None,
        new_state_hidden: torch.Tensor | None = None,
    ) -> MtpAsyncCompletion:
        """Advance GPU relay/context state and enqueue a nonblocking D2H record."""
        n = self.batch_size
        if n == 0:
            raise RuntimeError("MTP async batch is not installed")
        slot = self._next_slot
        if self._busy[slot]:
            raise RuntimeError("MTP async completion ring overflow")
        self._next_slot = 1 - slot
        self._busy[slot] = True

        na = num_accepted_drafts[:n].to(torch.int64)
        valid = na + 1
        out = self._device[slot][:n]
        out[:, 0].copy_(valid)
        grid = out[:, 1:]
        grid.fill_(-1)
        grid[:, 0].copy_(current_x1[:n].to(torch.int64))
        if self.k:
            accepted = self._draft_cols < na.unsqueeze(1)
            grid[:, 1:].copy_(
                torch.where(
                    accepted,
                    drafts[:n, : self.k].to(torch.int64),
                    torch.full_like(
                        drafts[:n, : self.k], -1, dtype=torch.int64
                    ),
                )
            )

        # These tensors are the authoritative inputs for the next iteration;
        # no CPU acceptance result is needed to advance the model pipeline.
        self.context_lens[:n].add_(valid)
        self.relay_tokens[:n].copy_(next_bonus[:n])
        self.relay_hidden[:n].copy_(next_hidden[:n])
        self.resume_num_accepted[:n].copy_(valid)

        # A prefill row whose prompt completes in this mixed forward already
        # has exactly the relay MTP needs: its final prompt hidden predicts the
        # sampled x1 token. Append those rows to the live request-id keyed GPU
        # cohort immediately, before the CPU scheduler sees the completion.
        # The next batch can therefore include them without a bootstrap/drain.
        new_ids = tuple(int(x) for x in new_state_seq_ids)
        nn = len(new_ids)
        if nn:
            if (
                new_state_context_lens is None
                or new_state_tokens is None
                or new_state_hidden is None
            ):
                raise ValueError("new MTP state rows require context/token/hidden")
            if n + nn > self.max_batch_size:
                raise RuntimeError(
                    f"MTP async cohort overflow: {n}+{nn}>{self.max_batch_size}"
                )
            if set(new_ids).intersection(self.seq_ids):
                raise RuntimeError("new MTP state rows duplicate an existing request")
            sl = slice(n, n + nn)
            self.context_lens[sl].copy_(new_state_context_lens[:nn])
            self.relay_tokens[sl].copy_(new_state_tokens[:nn])
            self.relay_hidden[sl].copy_(new_state_hidden[:nn])
            self.resume_num_accepted[sl].fill_(1)
            self.seq_ids = self.seq_ids + new_ids
            self.batch_size = n + nn

        extra_ids = tuple(int(x) for x in extra_seq_ids)
        ne = len(extra_ids)
        if ne:
            if extra_tokens is None or extra_tokens.numel() < ne:
                raise ValueError("mixed MTP completion is missing prefill tokens")
            self._extra_device[slot][:ne].copy_(extra_tokens[:ne].to(torch.int64))

        event = self._events[slot]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(producer_stream)
            self._host[slot][:n].copy_(out, non_blocking=True)
            if ne:
                self._extra_host[slot][:ne].copy_(
                    self._extra_device[slot][:ne], non_blocking=True
                )
            event.record(self.copy_stream)
        return MtpAsyncCompletion(
            self,
            slot,
            tuple(int(x) for x in self.seq_ids[:n]),
            n,
            event,
            output_seq_ids=tuple(int(x) for x in self.seq_ids[:n]) + extra_ids,
            extra_batch_size=ne,
        )

    def _release(self, slot: int) -> None:
        if not self._busy[slot]:
            raise RuntimeError("MTP async completion slot released twice")
        self._busy[slot] = False

    def reset(self) -> None:
        if any(self._busy):
            raise RuntimeError("drain MTP async completions before reset")
        self.seq_ids = ()
        self.batch_size = 0
