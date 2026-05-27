"""Delta-based schedule broadcast between rank-0 (driver) and followers.

Why a new module
================

Before this module existed, every scheduling iteration rank-0 pickled a
list of (post-processed) ``Sequence`` objects and shipped it via zmq to
every other TP / PP follower. Even after ``Scheduler.post_schedule``
stripped ``token_ids`` for non-VL non-rep-penalty seqs, each iteration
still re-pickled:

* the page_table for every running seq (grows linearly with sequence
  length; 128-tok decode = 8 ints, 8k-context = 512 ints, **per seq**),
* the full ``mm_contents`` dict for every prefill seq (image bytes /
  URLs -- can be megabytes),
* per-seq immutable sampling params (temperature, top_p, top_k,
  repetition_penalty, output_len, finish_tokens, ignore_eos),
* the original token_ids list when ``repetition_penalty != 1.0`` or
  during a VL prefill chunk.

Steady-state decode on TP=2 / Qwen3-0.6B was ~64 KB / iter just for the
schedule broadcast, and overlap-mode profile showed rank-1 lagging
rank-0 by ~700us per iteration largely because of the
pickle + zmq + unpickle round-trip; ~99 % of the bytes were data the
follower already had from a previous iteration.

This module fixes that by making the followers *stateful*. Each
follower maintains a :class:`FollowerSeqStore` keyed by ``seq_id``. The
wire format becomes a delta:

* :class:`SeqRegister` -- sent **once** per (seq, follower-group) pair,
  carries every immutable per-seq field (prompt token_ids, sampling
  params, ``mm_contents``).
* :class:`SeqUpdate` -- sent **every iter** for every seq in the batch,
  carries only ``computed_token_num``, ``to_compute_token_num``,
  ``to_compute_tokens`` (the tokens to feed the model this iter), and
  newly-allocated page ids appended to the seq's page_table.
* :class:`SchedulePayload.frees` -- piggybacks seq_ids to evict from
  the follower's mirror (also used to drop the follower's
  ``embedding_cache`` entry on VL workloads, which previously leaked
  -- followers never received a free notification).

Steady-state decode payload drops from ~64 KB to ~0.5-2 KB per iter
(~30-100x), prefill registers are amortized over hundreds of decode
iterations, and ``mm_contents`` (the bulk of VL prefill cost) is sent
exactly once per request.

Design notes
============

* Followers stay **stateless about KV pages**. Page allocation is
  centralized on rank-0; followers only mirror the resulting page_table
  to drive ``_cal_block_table`` / ``_cal_slot_mapping``. Preemption is
  signaled via :class:`SeqUpdate.page_table_reset` (the only update
  field that's normally ``None``).

* The :class:`FollowerSeq` mirror duck-types ``Sequence`` for every
  attribute that ``InputData.cal_input``, ``InputData.prepare_sample``,
  and ``ModelRunner._mm_prepare_cpu`` actually read on a follower. We
  did *not* subclass ``Sequence`` -- ``Sequence`` carries fields the
  follower never touches (e.g. ``prompt`` / ``output`` strings, the
  tokenizer-incremental detokenize cursor), and shrinking the mirror
  helps both pickling speed (registers stay small) and per-iteration
  GC pressure.

* The driver registry (:class:`DriverPayloadBuilder`) is **one shared
  set across all follower targets**. We rely on the invariant that all
  followers receive the same logical sequence of schedule payloads
  (zmq PUSH/PULL is FIFO per pair and we fan out the same payload to
  every target). If a follower could miss a message this would break;
  today they can't, and the simpler bookkeeping is a real win.

* Async / overlap compatibility: ``apply_payload`` is a few dict
  inserts + list comprehensions -- no GPU work, no I/O. Safe to call
  on the worker's main thread between an outstanding GPU forward and
  the next ``_launch_batch``, which is exactly where the existing
  overlap path puts the receive.

* VL specifics: ``mm_contents`` and the full prompt ``token_ids`` are
  shipped in :class:`SeqRegister`. ``_mm_prepare_cpu`` runs only on
  ``is_first_pp_rank()`` followers and reads both. Decode iterations
  don't reference ``mm_contents`` (the embed cache covers them), and
  for non-VL non-rep-penalty seqs we skip accumulating ``token_ids``
  on the follower entirely (just mirror the prompt slice for the
  initial prefill; later iters use ``SeqUpdate.to_compute_tokens``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from gllm.sequence import Sequence


# ---------------------------------------------------------------------------
# Wire-format dataclasses (pickled by zmq)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SeqRegister:
    """One-time per-seq registration sent to a follower group.

    Carries every field of ``Sequence`` that the follower will need
    across the seq's lifetime *and* that is either fully immutable or
    bootstrap-only (the prompt token_ids are mutated only by the
    follower itself if it has to accumulate decode tokens for
    repetition-penalty mask building; everything else is immutable).
    """

    seq_id: int
    prompt_token_ids: List[int]
    prompt_len: int
    finish_tokens: List[int]
    ignore_eos: bool
    output_len: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    # ``mm_contents`` is the dict produced by
    # ``ModelRunner.extract_modify_mm`` ({"image": [...], "video": [...]}).
    # For non-VL or non-multimodal requests it's ``None``.
    mm_contents: Optional[Dict] = None
    # If the follower must keep building the per-vocab repetition-penalty
    # mask, it has to accumulate decode tokens into ``token_ids``. Same
    # for VL prefill (until the prompt is fully consumed). For
    # non-VL non-rep-penalty seqs we skip the accumulation (small but
    # non-trivial Python overhead per decode).
    needs_token_id_accumulation: bool = False


@dataclass(slots=True)
class SeqUpdate:
    """Per-iteration delta for a single seq in this batch.

    Order in :attr:`SchedulePayload.updates` is the batch order for
    ``InputData.cal_input``.
    """

    seq_id: int
    computed_token_num: int
    to_compute_token_num: int
    # The exact slice of token ids to feed the model this iter
    # (==``seq[computed_token_num:seq_len]`` on the driver). We always
    # send this so the follower's ``_cal_tokens`` is a single list
    # reference (no slicing into ``token_ids`` needed -- works even
    # when the follower doesn't keep the full history).
    to_compute_tokens: List[int]
    # Page ids appended to the seq's page_table this iter. Empty list
    # in the common decode case where the existing tail page still has
    # room (only one new page per ``page_size`` decode steps).
    new_page_ids: List[int] = field(default_factory=list)
    # Set on the seq's first scheduling and after preemption: the
    # follower resets its mirror's page_table to ``page_table_reset``
    # before appending ``new_page_ids``. Almost always ``None``.
    page_table_reset: Optional[List[int]] = None


@dataclass(slots=True)
class SchedulePayload:
    """Single message rank-0 -> {one follower stage} per iteration.

    ``mrope_positions`` is a special PP-only piggyback: for the
    non-first-PP follower group the driver builds the m-rope positions
    during its own ``cal_input`` (we ship them to spare downstream PP
    stages the redundant ``_mm_prepare_cpu`` work). TP followers of
    the first PP stage receive ``None`` -- they run
    ``_mm_prepare_cpu`` locally.
    """

    registers: List[SeqRegister] = field(default_factory=list)
    updates: List[SeqUpdate] = field(default_factory=list)
    frees: List[int] = field(default_factory=list)
    mrope_positions: Optional[torch.Tensor] = None
    control_cmd: int = 0
    control_data: Any = None

    def is_empty(self) -> bool:
        return (
            not self.registers
            and not self.updates
            and not self.frees
            and self.mrope_positions is None
            and self.control_cmd == 0
        )


# ---------------------------------------------------------------------------
# Driver-side builder (rank-0)
# ---------------------------------------------------------------------------


class DriverPayloadBuilder:
    """Builds :class:`SchedulePayload` from rank-0 scheduler state.

    Tracks per-seq cursors so each iteration's payload is a true delta:

    * ``_known``: seq_ids already registered on followers. Adding to
      this set means we will *not* re-ship the seq's immutable fields
      on subsequent iterations.
    * ``_last_pages_len``: ``len(seq.page_table)`` as of the last
      payload we built for this seq. Lets us slice off only the newly
      appended page ids. Preemption (``len < _last_pages_len``)
      triggers a ``page_table_reset`` so the follower drops stale page
      ids.

    Lifetime: one instance per worker (rank-0). The same instance is
    used for both follower groups (``schedule_first_pp_sockets`` and
    ``schedule_other_sockets``) because every group sees the same
    logical schedule stream.
    """

    def __init__(self):
        self._known: set[int] = set()
        self._last_pages_len: Dict[int, int] = {}

    # ------------------------------------------------------------------ free

    def forget(self, seq_id: int) -> None:
        """Drop driver-side tracking for a seq the followers will free."""
        self._known.discard(seq_id)
        self._last_pages_len.pop(seq_id, None)

    # ------------------------------------------------------------------ build

    def build(
        self,
        scheduled_seqs: List[Sequence],
        frees: List[int],
        mrope_positions: Optional[torch.Tensor] = None,
        control_cmd: int = 0,
        control_data: Any = None,
        use_mm: bool = False,
    ) -> SchedulePayload:
        """Snapshot ``scheduled_seqs`` into a payload + update cursors.

        Must be called from the same thread that mutates the underlying
        ``Sequence`` objects (i.e. the worker's main loop). All fields
        we copy out are either immutable or copied into fresh lists, so
        the payload is safe to hand off to the persistent zmq sender
        thread even if the main thread continues mutating ``seq``
        afterwards (e.g. ``process_output_finalize`` rewriting a
        placeholder).
        """
        registers: List[SeqRegister] = []
        updates: List[SeqUpdate] = []

        for seq in scheduled_seqs:
            sid = seq.seq_id
            if sid not in self._known:
                # Decide whether the follower needs to keep accumulating
                # token_ids over the seq's lifetime. ``mm_contents`` is
                # only consulted by ``_mm_prepare_cpu`` for *uncached*
                # prefill seqs (a one-shot read), so we don't gate on
                # it; we gate on rep penalty, which needs token_ids on
                # *every* decode for the per-vocab mask. We also keep
                # token_ids alive for any VL seq for the duration of
                # its prefill so ``_mm_prepare_cpu``'s
                # ``torch.tensor(seq.token_ids, ...)`` + ``isin`` path
                # has a full prompt to walk.
                needs_token_id_accumulation = (
                    seq.repetition_penalty != 1.0
                )
                registers.append(
                    SeqRegister(
                        seq_id=sid,
                        # ``list(...)`` to materialize a fresh list so
                        # the follower's mirror doesn't accidentally
                        # alias the driver's ``Sequence.token_ids``
                        # (which the driver continues to mutate via
                        # ``seq.append`` / placeholder rewrites in
                        # ``OverlapScheduler``).
                        prompt_token_ids=list(seq.token_ids[: seq.prompt_len]),
                        prompt_len=seq.prompt_len,
                        finish_tokens=list(seq.finish_tokens),
                        ignore_eos=seq.ignore_eos,
                        output_len=seq.output_len,
                        temperature=seq.temperature,
                        top_p=seq.top_p,
                        top_k=seq.top_k,
                        repetition_penalty=seq.repetition_penalty,
                        # ``mm_contents`` is a small dict of refs / URLs
                        # / bytes; pickle-by-reference is fine. The
                        # driver mutates it only in
                        # ``extract_modify_mm`` *before* the seq is
                        # ever scheduled, so it's effectively
                        # immutable here.
                        mm_contents=seq.mm_contents if use_mm else None,
                        needs_token_id_accumulation=needs_token_id_accumulation,
                    )
                )
                self._known.add(sid)
                # Reset page_table tracking for the new registration.
                self._last_pages_len[sid] = 0

            last_n = self._last_pages_len.get(sid, 0)
            cur_n = len(seq.page_table)
            page_table_reset: Optional[List[int]] = None
            if cur_n < last_n:
                # Preemption: ``Sequence.preempt`` cleared page_table.
                # Re-baseline the follower's mirror from scratch.
                page_table_reset = []
                new_page_ids = list(seq.page_table)
            else:
                # Steady state: ship only the appended tail.
                new_page_ids = list(seq.page_table[last_n:cur_n])
            self._last_pages_len[sid] = cur_n

            updates.append(
                SeqUpdate(
                    seq_id=sid,
                    computed_token_num=seq.computed_token_num,
                    to_compute_token_num=seq.to_compute_token_num,
                    # ``list(...)`` snapshot to detach from any later
                    # mutation of ``seq.token_ids`` by the main loop.
                    to_compute_tokens=list(
                        seq[seq.computed_token_num : seq.seq_len]
                    ),
                    new_page_ids=new_page_ids,
                    page_table_reset=page_table_reset,
                )
            )

        # Garbage-collect cursors for freed seqs *after* building this
        # payload, so the FREE list still references them and the
        # follower can clean up its own mirror.
        for sid in frees:
            self.forget(sid)

        return SchedulePayload(
            registers=registers,
            updates=updates,
            frees=list(frees),
            mrope_positions=mrope_positions,
            control_cmd=control_cmd,
            control_data=control_data,
        )


# ---------------------------------------------------------------------------
# Follower-side mirror
# ---------------------------------------------------------------------------


class FollowerSeq:
    """Lightweight per-seq mirror on a TP/PP follower rank.

    Quacks like ``gllm.sequence.Sequence`` for every attribute that
    :class:`gllm.input_data.InputData` (``cal_input`` /
    ``prepare_sample``) and
    :meth:`gllm.model_runner.ModelRunner._mm_prepare_cpu` actually read
    on a follower. Intentionally **not** a subclass of ``Sequence`` --
    we want a tight slot layout and an obvious surface contract
    (anything on this class is a thing followers genuinely consume).

    Attribute notes:

    * ``token_ids`` is ``None`` when the follower doesn't need to
      maintain a token-id mirror (i.e. non-VL prefill *and*
      ``repetition_penalty == 1.0``). In that case
      ``_cal_tokens`` uses ``to_compute_tokens`` directly and
      ``_mm_prepare_cpu`` is never called for this seq.
    """

    __slots__ = (
        "seq_id",
        "prompt_len",
        "token_ids",
        "page_table",
        "computed_token_num",
        "to_compute_token_num",
        "to_compute_tokens",
        "mm_contents",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "finish_tokens",
        "ignore_eos",
        "output_len",
        "_keeps_token_ids",
    )

    def __init__(self, reg: SeqRegister, mm_needs_token_ids: bool = False):
        self.seq_id = reg.seq_id
        self.prompt_len = reg.prompt_len
        # Keep token_ids alive throughout the seq's lifetime when:
        #   * VL: ``_mm_prepare_cpu`` walks the prompt to build the
        #     ``is_multimodal`` mask AND to compute MROPE positions on the
        #     uncached prefill path. This is true for *every* prefill seq
        #     in a VL model (even text-only ones with ``mm_contents=None``),
        #     because ``MRotaryEmbedding.get_input_positions`` reads
        #     ``seq.token_ids`` unconditionally. ``mm_needs_token_ids`` --
        #     a per-engine flag set by the follower when the loaded model
        #     uses multimodal embeddings -- forces ``_keeps_token_ids`` on
        #     for these seqs; otherwise a text-only prompt to a VL model
        #     would crash on the follower with ``torch.tensor(None)``.
        #   * repetition_penalty != 1.0: ``build_repetition_penalty_mask``
        #     scans the full token history on every decode.
        # Otherwise we can drop the prompt history after the first
        # prefill iteration to save memory + per-decode list grow cost.
        self._keeps_token_ids = (
            reg.needs_token_id_accumulation
            or reg.mm_contents is not None
            or mm_needs_token_ids
        )
        self.token_ids: Optional[List[int]] = (
            list(reg.prompt_token_ids) if self._keeps_token_ids else None
        )
        self.page_table: List[int] = []
        # ``computed_token_num`` and friends are overwritten by the
        # very first ``apply_update`` (which always lands together with
        # the register), so the seed values just need to be
        # well-typed.
        self.computed_token_num = 0
        self.to_compute_token_num = 0
        self.to_compute_tokens: Optional[List[int]] = None
        self.mm_contents = reg.mm_contents
        self.temperature = reg.temperature
        self.top_p = reg.top_p
        self.top_k = reg.top_k
        self.repetition_penalty = reg.repetition_penalty
        self.finish_tokens = reg.finish_tokens
        self.ignore_eos = reg.ignore_eos
        self.output_len = reg.output_len

    # ---- duck-typed Sequence surface --------------------------------------

    @property
    def seq_len(self) -> int:
        return self.computed_token_num + self.to_compute_token_num

    @property
    def computed_prompt(self) -> bool:
        return self.computed_token_num >= self.prompt_len

    def __len__(self) -> int:
        if self.token_ids is not None:
            return len(self.token_ids)
        # Best-effort fallback for callers that compute "current
        # position" from ``len(seq)``. The driver-side ``Sequence``
        # has ``len = prompt_len + #generated``; we mirror that
        # without keeping per-token state.
        return self.seq_len

    def __getitem__(self, key):
        if self.token_ids is None:
            raise RuntimeError(
                f"FollowerSeq({self.seq_id}) has no token_ids mirror; "
                "this seq was registered without "
                "needs_token_id_accumulation and is not VL."
            )
        return self.token_ids[key]

    # ---- update plumbing --------------------------------------------------

    def apply_update(self, upd: SeqUpdate) -> None:
        """In-place absorb a per-iter delta from the driver."""
        self.computed_token_num = upd.computed_token_num
        self.to_compute_token_num = upd.to_compute_token_num
        self.to_compute_tokens = upd.to_compute_tokens

        if upd.page_table_reset is not None:
            # Preemption / first scheduling.
            self.page_table = list(upd.page_table_reset)
        if upd.new_page_ids:
            self.page_table.extend(upd.new_page_ids)

        if self._keeps_token_ids and upd.to_compute_tokens:
            # Append only the actually-new tokens. For an uncached
            # prefill chunk this is just the chunk that's about to be
            # consumed (and it already exists in ``token_ids`` since
            # ``__init__`` materialized the prompt -- we skip the
            # extend in that case to avoid duplicating the prompt).
            assert self.token_ids is not None
            new_end = upd.computed_token_num + upd.to_compute_token_num
            cur_end = len(self.token_ids)
            if new_end > cur_end:
                # Take only the suffix that we don't already have. This
                # handles both decode (always +1 token after prefill
                # done) and chunked-prefill carry-over without
                # duplicating the prompt window we copied at register
                # time.
                self.token_ids.extend(
                    upd.to_compute_tokens[cur_end - upd.computed_token_num :]
                )


class FollowerSeqStore:
    """Per-rank registry of :class:`FollowerSeq` mirrors.

    Lives on every worker except rank-0. Single-threaded by
    construction (only the worker's main loop touches it). API is
    explicitly delta-shaped:

    * :meth:`apply_payload` consumes a :class:`SchedulePayload`
      atomically and returns the list of mirrors that constitute this
      iteration's batch (preserving payload order, which is the batch
      order rank-0 wants for ``cal_input``).

    * :meth:`active_count` is for diagnostics only (sanity checks in
      tests, no impact on the hot path).
    """

    def __init__(self, mm_needs_token_ids: bool = False):
        # ``mm_needs_token_ids`` is a one-shot per-engine flag set by the
        # worker when the loaded model is multimodal (``ModelRunner.use_mm``).
        # Propagated to every :class:`FollowerSeq` so its ``_keeps_token_ids``
        # gate stays on for text-only prompts to a VL model -- without this,
        # ``_mm_prepare_cpu`` crashes on followers with
        # ``RuntimeError: Could not infer dtype of NoneType`` when it does
        # ``torch.tensor(seq.token_ids)`` on a seq whose follower mirror
        # dropped the prompt history (because ``mm_contents`` is None and
        # ``needs_token_id_accumulation`` is False).
        self._table: Dict[int, FollowerSeq] = {}
        self._mm_needs_token_ids = mm_needs_token_ids

    def apply_payload(self, payload: SchedulePayload) -> List[FollowerSeq]:
        """Apply registers + updates, then free what's evicted.

        Returns the ordered batch of mirrors corresponding to
        ``payload.updates``. ``free`` is processed last so a payload
        that frees a seq it also re-registers (currently impossible,
        but cheap to be defensive about) does the right thing.
        """
        for reg in payload.registers:
            # Overwrite an existing entry rather than asserting -- if
            # rank-0's registry believes the follower needs a fresh
            # register (e.g. after a state-sync recovery in a future
            # PD-disagg path), it would resend. Today this branch is
            # never hit under normal operation.
            self._table[reg.seq_id] = FollowerSeq(
                reg, mm_needs_token_ids=self._mm_needs_token_ids
            )

        seqs: List[FollowerSeq] = []
        for upd in payload.updates:
            seq = self._table.get(upd.seq_id)
            if seq is None:
                # The only way to reach here is a wire-protocol bug --
                # an update for a seq we never registered. Loudly fail
                # rather than silently mis-batch; we'd otherwise stage
                # garbage into ``cal_input`` and corrupt the forward.
                raise KeyError(
                    f"FollowerSeqStore: update for unregistered seq_id="
                    f"{upd.seq_id}; sender state is out of sync."
                )
            seq.apply_update(upd)
            seqs.append(seq)

        for sid in payload.frees:
            # ``pop(..., None)``: tolerate redundant frees (e.g. the
            # frontend / rank-0 may double-emit on shutdown).
            self._table.pop(sid, None)

        return seqs

    def evict(self, seq_id: int) -> Optional[FollowerSeq]:
        """Explicit eviction hook (used by shutdown / abort paths)."""
        return self._table.pop(seq_id, None)

    def active_count(self) -> int:
        return len(self._table)


__all__ = [
    "SeqRegister",
    "SeqUpdate",
    "SchedulePayload",
    "DriverPayloadBuilder",
    "FollowerSeq",
    "FollowerSeqStore",
]
