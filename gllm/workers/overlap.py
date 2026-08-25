"""Worker driver loops for overlap scheduling (TP only, pp_size must be 1).

Distributed scheduler design (v2)
=================================

Pre-refactor, only rank 0 (PP=0, TP=0) ran a scheduler. Every iteration
it pickled a delta-style :class:`SchedulePayload` and pushed it over
zmq to each TP follower (and PP-other rank). Profiling showed two
expensive things on the critical path:

* ``send_pyobj`` + ``poll`` skew of ~1.7 ms between rank 0 and the TP
  followers, which inflated the *first* AR kernel of every forward to
  ~2 ms while rank 0 spun on the followers.
* ~50-200 us per-iter pickle + recv + ``apply_payload`` cost on the
  followers, even after the delta refactor.

The new design moves the scheduler onto **every PP-0 TP rank** (a
"column driver"). Each column driver:

1. Receives new front-end work (new requests, aborts, control commands)
   via :meth:`zmqComm.broadcast_input_to_tp` -- a single per-iter
   pyobj fan-out from rank 0 to its PP=0 TP peers over ipc:// zmq
   PUSH/PULL. Rank 0 sends ``None`` on the steady-state decode case
   (which the receiver pickles into ~5 bytes) so peers stay
   lock-stepped without any per-iter NCCL traffic. Steady-state cost
   is ~1-3 us / iter and stays entirely on the CPU side, freeing
   NVLink for the model's per-layer all-reduce.
2. Runs the scheduler locally with the same inputs. Determinism is the
   load-bearing invariant -- ``IDAllocator`` is FIFO-deque-backed and
   the only stochastic call (``random.randint(0, pp_size-1)``)
   collapses to ``0`` for ``pp_size == 1``, so all column drivers
   produce identical schedules / page tables / free orders.
3. Builds its own ``InputData`` and launches forward. Sampled tokens
   are still NCCL-broadcast on the TP group inside ``run_batch_async``
   (this part of the topology is unchanged), but every PP-0 TP rank
   now D2H-copies the result so its local scheduler can finalize
   independently.
4. Only rank 0 forwards the resulting ``IPCPackage`` back to the
   front-end via ``comm.send_output`` -- the others compute the same
   one and discard it.

Compatibility note: this module rejects ``pp_size > 1`` like before.
The new design generalizes (each column driver would also push its
own per-column ``SchedulePayload`` to its PP-other followers), but
:class:`OverlapModelRunner` itself is single-stage; PP > 1 falls back
to the (also-refactored) :class:`gllm.workers.worker.Worker` path.
"""

from collections import deque
from dataclasses import dataclass, field

import torch

from gllm.distributed.parallel_state import (
    dp_all_gather_meta,
    is_dp_attn,
    set_dp_forward_counts,
)
from gllm.runtime.input_data import InputData
from gllm.runtime.model_runner import OverlapModelRunner
from gllm.scheduling.scheduler import OverlapScheduler
from gllm.workers.worker import Worker


@dataclass
class _PendingMtpBatch:
    """One MTP item in gLLM's launch-current/collect-previous pipeline."""

    completion: object
    seqs: list
    decode_seqs: list = field(default_factory=list)
    deferred: object = None
    # Sequences logically retired while this batch was already in flight.
    # Their pages can be released only after this completion is ready.
    release_after: list = field(default_factory=list)


@dataclass(frozen=True)
class _MtpBatchPlan:
    """One scheduler tick's MTP decision, independent of execution details.

    ``decode_ids`` is the only identity carried across a possible predecessor
    drain. The prefetched batch itself may be compacted after EOS/frees, so the
    launch path re-partitions surviving rows by these stable request ids.
    """

    speculate: bool = False
    greedy: bool = False
    decode_ids: tuple = ()

    @property
    def use_async(self) -> bool:
        return self.speculate and self.greedy


@dataclass(frozen=True)
class _MtpBatch:
    """A scheduled batch split once into its MTP decode/prefill regions."""

    seqs: list
    decode: list
    prefill: list


class OverlapWorker(Worker):
    """Overlap-scheduling worker with FutureMap (single PP stage only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pp_size > 1:
            raise ValueError(
                "overlap_scheduling requires pp_size=1; use the default worker for PP>1"
            )
        if not isinstance(self.model_runner, OverlapModelRunner):
            raise TypeError(
                "OverlapWorker requires OverlapModelRunner when overlap_scheduling is enabled"
            )

    def init(self):
        Worker.init(self)
        self._prefetched_input = None
        self._gpu_pending = deque()
        self._mtp_pending = deque()
        # Fixed after ``init``: DP-attention + EP needs the per-iter cross-DP
        # barrier + dummy-batch lockstep in ``run_pp0``; plain TP does not.
        self._dp = is_dp_attn()

    def _init_role_state(self):
        """Override base setup: every PP=0 TP rank gets an OverlapScheduler.

        ``pp_size == 1`` is a hard precondition (enforced in
        ``__init__``), so every rank lands on the ``is_first_pp_rank``
        branch -- no FollowerSeqStore / InputData queue is created.
        We deliberately skip the :class:`DriverPayloadBuilder` setup
        too: the per-column zmq fanout that the base path uses to ship
        delta payloads to PP-other followers has nothing to do for
        PP=1, and not building the payload also means we don't drain
        ``scheduler.consume_pending_follower_frees``.
        """
        self.scheduler = OverlapScheduler(
            self.pp_size,
            self.model_runner,
            self.schedule_method,
        )

    # ------------------------------------------------------------------
    # Forward-pipeline helpers
    # ------------------------------------------------------------------
    #
    # ``recv_ipc_package`` / ``check_abort_seqs`` / ``_translate_control_cmd``
    # are inherited unchanged from :class:`Worker`. The new column-driver
    # base class already runs them on every PP=0 TP rank with the zmq
    # input fan-out, which is exactly what overlap mode needs.

    def _make_scheduled_input(self, seqs) -> InputData:
        """Build either a regular CPU view or minimal fused-MTP bookkeeping."""
        input_data = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        if self.model_runner.mtp_prep_eligible(seqs):
            # The GPU prep path owns all per-token arrays; retain only the batch
            # partition consumed by the next worker tick.
            input_data.seqs = seqs
            input_data.embedding_size = 0
            input_data.is_mtp_verify = False
            input_data.num_mtp_verify_rows = 0
            input_data.num_decodes = len(seqs)
            input_data.num_decode_tokens = len(seqs)
            input_data.num_prefills = 0
            input_data.max_query_len = 1
        else:
            input_data.cal_input(seqs)
        return input_data

    def _build_prefetched_input(self) -> None:
        """Schedule the next batch locally; no inter-TP zmq send.

        Pre-refactor we'd build a delta-style :class:`SchedulePayload`
        and ship it to TP followers here so their ``cal_input``
        overlapped with ours. With the column-driver design every TP
        rank reaches this method on its own schedule loop, runs the
        same deterministic scheduler against the same state, and
        builds its own ``InputData`` -- so there's nothing to send.
        """
        # Drain the scheduler's pending-follower-frees accumulator
        # every iter. Pre-refactor, ``Worker._build_schedule_payload``
        # consumed it on the way to building the per-iter delta
        # payload; the new design has no payload to build (PP=1, no
        # followers), so the list would otherwise grow unbounded as
        # seqs hit max_len / EOS. Cheap (a list = []) and keeps
        # peak-memory predictable.
        with torch.profiler.record_function("gllm::schedule_and_cpu_prepare"):
            self.scheduler.consume_pending_follower_frees()
            schedule_seqs = self.scheduler.schedule_once()
            if len(schedule_seqs) == 0:
                self._prefetched_input = None
                return
            self._prefetched_input = self._make_scheduled_input(schedule_seqs)

    def _launch_batch(self, input_data: InputData, dp_padded_size=None):
        # Pipelined input prep:
        #   * ``prepare_input_cpu`` is pure CPU work and overlaps with the
        #     previous batch's GPU forward.
        #   * ``prepare_input_gpu`` enqueues H2D + (VL) embed work on the
        #     overlap runtime's ``prep_stream``, which GPU-waits for the
        #     previous batch's ``input_consumed_event``. There is no
        #     host-side sync here -- the ordering is entirely expressed via
        #     CUDA events, so the host thread races ahead and the GPU bubble
        #     between back-to-back forwards collapses to the cross-stream
        #     wait_event cost.
        #   * ``run_batch_async`` then dispatches forward+sample on
        #     ``forward_stream`` with a wait_event on ``input_ready_event``.
        self.model_runner.prepare_input_cpu(input_data)
        self.model_runner.prepare_input_gpu()
        return self.model_runner.run_batch_async(dp_padded_size=dp_padded_size)

    def _build_dummy_input(self, size: int = 1) -> InputData:
        """Build a throwaway ``size``-token decode batch for an idle DP group.

        Idle groups must still enter the forward (its MoE layers run a
        collective over the whole DP/EP world), so they ride along with a dummy
        batch whose sampled tokens are discarded. The dummy references the
        memory manager's dummy pages, so it never touches real KV state.
        """
        seqs = self.model_runner.create_dummy_seqs(size, runtime=True)
        dummy = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        dummy.cal_input(seqs)
        return dummy

    def _collect_batch(self, entry) -> None:
        """Wait for a batch's D2H copy and finalize its seq state.

        Every PP-0 TP rank reads its own ``_next_tokens_bufs`` slot (each rank
        did its own D2H copy from the broadcast tokens inside
        ``run_batch_async``) and updates its local GenerationSequence state / frees pages;
        only the frontend poller (rank 0, or each DP group's ``tp_rank == 0``)
        forwards the resulting ``IPCPackage``. ``is_dummy`` batches (idle DP
        groups) and empty ``deferred`` carry no output and are skipped.
        """
        copy_done, batch_size, buf_idx, deferred, _input_data, is_dummy, lp_k = entry
        copy_done.synchronize()
        if is_dummy or deferred is None:
            return
        tokens = self.model_runner._next_tokens_bufs[buf_idx][:batch_size].tolist()
        # Logprobs were staged only on the output rank (== the frontend poller
        # for PP=1); other columns skip the read since their IPC is discarded.
        logprobs = None
        if lp_k is not None and self._polls_frontend():
            logprobs = self._read_logprobs(buf_idx, batch_size, lp_k)
        ipc_package = self.scheduler.process_output_finalize(
            deferred, tokens, logprobs
        )
        if ipc_package is not None and self._polls_frontend():
            self.comm.send_output(ipc_package)

    def _read_logprobs(self, buf_idx: int, batch_size: int, lp_k: int):
        """Materialize this batch's staged logprobs as a per-row Python list.

        Returns a list of ``(sampled_logprob, top_ids, top_vals)`` indexed by
        the batch (== schedule) position, which ``process_output_finalize``
        keys into by ``batch_idx``.
        """
        mr = self.model_runner
        sampled = mr._lp_sampled_bufs[buf_idx][:batch_size].tolist()
        if lp_k > 0:
            ids = mr._lp_topid_bufs[buf_idx][:batch_size, :lp_k].tolist()
            vals = mr._lp_topval_bufs[buf_idx][:batch_size, :lp_k].tolist()
        else:
            ids = [[] for _ in range(batch_size)]
            vals = [[] for _ in range(batch_size)]
        return [(sampled[i], ids[i], vals[i]) for i in range(batch_size)]

    def _drain_pending(self) -> None:
        while self._gpu_pending:
            self._collect_batch(self._gpu_pending.popleft())

    def _collect_mtp_batch(
        self,
        pending: _PendingMtpBatch,
        *,
        materialize_state: bool,
        defer_frees_to=None,
    ) -> None:
        """Collect an MTP item after its successor has optionally launched."""
        committed = self.model_runner.finalize_mtp_async(
            pending.completion,
            pending.decode_seqs,
            materialize_state=materialize_state,
            materialize_seqs=pending.seqs,
        )
        output_ids = getattr(pending.completion, "output_seq_ids", ())
        if output_ids and tuple(s.seq_id for s in pending.seqs) != output_ids:
            raise RuntimeError("MTP async completion no longer matches mixed batch")

        # These pages were protected because this just-completed batch still
        # referenced them.  Its completion event is now satisfied, so physical
        # release is safe before the allocator builds another batch.
        for seq in pending.release_after:
            self.model_runner.free(seq)
        pending.release_after.clear()

        ipc_package = self.scheduler.process_mtp_output_finalize(
            pending.deferred,
            committed,
            defer_frees=defer_frees_to,
        )
        if ipc_package is not None and self._polls_frontend():
            self.comm.send_output(ipc_package)

    def _drain_mtp_pending(self) -> None:
        """Drain MTP and materialize only the newest GPU state to the CPU."""
        while self._mtp_pending:
            pending = self._mtp_pending.popleft()
            self._collect_mtp_batch(
                pending,
                materialize_state=not self._mtp_pending,
            )

    def _filter_prefetched_freed(self, schedule_seqs):
        """Drop rows retired by the predecessor after this batch was scheduled."""
        if not any(getattr(s, "_overlap_freed", False) for s in schedule_seqs):
            return schedule_seqs
        kept = [s for s in schedule_seqs if not getattr(s, "_overlap_freed", False)]
        if self.scheduler.batch_running and self.scheduler.batch_running[-1] is schedule_seqs:
            if kept:
                self.scheduler.batch_running[-1] = kept
            else:
                # ``schedule_once`` uses the number of in-flight batches as a
                # hard PP-capacity gate. Leaving ``[]`` behind consumes that
                # slot forever (especially visible at pp_size=1), so later
                # requests can be admitted but never scheduled.
                self.scheduler.batch_running.pop()
        return kept

    def _refresh_prefetched_input(self) -> None:
        """Rebuild a batch scheduled against now-compacted MTP placeholders."""
        if self._prefetched_input is None:
            return
        seqs = self._filter_prefetched_freed(self._prefetched_input.seqs)
        if not seqs:
            self._prefetched_input = None
            return
        self._prefetched_input = self._make_scheduled_input(seqs)

    def _publish_mtp_relay_only(self) -> bool:
        """Publish relay-only x1 tokens required by ordinary decode."""
        relay_package = self.scheduler.materialize_mtp_relay_only()
        if relay_package is None:
            return False
        if self._polls_frontend():
            self.comm.send_output(relay_package)
        return True

    def _mtp_decode_prefix(self, input_data) -> list:
        """Return the legal leading decode partition, or an empty list.

        This is the single structural gate for pure and mixed MTP. Performance
        policy (batch-size crossover) remains in ``mtp_begin_iter``; execution
        policy (sync vs async) remains in :class:`_MtpBatchPlan`.
        """
        if (
            not self.model_runner.mtp_enabled
            or self._dp
            or input_data is None
            or not input_data.seqs
        ):
            return []
        num_prefills = int(getattr(input_data, "num_prefills", 0))
        if num_prefills == 0:
            return (
                list(input_data.seqs)
                if input_data.seqs[-1].computed_prompt
                else []
            )
        nd = int(getattr(input_data, "num_decodes", 0))
        decode = list(input_data.seqs[:nd])
        return decode if decode and all(s.computed_prompt for s in decode) else []

    def _plan_mtp_batch(self) -> _MtpBatchPlan:
        """Classify the prefetched tick and cache one speculation decision."""
        decode = self._mtp_decode_prefix(self._prefetched_input)
        speculate = self.model_runner.mtp_begin_iter(
            len(decode) if decode else None
        )
        if not speculate:
            return _MtpBatchPlan()
        seqs = self._prefetched_input.seqs
        greedy = all(
            s.top_k == 1
            and not (
                s.temperature > 1e-5
                and abs(s.temperature - 1.0) > 1e-5
            )
            for s in seqs
        )
        return _MtpBatchPlan(
            speculate=True,
            greedy=greedy,
            decode_ids=tuple(s.seq_id for s in decode),
        )

    def _partition_mtp_batch(self, seqs, decode_ids) -> _MtpBatch:
        """Apply the plan's stable request identities to a surviving batch."""
        decode_ids = set(decode_ids)
        decode = [seq for seq in seqs if seq.seq_id in decode_ids]
        prefill = [seq for seq in seqs if seq.seq_id not in decode_ids]
        # Keep the scheduler-owned list identity: if a predecessor retires a
        # row, _filter_prefetched_freed must compact batch_running in place with
        # the same row layout used by deferred completion indices.
        return _MtpBatch(seqs=seqs, decode=decode, prefill=prefill)

    def _take_mtp_batch(self, plan: _MtpBatchPlan) -> _MtpBatch:
        """Consume and compact the current prefetched MTP batch."""
        prefetched = self._prefetched_input
        if prefetched is None:
            return _MtpBatch([], [], [])
        schedule_seqs = prefetched.seqs
        self._prefetched_input = None
        schedule_seqs = self._filter_prefetched_freed(schedule_seqs)
        return self._partition_mtp_batch(schedule_seqs, plan.decode_ids)

    def _launch_mtp_step(self, batch: _MtpBatch, *, asynchronous):
        """Enqueue one pure or mixed MTP step on the graph capture stream."""
        default_stream = torch.cuda.current_stream()
        forward_stream = self.model_runner.forward_stream
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_stream(default_stream)
            if batch.prefill:
                result = self.model_runner.step_once_mtp_mixed(
                    batch.decode,
                    batch.prefill,
                    async_publish=asynchronous,
                )
            elif asynchronous:
                self.model_runner.prepare_input_mtp(batch.decode)
                result = self.model_runner.step_once_mtp_async()
            else:
                if self.model_runner.mtp_prep_eligible(batch.decode):
                    self.model_runner.prepare_input_mtp(batch.decode)
                else:
                    self.model_runner.prepare_input(batch.decode)
                result = self.model_runner.step_once()
        return result, default_stream, forward_stream

    def _run_mtp(self, plan: _MtpBatchPlan) -> None:
        """Execute one planned MTP state transition.

        Common lifecycle lives here: settle the ordinary overlap queue, choose
        whether GPU MTP state may remain in flight, compact stale rows, then
        hand the same ``(schedule, decode, prefill)`` partition to one of two
        commit policies. The launch itself is shared by both policies.
        """
        self._drain_pending()
        asynchronous = plan.use_async
        if not asynchronous:
            self._drain_mtp_pending()
            self._refresh_prefetched_input()
        if self._prefetched_input is None:
            self._build_prefetched_input()
            return

        batch = self._take_mtp_batch(plan)
        if not batch.seqs:
            if asynchronous:
                self._drain_mtp_pending()
            self._build_prefetched_input()
            return
        if not batch.decode:
            if asynchronous:
                self._drain_mtp_pending()
            # Every decode row retired under the prefetched batch, but its
            # prefill suffix is still real scheduled work. Let the ordinary
            # overlap path consume it next tick.
            self._prefetched_input = self._make_scheduled_input(batch.prefill)
            return

        if asynchronous:
            self._launch_mtp_async_batch(batch)
        else:
            self._commit_mtp_sync(batch)

    def _commit_mtp_sync(self, batch: _MtpBatch) -> None:
        """Launch and immediately commit a sampling/non-overlappable MTP step."""
        # A fused MTP step runs no decode forward, so building the decode
        # batch's per-token input arrays here would be immediately overwritten
        # by the draft/verify prep inside ``_mtp_decode``. Skip straight to the
        # batch bookkeeping when the fused fast path is guaranteed to be taken.
        # MTP graphs are captured on ``OverlapModelRunner.forward_stream`` (see
        # its ``capture_graph`` override), so replay and every metadata/state
        # update feeding that replay must run on the same stream. Previously
        # this synchronous bypass executed on the caller's default stream. That
        # violated the capture/replay stream contract and also raced SSM block
        # zero/free operations against the next verify, causing silent GDN state
        # drift and eventually illegal memory accesses on long generations.
        next_tokens, default_stream, forward_stream = self._launch_mtp_step(
            batch, asynchronous=False
        )
        # ``process_output`` may immediately free/zero accepted sequences on
        # the default stream. Make that work wait for the verify/state commit.
        default_stream.wait_stream(forward_stream)
        if next_tokens is not None:
            self.scheduler.add_next_tokens(next_tokens, self.model_runner._last_logprobs)
            ipc_package = self.scheduler.process_output()
            if ipc_package is not None and self._polls_frontend():
                self.comm.send_output(ipc_package)
        # Build the next iter AFTER finalize so any max_len/eos seqs are freed.
        self._build_prefetched_input()

    def _launch_mtp_async_batch(self, batch: _MtpBatch) -> None:
        """Launch one batch with gLLM's launch-current/collect-previous cadence.

        The successor consumes GPU-resident relay/context/acceptance state. CPU
        placeholder compaction and user output for N happen only after N+1 has
        been enqueued, exactly like the ordinary FutureMap overlap pipeline.
        """
        decode_ids = {seq.seq_id for seq in batch.decode}

        prev = self._mtp_pending[0] if self._mtp_pending else None
        can_chain = bool(
            prev is not None
            and self.model_runner.mtp_async_can_chain(batch.decode)
        )
        if prev is not None and not can_chain:
            # Membership/order changes are explicit drain boundaries. The batch
            # was already selected optimistically, but after compaction its row
            # set remains legal; rebuild only its input view below.
            self._drain_mtp_pending()
            seqs = self._filter_prefetched_freed(batch.seqs)
            if not seqs:
                self._build_prefetched_input()
                return
            batch = self._partition_mtp_batch(seqs, decode_ids)
            if not batch.decode:
                self._prefetched_input = self._make_scheduled_input(batch.prefill)
                return

        default_stream = torch.cuda.current_stream()
        forward_stream = self.model_runner.forward_stream
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_stream(default_stream)
            if can_chain:
                self.model_runner.mtp_async_remap(batch.decode)
        with torch.profiler.record_function("gllm::mtp_launch_current"):
            completion, _, _ = self._launch_mtp_step(
                batch, asynchronous=True
            )
        current = _PendingMtpBatch(
            completion=completion,
            seqs=batch.seqs,
            decode_seqs=batch.decode,
        )

        # The launch above is the overlap boundary. Now collect N while N+1 is
        # executing. Finalize before reserving N+1's placeholders so compacting
        # N cannot invalidate absolute positions owned by N+1.
        if prev is not None and can_chain:
            assert self._mtp_pending.popleft() is prev
            with torch.profiler.record_function("gllm::mtp_collect_previous"):
                self._collect_mtp_batch(
                    prev,
                    materialize_state=False,
                    defer_frees_to=current.release_after,
                )

        current.deferred = self.scheduler.process_mtp_output_deferred(
            decode_rows=len(batch.decode),
            width=1 + self.model_runner._mtp_k,
        )
        self._mtp_pending.append(current)

        # CPU scheduling/page allocation for the next tick overlaps the current
        # target verify and its async completion copy.
        self._build_prefetched_input()

    def run_pp0(self):
        """Per-iter loop run by every PP-0 TP rank.

        Ordering is launch-first, collect-later -- but every TP rank executes
        it independently. Determinism + identical inputs keeps every rank's
        queue / future-map / scheduler in lockstep without any inter-TP zmq
        traffic on the critical path.

        DP-attention + EP (``self._dp``) wraps the same launch->collect pipeline
        in a per-iter cross-DP barrier (a tiny ``dp_all_gather_meta``
        all-gather) so the world stays lockstep through the MoE collectives:

        * every group agrees whether *anyone* has work (else nobody launches --
          a lone MoE collective would hang -- and the pipeline drains);
        * every group agrees whether the whole world can take the CUDA-graph
          path this step (only when *all* groups are pure decode / idle) and on
          the common bucket, so the captured global MoE batch matches;
        * an idle group rides along with a 1-token dummy so its MoE collective
          still joins; its sampled token is discarded.

        The agreed counts are published via ``set_dp_forward_counts`` right
        before dispatch (consumed synchronously as the eager forward is enqueued,
        or baked at capture time for graphs), so they never race the in-flight
        batch.
        """
        self.check_abort_seqs()
        # ``recv_ipc_package`` also drives the disagg coordinator (TP0) and
        # applies its fanned-out ADMIT / EMB_READY events on every column.
        self.recv_ipc_package()

        # Bootstrap on the first iter, otherwise this is a no-op
        # (the previous iter's tail already built next-iter's input).
        if self._prefetched_input is None:
            self._build_prefetched_input()

        # One plan owns the pure/mixed and sync/async decision for this tick.
        # DP+EP and batch-size crossover policy are encapsulated by the planner;
        # execution below only dispatches the selected state transition.
        mtp_plan = self._plan_mtp_batch()
        if self._mtp_pending and not mtp_plan.speculate:
            # Mixed/prefill/plain-decode metadata is CPU-derived, so leaving a
            # stable MTP cohort first materializes its newest GPU checkpoint and
            # compacts the optimistic token placeholders. A completed mixed
            # prefill may then own x1 only in the GPU relay; publish it before
            # rebuilding ordinary input, otherwise token and position shapes
            # disagree during an MTP batch-size crossover.
            self._drain_mtp_pending()
            self._publish_mtp_relay_only()
            self._refresh_prefetched_input()
            if self._prefetched_input is None:
                self._build_prefetched_input()
                return
            mtp_plan = self._plan_mtp_batch()
        if mtp_plan.speculate:
            self._run_mtp(mtp_plan)
            return

        # A completed mixed prefill may still own an x1 token solely in the
        # materialized MTP relay. Ordinary decode cannot consume that protocol:
        # publish x1 to token_ids/frontend as its one uncached input, then
        # rebuild any batch that was prepared against the relay-only shape.
        if self._publish_mtp_relay_only():
            self._refresh_prefetched_input()
            if self._prefetched_input is None:
                self._build_prefetched_input()
                return

        input_data = self._prefetched_input
        is_dummy = False
        dp_padded_size = None

        if self._dp:
            # Cross-DP barrier: agree on who runs + the graph decision.
            if input_data is not None:
                real_ntok = int(input_data.tokens_cpu.shape[0])
                is_decode = bool(input_data.seqs[-1].computed_prompt)
            else:
                real_ntok = 0
                is_decode = True  # idle groups don't veto the graph path
            counts, decode_flags = dp_all_gather_meta(real_ntok, is_decode)
            if sum(counts) == 0:
                # Nobody has work: skip the forward in unison, drain the pipe.
                self._drain_pending()
                return
            if input_data is None:
                input_data = self._build_dummy_input(1)
                is_dummy = True
            fwd_counts = [c if c > 0 else 1 for c in counts]
            if all(bool(d) for d in decode_flags):
                dp_padded_size = self.model_runner.dp_select_bucket(max(fwd_counts))
            set_dp_forward_counts(
                [dp_padded_size] * self.dp_size
                if dp_padded_size is not None
                else fwd_counts
            )

        pending_before = len(self._gpu_pending)

        if input_data is not None:
            # Keep the InputData alive in ``_gpu_pending`` until the batch
            # finishes -- ``prep_stream`` is still DMA'ing from its CPU tensors.
            self._prefetched_input = None
            try:
                copy_done, batch_size, future_slot_ids, buf_idx, lp_k = (
                    self._launch_batch(input_data, dp_padded_size=dp_padded_size)
                )
            finally:
                if self._dp:
                    set_dp_forward_counts(None)
            deferred = (
                None
                if is_dummy
                else self.scheduler.process_output_deferred(future_slot_ids)
            )
            self._gpu_pending.append(
                (copy_done, batch_size, buf_idx, deferred, input_data, is_dummy, lp_k)
            )

        if pending_before > 0:
            self._collect_batch(self._gpu_pending.popleft())

        # Build the next iter AFTER finalize so any max_len/eos seqs from this
        # iter are already freed when we reschedule.
        self._build_prefetched_input()


def run_overlap_worker(worker: OverlapWorker):
    """Tight per-iter loop for the overlap path (PP=1 only)."""
    try:
        worker.init()
        # PP=1 means every rank in the world is on PP-0; the unified
        # ``run_pp0`` body covers driver and follower, TP and DP+EP alike.
        while True:
            worker.run_pp0()
    except KeyboardInterrupt:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
