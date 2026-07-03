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
to the (also-refactored) :class:`gllm.worker.Worker` path.
"""

from collections import deque

from gllm.dist_utils import (
    dp_all_gather_meta,
    is_dp_attn,
    set_dp_forward_counts,
)
from gllm.input_data import InputData
from gllm.model_runner import OverlapModelRunner
from gllm.scheduler import OverlapScheduler
from gllm.worker import Worker


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
        self.scheduler.consume_pending_follower_frees()
        schedule_seqs = self.scheduler.schedule_once()
        if len(schedule_seqs) == 0:
            self._prefetched_input = None
            return
        next_input = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        next_input.cal_input(schedule_seqs)
        self._prefetched_input = next_input

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
        seqs = self.model_runner.create_dummy_seqs(size)
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
        ``run_batch_async``) and updates its local Sequence state / frees pages;
        only the frontend poller (rank 0, or each DP group's ``tp_rank == 0``)
        forwards the resulting ``IPCPackage``. ``is_dummy`` batches (idle DP
        groups) and empty ``deferred`` carry no output and are skipped.
        """
        copy_done, batch_size, buf_idx, deferred, _input_data, is_dummy = entry
        copy_done.synchronize()
        if is_dummy or deferred is None:
            return
        tokens = self.model_runner._next_tokens_bufs[buf_idx][:batch_size].tolist()
        ipc_package = self.scheduler.process_output_finalize(deferred, tokens)
        if ipc_package is not None and self._polls_frontend():
            self.comm.send_output(ipc_package)

    def _drain_pending(self) -> None:
        while self._gpu_pending:
            self._collect_batch(self._gpu_pending.popleft())

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
                copy_done, batch_size, future_slot_ids, buf_idx = (
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
                (copy_done, batch_size, buf_idx, deferred, input_data, is_dummy)
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
