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

    def _launch_batch(self, input_data: InputData):
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
        return self.model_runner.run_batch_async()

    def _collect_batch(self, pending_entry, deferred_seqs):
        """Wait for the batch's CPU buffer and finalize seq state.

        Every PP-0 TP rank reads its own ``_next_tokens_bufs`` slot --
        each rank does its own D2H copy from the broadcast tokens
        tensor inside ``run_batch_async``. ``deferred_seqs`` is the
        per-rank metadata produced by ``process_output_deferred`` and
        is used by every rank to update its local Sequence state /
        free pages. Only rank 0 sends the resulting ``IPCPackage`` to
        the frontend.
        """
        copy_done, batch_size, buf_idx, _input_data = pending_entry
        copy_done.synchronize()
        tokens = self.model_runner._next_tokens_bufs[buf_idx][
            :batch_size
        ].tolist()
        self._finalize_deferred(deferred_seqs, tokens)

    def _finalize_deferred(self, deferred_seqs, tokens):
        if deferred_seqs is None or tokens is None:
            return
        ipc_package = self.scheduler.process_output_finalize(deferred_seqs, tokens)
        if ipc_package is not None and self.rank == 0:
            self.comm.send_output(ipc_package)

    # ------------------------------------------------------------------
    # Driver loop -- now identical for every PP-0 TP rank
    # ------------------------------------------------------------------

    def run_pp0(self):
        """Per-iter loop run by every PP-0 TP rank under the new design.

        Ordering is the same as the previous ``run_driver`` for
        rank 0 -- launch first, collect later -- but every TP rank
        executes it independently. Determinism + identical inputs
        keeps every rank's queue / future-map / scheduler in lockstep
        without any inter-TP zmq traffic on the critical path.
        """
        self.check_abort_seqs()
        self.recv_ipc_package()
        self._disagg_poll()

        # Bootstrap on the first iter, otherwise this is a no-op
        # (the previous iter's tail already built next-iter's input).
        if self._prefetched_input is None:
            self._build_prefetched_input()

        pending_before = len(self._gpu_pending)

        if self._prefetched_input is not None:
            # Keep the InputData alive in ``_gpu_pending`` until the
            # batch finishes -- ``prep_stream`` is still DMA'ing from
            # its CPU tensors when we reach this line.
            input_data = self._prefetched_input
            self._prefetched_input = None
            copy_done, batch_size, future_slot_ids, buf_idx = self._launch_batch(
                input_data
            )
            deferred = self.scheduler.process_output_deferred(future_slot_ids)
            self._gpu_pending.append(
                (copy_done, batch_size, buf_idx, deferred, input_data)
            )

        if pending_before > 0:
            copy_done, batch_size, buf_idx, deferred, input_data = (
                self._gpu_pending.popleft()
            )
            self._collect_batch(
                (copy_done, batch_size, buf_idx, input_data), deferred
            )

        # Build + (no-op zmq send) the next iter AFTER finalize so
        # any max_len/eos seqs from this iter are already freed when
        # we reschedule.
        self._build_prefetched_input()


def run_overlap_worker(worker: OverlapWorker):
    """Tight per-iter loop for the overlap path (PP=1 only)."""
    try:
        worker.init()
        # PP=1 means every rank in the world is on PP-0; the unified
        # ``run_pp0`` body covers driver and follower alike.
        while True:
            worker.run_pp0()
    except KeyboardInterrupt:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
