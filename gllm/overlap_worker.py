"""Worker driver loops for overlap scheduling (TP only, pp_size must be 1)."""

from collections import deque
from logger import logger

from gllm.dist_schedule import SchedulePayload
from gllm.dist_utils import (
    get_tp_size,
    get_world_size,
    is_output_rank,
)
from gllm.input_data import InputData
from gllm.model_runner import OverlapModelRunner
from gllm.scheduler import OverlapScheduler
from gllm.worker import Worker, run_worker


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
        super().init()
        if self.rank == 0:
            self.scheduler = OverlapScheduler(
                self.pp_size,
                self.model_runner,
                self.schedule_method,
            )
        self._prefetched_input = None
        self._gpu_pending = deque()

    def _build_prefetched_input(self) -> None:
        schedule_seqs = self.scheduler.schedule_once()
        if len(schedule_seqs) == 0:
            self._prefetched_input = None
            return
        # Ship the delta-style :class:`SchedulePayload` to TP followers
        # (same PP stage) BEFORE building this batch's ``InputData``
        # locally. ``cal_input`` is pure numpy/torch CPU work that
        # touches only read-only seq attributes, so it overlaps cleanly
        # with the persistent zmq sender thread pickling the payload
        # and with the TP follower's own ``cal_input`` once it
        # receives. Sending first was critical for the previous fix
        # (rank-1 入图同步, see tp2-perf-fix.md) -- the delta refactor
        # cuts the wire payload by another ~30x (no more per-iter
        # ``page_table`` / sampling-params / ``mm_contents`` re-send),
        # which should shrink the residual position-0 AR stall further.
        #
        # pp_size > 1 is rejected at OverlapWorker.__init__ so we
        # don't need a second send for PP-other here.
        if get_world_size() > 1:
            payload = self._build_schedule_payload(schedule_seqs)
            if payload is not None:
                self.comm.send_schedule_payload(payload, is_first_pp=True)
        next_input = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        next_input.cal_input(schedule_seqs)
        self._prefetched_input = next_input

    def _recv_prefetched_input(self):
        payload = self.comm.recv_schedule_payload()
        if payload is None:
            return None

        if payload.control_cmd != 0:
            self._apply_control_cmd(payload.control_cmd, payload.control_data)

        seqs = self.follower_store.apply_payload(payload)

        # Same ordering rationale as ``Worker.recv_schedule_payload``:
        # apply frees *after* materializing the batch. With overlap
        # scheduling the freed seq cannot be in this batch (the
        # scheduler removes it from ``seqs_to_decode`` /
        # ``seqs_to_prefill`` before scheduling), so this is just
        # defensive ordering for the edge case where a payload
        # happens to free + reschedule the same id.
        if payload.frees:
            for sid in payload.frees:
                self.model_runner.free_follower_state(sid)

        if not seqs:
            return None

        input_data = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        input_data.cal_input(seqs)
        if payload.mrope_positions is not None:
            input_data.set_mrope_position(payload.mrope_positions)
        return input_data

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

    def _collect_batch(self, pending_entry, deferred_seqs=None):
        copy_done, batch_size, buf_idx, _input_data = pending_entry
        copy_done.synchronize()
        if is_output_rank():
            tokens = self.model_runner._next_tokens_bufs[buf_idx][
                :batch_size
            ].tolist()
            if self.rank == 0:
                self._finalize_deferred(deferred_seqs, tokens)

    def _finalize_deferred(self, deferred_seqs, tokens):
        if deferred_seqs is None or tokens is None:
            return
        ipc_package = self.scheduler.process_output_finalize(deferred_seqs, tokens)
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def run_driver(self):
        """Rank-0 driver: schedule next batch while GPU runs the previous one."""
        self.check_abort_seqs()
        self.recv_ipc_package()

        self._build_prefetched_input()
        pending_before = len(self._gpu_pending)

        if self._prefetched_input is not None:
            # ``input_data`` is kept alive in ``_gpu_pending`` until the batch
            # finishes on the GPU. Without this, the only Python reference to
            # this batch's CPU input tensors (``tokens_cpu``, ``slot_mapping_cpu``
            # etc.) lives on ``model_runner.input_data`` and would be
            # overwritten by the *next* iteration's ``prepare_input_cpu`` --
            # potentially while ``prep_stream`` is still DMA'ing from those
            # buffers. Holding the original ``InputData`` here makes the
            # cross-batch lifetime explicit and trivially correct.
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

    def run_first_tp(self):
        """TP followers (pp_rank == 0 only).

        Ordering mirrors ``run_driver``: **launch first, collect last**.
        The earlier version did the opposite (``_collect_batch`` then
        ``_launch_batch``), which made rank-1 fully serial -- the CPU
        sync inside ``copy_done.synchronize()`` for batch N would block
        the host thread until batch N's forward+broadcast finished on
        GPU before rank-1 even started receiving batch N+1's schedule.
        Rank-0 meanwhile pipelined: it had already launched batch N+1
        well before that sync. The result was that rank-1 entered every
        forward ~2 ms after rank-0, and rank-0 spent that 2 ms spinning
        inside the *first* AR kernel of each forward waiting for rank-1
        to arrive. Profiler confirmed: position-0 AR mean was 1987 us,
        position-1+ was 4-7 us; reordering collapses that to ~5 us flat
        and shaves the bulk of the rank-0 AR p99 tail.

        Queue semantics: we capture ``pending_before`` *before* the new
        launch's ``append`` so the subsequent ``popleft`` still returns
        the older batch (N), never the just-launched one (N+1).
        """
        pending_before = len(self._gpu_pending)

        next_input = self._recv_prefetched_input()
        if next_input is not None:
            copy_done, batch_size, _future_slot_ids, buf_idx = self._launch_batch(
                next_input
            )
            # Keep ``next_input`` alive in the pending queue (see comment in
            # ``run_driver``) so its CPU input tensors outlive the prep_stream
            # DMA on the follower path too.
            self._gpu_pending.append(
                (copy_done, batch_size, buf_idx, None, next_input)
            )

        if pending_before > 0:
            copy_done, batch_size, buf_idx, _deferred, input_data = (
                self._gpu_pending.popleft()
            )
            self._collect_batch((copy_done, batch_size, buf_idx, input_data))


def run_overlap_worker(worker: OverlapWorker):
    run_worker(worker)
