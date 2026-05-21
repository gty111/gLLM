"""Worker driver loops for overlap scheduling (TP only, pp_size must be 1)."""

from collections import deque
from logger import logger

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
        # Ship ``schedule_seqs`` to the TP followers (same PP stage) BEFORE
        # building this batch's ``InputData`` locally. ``cal_input`` is pure
        # numpy/torch CPU work (~ms-level for typical decode batches) that
        # touches only read-only seq attributes, so it overlaps cleanly with
        # the pickling + zmq send happening in the background sender thread
        # and with the TP follower's own ``cal_input`` once it receives.
        #
        # Previously we sent *after* ``cal_input`` returned, which meant
        # rank-1 started its CPU prep ~``cal_input`` later than rank-0 every
        # batch. That lag propagated all the way to the first NCCL collective
        # of each forward and surfaced as a huge stall on rank-0's first
        # AllReduce per batch (profiler: rank-0 AR p99 ~4ms, mean 87us; rank-1
        # AR p99 ~80us, mean 11us). Sending first collapses that lag and lets
        # the two ranks enter every collective ~simultaneously.
        if get_world_size() > 1:
            self.comm.send_schedule_seqs(schedule_seqs, None, True)
        next_input = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        next_input.cal_input(schedule_seqs)
        if get_world_size() > 1:
            # Second send carries ``mrope_positions_cpu``, which is only
            # populated by ``cal_input`` (and only used by subsequent PP
            # stages for multimodal models). With pp_size=1 the recipient
            # list is empty, so this is a no-op; with pp_size>1 it must
            # remain after ``cal_input``.
            self.comm.send_schedule_seqs(
                schedule_seqs, next_input.mrope_positions_cpu, False
            )
        self._prefetched_input = next_input

    def _recv_prefetched_input(self):
        recv_data = self.comm.recv_schedule_seqs()
        if recv_data is None:
            return None
        if len(recv_data) == 3:
            seqs, mrope_positions, cmd_code = recv_data
            profile_session_dir = None
        elif len(recv_data) == 4:
            seqs, mrope_positions, cmd_code, profile_session_dir = recv_data
        else:
            raise ValueError(f"Unexpected schedule payload: {recv_data!r}")

        if cmd_code != 0:
            self._apply_control_cmd(cmd_code, profile_session_dir)
        if len(seqs) == 0:
            return None

        input_data = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        input_data.cal_input(seqs)
        if mrope_positions is not None:
            input_data.set_mrope_position(mrope_positions)
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
        """TP followers (pp_rank == 0 only)."""
        pending_before = len(self._gpu_pending)

        if pending_before > 0:
            copy_done, batch_size, buf_idx, _deferred, input_data = (
                self._gpu_pending.popleft()
            )
            self._collect_batch((copy_done, batch_size, buf_idx, input_data))

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


def run_overlap_worker(worker: OverlapWorker):
    run_worker(worker)
