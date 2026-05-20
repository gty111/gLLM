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
        next_input = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        next_input.cal_input(schedule_seqs)
        if get_world_size() > 1:
            self.comm.send_schedule_seqs(schedule_seqs, None, True)
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
        # Split input prep into a CPU phase (safe to run while the previous
        # batch is still on the GPU) and a GPU/H2D phase (must wait until the
        # previous forward has released the shared input buffers). The CPU
        # phase covers attribute copies, multimodal position calc, and
        # whatever image processing was deferred from the prefetch step;
        # those several milliseconds of Python work now overlap with the
        # tail of the previous forward instead of serializing behind it.
        self.model_runner.prepare_input_cpu(input_data)
        self.model_runner.sync_before_next_prepare()
        self.model_runner.prepare_input_gpu()
        return self.model_runner.run_batch_async()

    def _collect_batch(self, pending_entry, deferred_seqs=None):
        copy_done, batch_size, buf_idx = pending_entry
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
            copy_done, batch_size, future_slot_ids, buf_idx = self._launch_batch(
                self._prefetched_input
            )
            self._prefetched_input = None
            deferred = self.scheduler.process_output_deferred(future_slot_ids)
            self._gpu_pending.append((copy_done, batch_size, buf_idx, deferred))

        if pending_before > 0:
            copy_done, batch_size, buf_idx, deferred = self._gpu_pending.popleft()
            self._collect_batch((copy_done, batch_size, buf_idx), deferred)

    def run_first_tp(self):
        """TP followers (pp_rank == 0 only)."""
        pending_before = len(self._gpu_pending)

        if pending_before > 0:
            copy_done, batch_size, buf_idx, _deferred = self._gpu_pending.popleft()
            self._collect_batch((copy_done, batch_size, buf_idx))

        next_input = self._recv_prefetched_input()
        if next_input is not None:
            copy_done, batch_size, _future_slot_ids, buf_idx = self._launch_batch(
                next_input
            )
            self._gpu_pending.append((copy_done, batch_size, buf_idx, None))


def run_overlap_worker(worker: OverlapWorker):
    run_worker(worker)
