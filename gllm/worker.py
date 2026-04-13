import logging
import traceback
from collections import deque
from typing import Union

import torch
import torch.distributed as dist
from logger import logger

from gllm.comm import IPCPackage, zmqComm
from gllm.dist_utils import (
    get_last_pp_rank,
    get_next_pp_rank,
    get_pp_size,
    get_rank,
    get_world_size,
    init_dist,
    is_last_pp_rank,
    is_output_rank,
    recv_pp_data,
    send_pp_data,
)
from gllm.input_data import InputData
from gllm.model_runner import ModelRunner, AsyncModelRunner
from gllm.profiler_mixin import TorchProfilerMixin
from gllm.scheduler import Scheduler, AsyncScheduler


class Worker(TorchProfilerMixin):

    def __init__(
        self,
        model_runner: Union[ModelRunner,AsyncModelRunner],
        local_rank,
        pp_rank,
        tp_rank,
        pp_size,
        tp_size,
        use_ep,
        master_addr,
        master_port,
        comm: zmqComm,
        mp_alive,
        mp_load_progress,
        assigned_layers,
        schedule_method,
        scheduler_cls: Union[type[Scheduler],type[AsyncScheduler]]=Scheduler,
    ):
        self.model_runner = model_runner
        self.local_rank = local_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.use_ep = use_ep
        self.master_addr = master_addr
        self.master_port = master_port
        self.comm = comm
        self.mp_alive = mp_alive
        self.mp_load_progress = mp_load_progress
        self.assigned_layers = assigned_layers
        self.schedule_method = schedule_method
        self.use_mla = model_runner.model_loader.use_mla
        self.scheduler_cls = scheduler_cls
        self.init_profiler_state()

    def init_logger(self):
        tp_ep_log = "TP" if not self.use_ep or self.tp_size == 1 else "TP/EP"
        formatter = logging.Formatter(
            f"[%(asctime)s %(filename)s:%(lineno)d Worker{self.pp_rank*self.tp_size+self.tp_rank} "
            f"PP{self.pp_rank} {tp_ep_log}{self.tp_rank}] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        for handler in logger.handlers:
            handler.setFormatter(formatter)

    def init(self):
        self.init_logger()
        if self.pp_size > 1 or self.tp_size > 1:
            init_dist(
                self.pp_size,
                self.tp_size,
                self.use_ep,
                self.local_rank,
                self.pp_rank,
                self.tp_rank,
                self.master_addr,
                self.master_port,
                self.assigned_layers,
            )
        self.rank = get_rank()
        torch.cuda.set_device(f"cuda:{self.local_rank}")

        self.comm.init()

        self.model_runner.init(self.mp_load_progress)
        self.hidden_size = self.model_runner.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual

        if self.rank == 0:
            self.scheduler = self.scheduler_cls(
                self.pp_size,
                self.model_runner,
                self.schedule_method,
            )
        else:
            # Input data for each rank except 0
            self.schedule_queue = deque()

        self.mp_alive[self.local_rank] = 1

        logger.info(f"Initialization complete")

    # driver worker => other workers
    def recv_schedule_seqs(self):
        recv_data = self.comm.recv_schedule_seqs()
        if recv_data is not None:
            profile_session_dir = None
            cmd_code = 0
            if len(recv_data) == 3:
                seqs, mrope_positions, cmd_code = recv_data
            elif len(recv_data) == 4:
                seqs, mrope_positions, cmd_code, profile_session_dir = recv_data
            else:
                raise Exception(f"Fail to parse {recv_data = }")

            if cmd_code != 0:
                self._apply_control_cmd(cmd_code, profile_session_dir)

            if len(seqs) == 0:
                return

            input_data = InputData(
                use_buffer=False,
                memory_manager=self.model_runner.memory_manager,
                max_seq_length=self.model_runner.model_max_length,
            )
            input_data.cal_input(seqs)
            if mrope_positions is not None:
                input_data.set_mrope_position(mrope_positions)
            self.schedule_queue.append(input_data)

    def forward_pp(self):
        if len(self.schedule_queue) != 0:
            input_data: InputData = self.schedule_queue.popleft()
            # pp last rank => pp next rank
            recv_pp_data(
                get_last_pp_rank(),
                input_data.tokens_cpu.shape[0],
                self.model_runner.input_hidden_states,
                self.model_runner.input_residual,
                self.ret_residual,
            )
            self.model_runner.prepare_input(input_data=input_data)
            output = self.model_runner.step_once()
            if is_output_rank():
                self.comm.send_tokens(output)
            elif not is_last_pp_rank():
                send_pp_data(output, get_next_pp_rank())

    def recv_ipc_package(self):
        # To avoid request accumulation, we fetch all packages in comm
        cum_ipc_package = IPCPackage([])
        while True:
            ipc_package: IPCPackage = self.comm.recv_ipc_package()
            if ipc_package is not None:
                cum_ipc_package.schedule_lists.extend(ipc_package.schedule_lists)
                cum_ipc_package.abort_ids.extend(ipc_package.abort_ids)
                if ipc_package.log is not None:
                    self.scheduler.set_log(ipc_package.log)
                if ipc_package.control_cmd is not None:
                    self.sync_control_cmd(ipc_package.control_cmd)
            else:
                break
        if (
            len(cum_ipc_package.schedule_lists) != 0
            or len(cum_ipc_package.abort_ids) != 0
        ):
            self.scheduler.add_new_requests(cum_ipc_package.schedule_lists)
            self.scheduler.add_abort_ids(cum_ipc_package.abort_ids)

    def recv_next_tokens(self):
        if self.pp_size != 1:  # recv tokens from last rank
            next_tokens = self.comm.recv_tokens()
            if next_tokens is not None:
                self.scheduler.add_next_tokens(next_tokens)

    def check_abort_seqs(self):
        ipc_package = self.scheduler.check_abort_seqs()
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def process_output(self):
        ipc_package = self.scheduler.process_output()
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def forward_tp(self):
        if len(self.schedule_queue) != 0:
            input_data: InputData = self.schedule_queue.popleft()
            self.model_runner.prepare_input(input_data=input_data)
            output = self.model_runner.step_once()
            if get_pp_size() != 1:
                send_pp_data(output, get_next_pp_rank())

    def schedule_forward(self):
        schedule_seqs = self.scheduler.schedule_once()
        if len(schedule_seqs) != 0:
            if get_world_size() > 1:
                self.comm.send_schedule_seqs(schedule_seqs, None, True)
            self.model_runner.prepare_input(schedule_seqs)
            if get_world_size() > 1:
                self.comm.send_schedule_seqs(
                    schedule_seqs,
                    self.model_runner.input_data.mrope_positions_cpu,
                    False,
                )
            output = self.model_runner.step_once()

            if not is_output_rank():
                send_pp_data(output, get_next_pp_rank())
            else:
                self.scheduler.add_next_tokens(output)

    def run_driver(self):
        self.check_abort_seqs()
        self.recv_ipc_package()
        self.recv_next_tokens()
        self.schedule_forward()
        self.process_output()

    def run_first_tp(self):
        self.recv_schedule_seqs()
        self.forward_tp()

    def run_other(self):
        self.recv_schedule_seqs()
        self.forward_pp()

    def handle_keyboardInterrupt(self):
        self.mp_alive[self.local_rank] = -1
        logger.info(f"Exit")
        if dist.is_initialized():
            dist.destroy_process_group()

    def handle_exception(self, e):
        logger.error(e)
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.mp_alive[self.local_rank] = -1
        
class AsyncWorker(Worker):
    def __init__(self, *args, **kwargs):
        kwargs["scheduler_cls"] = AsyncScheduler
        super().__init__(*args, **kwargs)
        
    def init(self):
        super().init()
        
        # Async scheduling state for driver (rank 0):
        self._prefetched_input = None
        self._prefetched_seqs = None
        # result_queue: FIFO of (event, num_cal_tokens, future_indices, deferred_seqs)
        # Each entry pairs a batch's GPU event with the seqs that need
        # finalization once D2H completes.
        self._result_queue = deque()
        # FutureMap for GPU circular buffer token ID buffering
        self.model_runner.init_async(num_prefill_chunks=256)
        self.future_map = self.model_runner.future_map
        
    def schedule_forward_async(self):
        """Async-scheduling version using FutureMap GPU circular buffer.

          Step 1 — Schedule next batch (CPU only, no GPU sync)
          Step 2 — Launch current batch on GPU (non-blocking) + deferred process
          Step 3 — Collect + finalize the batch from TWO iterations ago
                   (its D2H completed during last iteration's GPU work)

        result_queue depth is 1 in steady state: each iteration pushes one
        entry (Step 2) and pops one entry (Step 3, from a previous iteration).
        We never pop the entry we just pushed — only entries already in the
        queue before Step 2.

        Timeline:
          Iter N:  [schedule(N+1)] [launch(N+1)] [collect(N-1) + finalize(N-1)]
                                    ↑ GPU starts     ↑ D2H(N-1) done long ago
                                                     ← overlaps with GPU(N+1)
        """
        # --- Step 1: Receive new requests, schedule next batch (CPU only) ---
        self.check_abort_seqs()
        self.recv_ipc_package()
        self.recv_next_tokens()

        self._build_prefetched_input()

        # --- Step 2: Launch current batch on GPU (non-blocking) ---
        # Remember queue length BEFORE pushing, so Step 3 only pops old entries.
        num_pending_before = len(self._result_queue)

        if self._prefetched_input is not None:
            # Ensure forward_stream from the last batch has finished reading
            # shared GPU buffers before we overwrite them with H2D copies.
            self.model_runner.sync_before_next_prepare()
            self.model_runner.prepare_input(input_data=self._prefetched_input)
            event, batch_size, num_cal_tokens, future_indices, buf_idx = (
                self.model_runner.run_batch_async()
            )
            self._prefetched_input = None
            self._prefetched_seqs = None

            # Deferred process: create placeholder tokens, pair with event
            deferred_seqs = self.scheduler.process_output_deferred(
                future_indices
            )

            self._result_queue.append(
                (event, batch_size, num_cal_tokens, future_indices, buf_idx, deferred_seqs)
            )

        # --- Step 3: Collect + finalize a PREVIOUS batch ---
        # Only process entries that were in the queue before Step 2.
        # This ensures we never synchronize on the batch we just launched.
        if num_pending_before > 0:
            event, batch_size, num_cal_tokens, future_indices, buf_idx, deferred_seqs = (
                self._result_queue.popleft()
            )
            output = self.model_runner.step_collect_async(event, batch_size, num_cal_tokens, buf_idx)
            if not is_output_rank():
                send_pp_data(output, get_next_pp_rank())
            else:
                if deferred_seqs is not None:
                    ipc_package = self.scheduler.process_output_finalize(
                        deferred_seqs, output
                    )
                    if ipc_package is not None:
                        self.comm.send_output(ipc_package)
    
    def _build_prefetched_input(self):
        """Schedule and build InputData for the next batch (CPU side only).

        Called while the GPU is executing the current forward pass, so that CPU
        input preparation overlaps with GPU execution. The result is stored in
        self._prefetched_input / self._prefetched_seqs.

        For VL models, mm_prepare_inputs is called here (not in prepare_input)
        so that the vision encoder's GPU work runs on the default stream in
        parallel with the current batch's forward on forward_stream.  Any
        implicit cudaStreamSynchronize from .tolist()/.cpu() inside the vision
        encoder only blocks on default-stream work, not on forward_stream,
        avoiding a stall before run_batch_async.
        """
        schedule_seqs = self.scheduler.schedule_once()
        if len(schedule_seqs) != 0:
            next_input = InputData(
                use_buffer=False,
                memory_manager=self.model_runner.memory_manager,
                max_seq_length=self.model_runner.model_max_length,
            )
            next_input.cal_input(schedule_seqs)

            # Pre-compute VL embeddings on the default stream while the
            # current batch runs on forward_stream.
            if self.model_runner.use_mm:
                input_embeddings, mrope_positions = (
                    self.model_runner.mm_prepare_inputs(next_input.seqs)
                )
                # Stash on the prebuilt InputData so prepare_input can
                # skip mm_prepare_inputs and directly use these.
                next_input._mm_embeddings = input_embeddings
                next_input._mm_positions = mrope_positions
                # Also update mrope_positions_cpu so send_schedule_seqs
                # sends correct positions to other workers.
                next_input.mrope_positions_cpu = mrope_positions

            if get_world_size() > 1:
                self.comm.send_schedule_seqs(schedule_seqs, None, True)
                self.comm.send_schedule_seqs(
                    schedule_seqs,
                    next_input.mrope_positions_cpu,
                    False,
                )
            self._prefetched_input = next_input
            self._prefetched_seqs = schedule_seqs
        else:
            self._prefetched_input = None
            self._prefetched_seqs = None
    
    def run_driver(self):
        self.schedule_forward_async()


def run_worker(worker: Worker):
    try:
        worker.init()
        while True:
            if worker.rank == 0:
                worker.run_driver()
            elif worker.pp_rank == 0:
                worker.run_first_tp()
            else:
                worker.run_other()
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
