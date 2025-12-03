import torch
import torch.distributed as dist
import traceback
import logging

from collections import deque
from logger import logger

from gllm.input_data import InputData
from gllm.model_runner import ModelRunner
from gllm.comm import zmqComm, IPCPackage
from gllm.worker_scheduler import WorkerScheduler
from gllm.dist_utils import (init_dist, send_pp_data, recv_pp_data, 
                             get_rank, get_world_size, is_last_pp_rank,
                             get_pp_size, get_next_pp_rank, get_last_pp_rank,
                             is_output_rank)

# Used with PipeAsyncLLM
class Worker:

    def __init__(self, model_runner: ModelRunner, local_rank, pp_rank, tp_rank, 
                 pp_size, tp_size, use_ep, master_addr, master_port, comm: zmqComm, mp_alive,
                 mp_load_progress, assigned_layers, use_cp_schedule):
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
        self.use_cp_schedule = use_cp_schedule
        self.use_mla = model_runner.model_loader.use_mla

    def init_logger(self):
        tp_ep_log = 'TP' if not self.use_ep or self.tp_size == 1 else 'TP/EP'
        formater = logging.Formatter(
            f'[%(asctime)s %(filename)s:%(lineno)d Worker{self.pp_rank*self.tp_size+self.tp_rank} '
            f'PP{self.pp_rank} {tp_ep_log}{self.tp_rank}] %(levelname)s - %(message)s',
            datefmt="%H:%M:%S")
        for handler in logger.handlers:
            handler.setFormatter(formater)

    def init(self):
        self.init_logger()
        if self.pp_size > 1 or self.tp_size > 1:
            init_dist(self.pp_size, self.tp_size, self.use_ep, self.local_rank, self.pp_rank, self.tp_rank, self.master_addr, 
                    self.master_port, self.assigned_layers)
        self.rank = get_rank()
        torch.cuda.set_device(f'cuda:{self.local_rank}')
        
        
        self.comm.init()
        
        self.model_runner.init(self.mp_load_progress)
        self.hidden_size = self.model_runner.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        
        if self.rank == 0:
            self.worker_scheduler = WorkerScheduler(
                self.pp_size,
                self.model_runner, 
                self.use_cp_schedule,
            )
        elif self.pp_rank == 0:
            # Input data for each rank except 0
            self.schedule_queue = deque()
        else:
            # Input data for each rank except 0
            self.schedule_queue = deque()
            # Input data and intermediate data for rank except 0
            self.run_queue = deque()
            # Recv buffer
            max_tokens_ret = self.model_runner.max_tokens_ret
            self.recv_hidden_states = torch.zeros((max_tokens_ret, self.hidden_size))
            self.recv_residual = torch.zeros((max_tokens_ret, self.hidden_size))

        self.mp_alive[self.local_rank] = 1

        logger.info(f'Initialization complete')

    # driver worker => other workers
    def recv_schedule_seqs(self):
        recv_data = self.comm.recv_schedule_seqs()
        if recv_data is not None:
            seqs, mrope_positions = recv_data
            input_data = InputData(
                use_buffer=False,
                memory_manager=self.model_runner.memory_manager,
                max_seq_length=self.model_runner.tokenizer.model_max_length,
            )
            input_data.cal_input(seqs)
            if mrope_positions is not None:
                input_data.set_mrope_position(mrope_positions)
            self.schedule_queue.append(input_data)

    def forward_pp(self):
        if len(self.schedule_queue) != 0:
            input_data:InputData = self.schedule_queue.popleft()
            # pp last rank => pp next rank
            recv_pp_data(
                get_last_pp_rank(),
                input_data.tokens_cpu.shape[0],
                self.model_runner.input_hidden_states,
                self.model_runner.input_residual,
                self.ret_residual
            )
            self.model_runner.set_input(input_data)
            output = self.model_runner.step_once()
            if is_output_rank():
                self.comm.send_tokens(output)
            elif not is_last_pp_rank():
                send_pp_data(output, get_next_pp_rank())

    def recv_ipc_package(self):
        # To avoid request accumulation, we fetch all packages in comm
        cum_ipc_package = IPCPackage([])
        while True:
            ipc_package:IPCPackage = self.comm.recv_ipc_package()
            if ipc_package is not None:
                cum_ipc_package.schedule_lists.extend(ipc_package.schedule_lists)
                cum_ipc_package.abort_ids.extend(ipc_package.abort_ids)
                cum_ipc_package.log &= ipc_package.log
            else:
                break
        if len(cum_ipc_package.schedule_lists) != 0 or len(cum_ipc_package.abort_ids) != 0:
            self.worker_scheduler.add_new_requests(cum_ipc_package.schedule_lists)
            self.worker_scheduler.add_abort_ids(cum_ipc_package.abort_ids)
            self.worker_scheduler.set_log(cum_ipc_package.log)

    def recv_next_tokens(self):
        if self.pp_size != 1:  # recv tokens from last rank
            next_tokens = self.comm.recv_tokens()
            if next_tokens is not None:
                self.worker_scheduler.add_next_tokens(next_tokens)
                
    def check_abort_seqs(self):
        ipc_package = self.worker_scheduler.check_abort_seqs()
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def process_output(self):
        ipc_package = self.worker_scheduler.process_output()
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def forward_tp(self):
        if len(self.schedule_queue) != 0:
            input_data: InputData = self.schedule_queue.popleft()
            input_embeddings = None
            if self.model_runner.use_mm:
                input_embeddings, mrope_positions = self.model_runner.mm_prepare_inputs(input_data.seqs)
                self.model_runner.set_mrope_positions(mrope_positions)
            self.model_runner.set_input(input_data, input_embeddings)
            output = self.model_runner.step_once()
            if get_pp_size() != 1:
                send_pp_data(output, get_next_pp_rank())
    
    def schedule_forward(self):
        schedule_seqs = self.worker_scheduler.schedule_once()
        if len(schedule_seqs) != 0:
            input_embeddings = None
            mrope_positions = None
            if get_world_size() > 1:
                self.comm.send_schedule_seqs((schedule_seqs, None), True)
            if self.model_runner.use_mm:
                input_embeddings, mrope_positions = self.model_runner.mm_prepare_inputs(schedule_seqs)
                self.model_runner.set_mrope_positions(mrope_positions)
            if get_world_size() > 1:
                self.comm.send_schedule_seqs((schedule_seqs, mrope_positions), False)
            
            self.model_runner.cal_input(schedule_seqs, input_embeddings)
            output = self.model_runner.step_once()

            if not is_output_rank():
                send_pp_data(output, get_next_pp_rank())
            else:
                self.worker_scheduler.add_next_tokens(output)

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
        logger.info(f'Exit')
        if dist.is_initialized():
            dist.destroy_process_group()

    def handle_exception(self, e):
        logger.error(e)
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.mp_alive[self.local_rank] = -1


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
