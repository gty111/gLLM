import torch
import zmq
import pickle

from collections import deque
from logger import logger
from typing import List

from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner
from gllm.dist_utils import init_dist, send_pp_data, recv_pp_data
from gllm.scheduler import SchedulerOutput, DeltaSchedulerOutput
from gllm.utils import make_socket

# Used with PipeAsyncLLM
class Worker:
    
    def __init__(self, model_runner:ModelRunner, num_free_pages, pp_rank, pp_size, 
                 master_addr, master_port, schedule_ipc_path, output_ipc_path, token_ipc_path):
        self.model_runner = model_runner
        self.num_free_pages = num_free_pages
        self.rank = pp_rank
        self.world_size = pp_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.schedule_ipc_path = schedule_ipc_path
        self.output_ipc_path = output_ipc_path
        self.token_ipc_path = token_ipc_path
        
    def init(self):
        init_dist(self.world_size, self.rank, self.master_addr, self.master_port)
        torch.cuda.set_device(f'cuda:{self.rank}')
        zmq_ctx = zmq.Context()
        if self.rank == 0:
            # main process => GPU rank 0
            self.schedule_socket = make_socket(zmq_ctx, self.schedule_ipc_path, zmq.PULL) 
            # GPU rank 0 => main process
            self.output_socket = make_socket(zmq_ctx, self.output_ipc_path, zmq.PUSH)
            # seqs to schedule
            self.seqs_to_schedule = []
            # running batch 
            self.batch_running = deque()
            # GPU rank 0 => other GPU ranks : batched seqs
            self.gpu_schedule_socket = []
            for i in range(1,self.world_size):
                self.gpu_schedule_socket.append(make_socket(zmq_ctx, f'ipc:///tmp/gllm_schedule_{i}',zmq.PUSH))
            if self.world_size != 1:
                # GPU last rank => GPU rank 0 : next tokens
                self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PULL)
        else:
            # GPU rank 0 => other GPU ranks : batched seqs
            self.gpu_schedule_socket = make_socket(zmq_ctx, f'ipc:///tmp/gllm_schedule_{self.rank}', zmq.PULL)
            # Input data for each GPU rank except rank 0 
            self.schedule_queue = deque()
        if self.rank == self.world_size - 1 and self.world_size != 1:
            # GPU last rank => GPU rank 0 : next tokens
            self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PUSH)
        self.model_runner.init()
        self.dtype = self.model_runner.memory_manager.dtype
        self.hidden_size = self.model_runner.model_loader.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        
    def set_num_free_pages(self):
        self.num_free_pages.value = self.model_runner.memory_manager.get_num_free_pages()

    # GPU process except rank 0 
    def run(self):
        if self.gpu_schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.gpu_schedule_socket.recv(copy=False)
            seqs = pickle.loads(recv_bytes)
            self.schedule_queue.append(InputData(seqs,self.model_runner.memory_manager))
        if len(self.schedule_queue) == 0:
            return
        input_data = self.schedule_queue.popleft()
        data = recv_pp_data(
            self.rank-1, self.dtype, [input_data.tokens.shape[0],self.hidden_size], self.ret_residual)
        hidden_states = None
        residual = None
        if len(data) == 2:
            hidden_states, residual = data
        else:
            hidden_states = data
            residual = None
        output = self.model_runner.step_once(input_data,hidden_states,residual)
        if self.rank == self.world_size - 1:
            assert type(output) == list
            token_bytes = pickle.dumps(output)
            self.token_socket.send(token_bytes, copy=False)
        else:
            send_pp_data(output, self.rank+1)
    
    # PP schedule
    def schedule(self):
        # to_schedule_list => schedule_list (ref already_schedule_queue)
        num_total_seqs = len(self.seqs_to_schedule)
        for schedulerOutput,schedule_list in self.batch_running:
            num_total_seqs += len(schedule_list)
        
        if num_total_seqs <= self.world_size or self.world_size == 1:
            cur_schedule_list = self.seqs_to_schedule
            self.seqs_to_schedule = []
            return cur_schedule_list    
        
        num_schedule_seqs = num_total_seqs // self.world_size
        if len(self.batch_running) > self.world_size:
            return []
        if num_schedule_seqs > len(self.seqs_to_schedule):
            return []
        cur_schedule_list = self.seqs_to_schedule[:num_schedule_seqs]
        self.seqs_to_schedule = self.seqs_to_schedule[num_schedule_seqs:]
        return cur_schedule_list
    
    # rank 0 process
    def schedule_run(self):
        output = None
        act_schedule_list: List[Sequence] = []
        schedulerOutput = None

        if self.schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.schedule_socket.recv(copy=False)
            schedulerOutput = pickle.loads(recv_bytes)
            
            if isinstance(schedulerOutput, DeltaSchedulerOutput):
                self.seqs_to_schedule.extend(
                    schedulerOutput.delta_schedule_list)
                act_schedule_list = self.schedule()
            elif isinstance(schedulerOutput, SchedulerOutput):
                act_schedule_list = schedulerOutput.schedule_lists
            else:
                assert 0
        elif len(self.seqs_to_schedule) != 0:
            schedulerOutput = DeltaSchedulerOutput([],[])
            act_schedule_list = self.schedule()
        
        if len(act_schedule_list) != 0:
            input_data = InputData(act_schedule_list, self.model_runner.memory_manager)
            seqs_bytes = pickle.dumps(act_schedule_list)
            for i in range(1,self.world_size):
                self.gpu_schedule_socket[i-1].send(seqs_bytes,copy=False)
            self.batch_running.append((schedulerOutput, act_schedule_list))
            output = self.model_runner.step_once(input_data)
            
            if type(output) != list:
                send_pp_data(output, self.rank+1)
        
        return output
        
    # rank 0 process
    def process_output(self, output):
        next_tokens = None
        if isinstance(output,list) : # word_size == 1
            next_tokens = output
        elif self.world_size != 1 and self.token_socket.poll(timeout=0) != 0: # recv tokens from last rank
            recv_bytes = self.token_socket.recv(copy=False)
            next_tokens = pickle.loads(recv_bytes)
        
        if next_tokens is not None:
            schedulerOutput, act_schedule_list = self.batch_running.popleft()

            keep_indices = []
            free_indices = []
            for idx, seq in enumerate(act_schedule_list):
                seq.computed_prompt = True
                seq.token_ids.append(next_tokens[idx])
                if seq.is_finish():
                    free_indices.append(idx)
                    self.model_runner.memory_manager.free(seq)
                else:
                    keep_indices.append(idx)
            schedulerOutput.free_indices = free_indices
            if isinstance(schedulerOutput, DeltaSchedulerOutput):
                self.seqs_to_schedule.extend([act_schedule_list[i] for i in keep_indices])
            output_bytes = pickle.dumps((schedulerOutput, next_tokens))
            self.output_socket.send(output_bytes, copy=False)
 
def run_worker(worker: Worker):
    worker.init()
    logger.info(f'Worker {worker.rank} init')
    while True:
        if worker.rank == 0:
            worker.set_num_free_pages()
            output = worker.schedule_run()
            worker.process_output(output)   
        else:
            worker.run()