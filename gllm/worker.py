import torch.distributed as dist
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


class Worker:
    
    def __init__(self, model_runner:ModelRunner, num_free_pages, pp_rank, pp_size, 
                 master_addr, master_port, schedule_ipc_path, output_ipc_path, token_ipc_path):
        self.model_runner = model_runner
        self.num_free_pages = num_free_pages
        self.rank = pp_rank
        self.world_size = pp_size
        self.master_addr = master_addr
        self.master_port = master_port
        zmq_ctx = zmq.Context()
        if dist.get_rank() == 0:
            self.schedule_socket = make_socket(zmq_ctx, schedule_ipc_path, zmq.PULL)
            self.output_socket = make_socket(zmq_ctx, output_ipc_path, zmq.PUSH)
            self.to_schedule_list = []
            self.already_schedule_queue = deque()
            if dist.get_world_size() != 1:
                token_socket = make_socket(zmq_ctx, token_ipc_path, zmq.PULL)
        if dist.get_rank() == dist.get_world_size() - 1 and dist.get_world_size() != 1:
            token_socket = make_socket(zmq_ctx, token_ipc_path, zmq.PUSH)
            
    def init(self):
        init_dist(self.world_size, self.rank, self.master_addr, self.master_port)
        self.model_runner.init()
        
    def set_num_free_pages(self):
        self.num_free_pages.value = self.model_runner.memory_manager.get_num_free_pages()

    def run(self):
        input_data, hidden_states, residual = recv_pp_data(
            self.model_runner.model_loader.dtype, self.model_runner.memory_manager, self.rank-1)
        output = self.model_runner.step_once(input_data,hidden_states, residual)
        if self.rank == self.world_size - 1:
            assert type(output) == list
            token_bytes = pickle.dumps(output)
            self.token_socket.send(token_bytes, copy=False)
        else:
            send_pp_data(input_data, output, self.rank+1)
    
    def schedule_run(self):
        output = None
        act_schedule_list: List[Sequence] = None

        if self.schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.schedule_socket.recv(copy=False)
            schedulerOutput = pickle.loads(recv_bytes)
            
            if isinstance(schedulerOutput, DeltaSchedulerOutput):
                to_schedule_list.extend(
                    schedulerOutput.delta_schedule_list)
                act_schedule_list = to_schedule_list
                to_schedule_list = []
            elif isinstance(schedulerOutput, SchedulerOutput):
                act_schedule_list = schedulerOutput.schedule_lists
            else:
                assert 0
        elif len(to_schedule_list) != 0:
            act_schedule_list = to_schedule_list
            to_schedule_list = []
        
        if act_schedule_list is not None:
            input_data = InputData(act_schedule_list, self.model_runner.memory_manager)
            self.already_schedule_queue.append((schedulerOutput, act_schedule_list))
            output = self.model_runner.step_once(input_data)
            
            if isinstance(output,tuple):
                send_pp_data(input_data, output, dist.get_rank()+1)
        
        return output
        

    def process_output(self, output):
            next_tokens = None
            if isinstance(self.output,list) :
                next_tokens = output
            elif dist.get_world_size() != 1 and self.token_socket.poll(timeout=0) != 0:
                recv_bytes = self.token_socket.recv(copy=False)
                next_tokens = pickle.loads(recv_bytes)
            
            if next_tokens is not None:
                schedulerOutput, act_schedule_list = self.already_schedule_queue.popleft()

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
                    self.to_schedule_list.extend([act_schedule_list[i] for i in keep_indices])
                output_bytes = pickle.dumps((schedulerOutput, next_tokens))
                self.output_socket.send(output_bytes, copy=False)
 
def run_worker(worker: Worker):
    worker.init()
    logger.info(f'GPU process {worker.rank} init')
    while True:
        if worker.rank == 0:
            worker.set_num_free_pages()
            output = worker.schedule_run()
            worker.process_output(output)   
        else:
            worker.run()