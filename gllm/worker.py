import torch.multiprocessing as mp
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
    
    def __init__(self, model_runner:ModelRunner, mp_share_nums, mp_queue, num_free_pages, pp_rank, pp_size, 
                 master_addr, master_port, schedule_ipc_path, output_ipc_path, token_ipc_path,
                 interleaved_pp):
        self.model_runner = model_runner
        self.mp_share_nums = mp_share_nums
        self.mp_queue: mp.Queue = mp_queue
        self.num_free_pages = num_free_pages
        self.pp_rank = pp_rank # pp rank
        self.pp_size = pp_size # pp size
        self.master_addr = master_addr
        self.master_port = master_port
        self.schedule_ipc_path = schedule_ipc_path
        self.output_ipc_path = output_ipc_path
        self.token_ipc_path = token_ipc_path
        
        self.interleaved_pp = interleaved_pp
        self.device_size = self.pp_size if not interleaved_pp else self.pp_size//2
        self.device_rank = self.pp_rank if not interleaved_pp else self.pp_rank % self.device_size
    
    def get_pp_next_rank(self):
        # return device_rank of next pp_rank
        return (self.device_rank + 1) % self.device_size
        
    def get_pp_last_rank(self):
        # return device_rank of last pp_rank
        return (self.device_rank - 1 + self.device_size) % self.device_size
    
    def init(self):
        init_dist(self.pp_size, self.pp_rank, self.device_size, self.device_rank, self.master_addr, self.master_port)
        torch.cuda.set_device(f'cuda:{self.device_rank}')
        zmq_ctx = zmq.Context()
        if self.pp_rank == 0:
            # main process => GPU pp rank 0
            self.schedule_socket = make_socket(zmq_ctx, self.schedule_ipc_path, zmq.PULL) 
            # GPU pp rank 0 => main process
            self.output_socket = make_socket(zmq_ctx, self.output_ipc_path, zmq.PUSH)
            # seqs to schedule
            self.seqs_to_schedule = []
            # running batch 
            self.batch_running = deque()
            # GPU pp rank 0 => other GPU ranks : batched seqs
            self.gpu_schedule_socket = []
            for i in range(1,self.pp_size):
                self.gpu_schedule_socket.append(make_socket(zmq_ctx, f'ipc:///tmp/gllm_schedule_{i}',zmq.PUSH))
            if self.pp_size != 1:
                # GPU last pp rank => GPU pp rank 0 : next tokens
                self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PULL)
            if self.interleaved_pp:
                # Intermediate data for comm between NCCL groups
                self.run_queue = deque()
        else:
            # GPU pp rank 0 => other GPU pp ranks : batched seqs
            self.gpu_schedule_socket = make_socket(zmq_ctx, f'ipc:///tmp/gllm_schedule_{self.pp_rank}', zmq.PULL)
            # Input data for each GPU pp rank except pp rank 0 
            self.schedule_queue = deque()
            # Input data and intermediate data for each GPU pp rank except pp rank 0
            self.run_queue = deque()
        if self.pp_rank == self.pp_size - 1 and self.pp_size != 1:
            # GPU last pp rank => GPU pp rank 0 : next tokens
            self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PUSH)
        self.model_runner.init(self.mp_share_nums,self.interleaved_pp)
        self.dtype = self.model_runner.memory_manager.dtype
        self.hidden_size = self.model_runner.model_loader.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        
    def set_num_free_pages(self):
        self.num_free_pages.value = self.model_runner.memory_manager.get_num_free_pages()

    # GPU process except pp rank 0 
    def run(self):
        # model forward
        if len(self.run_queue) != 0:
            hidden_states = None
            residual = None
            input_data,intermediate_data = self.run_queue.popleft()
            if len(intermediate_data) == 4:
                if not (intermediate_data[0].is_completed() and intermediate_data[1].is_completed()):
                    self.run_queue.appendleft((input_data,intermediate_data))
                    return
                else:
                    hidden_states, residual = intermediate_data[2], intermediate_data[3]
            elif len(intermediate_data) == 2:
                if not intermediate_data[0].is_completed():
                    self.run_queue.appendleft((input_data,intermediate_data))
                    return
                else:
                    hidden_states = intermediate_data[1]
            else:
                assert 0
            output = self.model_runner.step_once(input_data,hidden_states,residual)
            if self.pp_rank == self.pp_size - 1:
                assert type(output) == list
                token_bytes = pickle.dumps(output)
                self.token_socket.send(token_bytes, copy=False)
            else:
                if not self.interleaved_pp or self.pp_rank != self.pp_size//2 - 1:
                    send_pp_data(output, self.get_pp_next_rank())
                else:
                    if len(output) == 2:
                        output = (output[0].to('cuda:0'),output[1].to('cuda:0')) 
                    else:
                        output = output.to('cuda:0')
                    self.mp_queue.put(output)
        
        # recv schedule seqs
        if self.gpu_schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.gpu_schedule_socket.recv(copy=False)
            seqs = pickle.loads(recv_bytes)
            self.schedule_queue.append(InputData(seqs,self.model_runner.memory_manager))
        if len(self.schedule_queue) != 0:
            # recv intermediate data
            if not self.interleaved_pp or self.device_rank != 0:
                input_data = self.schedule_queue.popleft()
                intermediate_data = recv_pp_data(
                    self.get_pp_last_rank(), self.dtype, 
                    [input_data.tokens.shape[0],self.hidden_size], self.ret_residual)
                self.run_queue.append((input_data,intermediate_data))
            else: # comm from mp queue
                if not self.mp_queue.empty():
                    input_data = self.schedule_queue.popleft()
                    intermediate_data = self.mp_queue.get() 
                    hidden_states = None
                    residual = None
                    if len(intermediate_data) == 2:
                        hidden_states, residual = intermediate_data
                    else:
                        hidden_states = intermediate_data
                    output = self.model_runner.step_once(input_data,hidden_states,residual)
                    send_pp_data(output, self.get_pp_next_rank())
                    
    
    # PP schedule
    def schedule(self):
        # to_schedule_list => schedule_list (ref already_schedule_queue)
        num_total_seqs = len(self.seqs_to_schedule)
        for schedulerOutput,schedule_list in self.batch_running:
            num_total_seqs += len(schedule_list)
        
        if num_total_seqs <= self.pp_size or self.pp_size == 1:
            cur_schedule_list = self.seqs_to_schedule
            self.seqs_to_schedule = []
            return cur_schedule_list    
        
        num_schedule_seqs = num_total_seqs // self.pp_size
        if len(self.batch_running) > self.pp_size:
            return []
        if num_schedule_seqs > len(self.seqs_to_schedule):
            return []
        cur_schedule_list = self.seqs_to_schedule[:num_schedule_seqs]
        self.seqs_to_schedule = self.seqs_to_schedule[num_schedule_seqs:]
        return cur_schedule_list
    
    # pp rank 0 process
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
            for i in range(1,self.pp_size):
                self.gpu_schedule_socket[i-1].send(seqs_bytes,copy=False)
            self.batch_running.append((schedulerOutput, act_schedule_list))
            output = self.model_runner.step_once(input_data)
            
            if type(output) != list:
                send_pp_data(output, self.get_pp_next_rank())
        
        return output
        
    # pp rank 0 process
    def process_output(self, output):
        next_tokens = None
        if isinstance(output,list) : # word_size == 1
            next_tokens = output
        elif self.pp_size != 1 and self.token_socket.poll(timeout=0) != 0: # recv tokens from last rank
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
    logger.info(f'Worker {worker.pp_rank} init')
    while True:
        if worker.pp_rank == 0:
            worker.set_num_free_pages()
            output = worker.schedule_run()
            worker.process_output(output)   
        else:
            worker.run()