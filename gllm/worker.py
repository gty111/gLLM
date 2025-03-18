import torch
import zmq
import pickle
import torch.distributed as dist
import traceback
import time

from collections import deque
from logger import logger
from typing import List

from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner
from gllm.dist_utils import init_dist, send_pp_data, recv_pp_data
from gllm.scheduler import SchedulerOutput
from gllm.utils import make_socket
from gllm.memory_manager import PrefixMemoryManager

# Used with PipeAsyncLLM
class Worker:
    
    def __init__(self, model_runner:ModelRunner, num_free_pages, pp_rank, pp_size, 
                 master_addr, master_port, schedule_ipc_path, output_ipc_path, token_ipc_path,mp_alive):
        self.model_runner = model_runner
        self.num_free_pages = num_free_pages
        self.pp_rank = pp_rank # pp rank
        self.pp_size = pp_size # pp size
        self.master_addr = master_addr
        self.master_port = master_port
        self.schedule_ipc_path = schedule_ipc_path
        self.output_ipc_path = output_ipc_path
        self.token_ipc_path = token_ipc_path
        self.mp_alive = mp_alive
    
    def get_pp_next_rank(self):
        # return device_rank of next pp_rank
        return (self.pp_rank + 1) % self.pp_size
        
    def get_pp_last_rank(self):
        # return device_rank of last pp_rank
        return (self.pp_rank - 1 + self.pp_size) % self.pp_size
    
    def init(self):
        init_dist(self.pp_size, self.pp_rank, self.pp_size, self.pp_rank, self.master_addr, self.master_port)
        torch.cuda.set_device(f'cuda:{self.pp_rank}')
        zmq_ctx = zmq.Context()
        if self.pp_rank == 0:
            # main process => rank 0
            self.schedule_socket = make_socket(zmq_ctx, self.schedule_ipc_path, zmq.PULL) 
            # rank 0 => main process
            self.output_socket = make_socket(zmq_ctx, self.output_ipc_path, zmq.PUSH)
            # seqs to schedule
            self.seqs_to_prefill: List[Sequence] = []
            self.seqs_to_decode: List[Sequence] = []
            # running batch 
            self.batch_running = deque()
            self.num_running_decode_seqs = 0
            self.log_time = 0
            # rank 0 => other ranks : batched seqs
            self.gpu_schedule_socket = []
            for i in range(1,self.pp_size):
                self.gpu_schedule_socket.append(make_socket(zmq_ctx, f'ipc:///tmp/gllm_schedule_{i}',zmq.PUSH))
            if self.pp_size != 1:
                # last rank => rank 0 : next tokens
                self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PULL)
        else:
            # rank 0 => other ranks : batched seqs
            self.gpu_schedule_socket = make_socket(zmq_ctx, f'ipc:///tmp/gllm_schedule_{self.pp_rank}', zmq.PULL)
            # Input data for each rank except 0 
            self.schedule_queue = deque()
            # Input data and intermediate data for rank except 0
            self.run_queue = deque()
        if self.pp_rank == self.pp_size - 1 and self.pp_size != 1:
            # last rank => rank 0 : next tokens
            self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PUSH)
        self.model_runner.init()
        self.dtype = self.model_runner.memory_manager.dtype
        self.hidden_size = self.model_runner.model_loader.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        self.max_decode_seqs = self.model_runner.max_decode_seqs
        self.max_batch_tokens = self.model_runner.max_batch_tokens
        self.num_threshold_free_pages = int(
            self.model_runner.ratio_threshold_free_pages * self.model_runner.memory_manager.get_num_free_pages())
        self.page_size = self.model_runner.page_size
        
        self.mp_alive[self.pp_rank] = 1
        self.set_num_free_pages()
    
    def get_num_free_pages(self):
        return self.model_runner.memory_manager.get_num_free_pages()
    
    def set_num_free_pages(self):
        self.num_free_pages.value = self.model_runner.memory_manager.get_num_free_pages()

    # rank except 0 
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
                send_pp_data(output, self.get_pp_next_rank())
        
        # recv schedule seqs
        if self.gpu_schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.gpu_schedule_socket.recv(copy=False)
            seqs = pickle.loads(recv_bytes)
            self.schedule_queue.append(InputData(seqs,self.model_runner.memory_manager))
        if len(self.schedule_queue) != 0:
            # recv intermediate data
            input_data = self.schedule_queue.popleft()
            intermediate_data = recv_pp_data(
                self.get_pp_last_rank(), self.dtype, 
                [input_data.tokens.shape[0],self.hidden_size], self.ret_residual)
            self.run_queue.append((input_data,intermediate_data))
                    
    # rank 0: check if preempt seqs 
    def check_preempt(self):
        preempt_ids = []
        while self.model_runner.memory_manager.get_num_free_slots() < self.max_batch_tokens and len(self.seqs_to_decode) != 0:
            seq_to_preempt = self.seqs_to_decode.pop()
            preempt_ids.append(seq_to_preempt.seq_id)
            self.model_runner.memory_manager.free(seq_to_preempt)
        self.seqs_to_prefill = preempt_ids + self.seqs_to_prefill
    
    # rank 0: PP schedule 
    def schedule(self):
        if len(self.batch_running) >= self.pp_size:
            return []
        
        self.check_preempt()
        
        schedule_prefill_seqs = []
        schedule_decode_seqs = []
        
        # prefill
        prefill_token_budget = max(self.get_num_free_pages()-self.num_threshold_free_pages,0)
        if self.pp_size > 1:
            prefill_token_budget = min(
                round(self.model_runner.memory_manager.get_memory_free() * self.max_batch_tokens),
                prefill_token_budget)
        
        prefill_batched_token_nums = 0
        for seq in self.seqs_to_prefill:
            if prefill_token_budget == 0:
                break
            self.model_runner.memory_manager.pre_allocate_page([seq])
            if len(seq.token_ids)-seq.computed_token_num <= prefill_token_budget:
                seq.to_compute_token_num = len(seq.token_ids) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                prefill_token_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += prefill_token_budget
                seq.to_compute_token_num = prefill_token_budget
                prefill_token_budget = 0
                
            schedule_prefill_seqs.append(seq)
        for seq in schedule_prefill_seqs:
            self.seqs_to_prefill.remove(seq)
        
        # decode
        num_total_decode_seqs = len(self.seqs_to_decode) + self.num_running_decode_seqs
        if num_total_decode_seqs < self.pp_size:
            decode_token_budget = num_total_decode_seqs
        else:
            decode_token_budget = num_total_decode_seqs // self.pp_size
        schedule_decode_seqs = self.seqs_to_decode[:decode_token_budget]
        self.seqs_to_decode = self.seqs_to_decode[decode_token_budget:]
        self.num_running_decode_seqs += len(schedule_decode_seqs)
        
        self.model_runner.memory_manager.pre_allocate_page(schedule_decode_seqs)
        for seq in schedule_decode_seqs:
            seq.to_compute_token_num = 1

        if time.time()-self.log_time > 1:
            self.log_time = time.time()
            log_info = '#wait: %4d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%' % (
                            len(self.seqs_to_prefill),
                            num_total_decode_seqs,
                            prefill_batched_token_nums,
                            len(schedule_decode_seqs),
                            '%.2f' % self.model_runner.memory_manager.get_memory_util())
            if isinstance(self.model_runner.memory_manager, PrefixMemoryManager):
                log_info += ' cache_hit_rate: %5s %%' % ('%.2f' % self.model_runner.memory_manager.get_cache_hit_rate())
                logger.info(log_info)
            else:
                logger.info(log_info)
        return schedule_prefill_seqs + schedule_decode_seqs
    
    # rank 0
    def schedule_run(self):
        output = None

        if self.schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.schedule_socket.recv(copy=False)
            schedulerOutput = pickle.loads(recv_bytes)
            self.seqs_to_prefill.extend(schedulerOutput.schedule_lists)
        
        if len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0:
            schedule_seqs = self.schedule()
            if len(schedule_seqs) != 0:
                input_data = InputData(schedule_seqs, self.model_runner.memory_manager)
                self.set_num_free_pages()
                if self.pp_size > 1:
                    seqs_bytes = pickle.dumps(schedule_seqs)
                    for i in range(1,self.pp_size):
                        self.gpu_schedule_socket[i-1].send(seqs_bytes,copy=False)
                self.batch_running.append(schedule_seqs)
                output = self.model_runner.step_once(input_data)
                
                if type(output) != list:
                    send_pp_data(output, self.get_pp_next_rank())
        
        return output
        
    # rank 0
    def process_output(self, output):
        next_tokens = None
        if isinstance(output,list) : # word_size == 1
            next_tokens = output
        elif self.pp_size != 1 and self.token_socket.poll(timeout=0) != 0: # recv tokens from last rank
            recv_bytes = self.token_socket.recv(copy=False)
            next_tokens = pickle.loads(recv_bytes)
        
        if next_tokens is not None:
            schedule_seqs:List[Sequence] = self.batch_running.popleft()
            assert len(next_tokens) == len(schedule_seqs)
            free_ids = []
            decode_seqs = []
            prefill_seqs = []
            
            send_indices = []
            for idx, seq in enumerate(schedule_seqs):
                if seq.computed_prompt():
                    self.num_running_decode_seqs -= 1
                seq.computed_token_num += seq.to_compute_token_num
                if seq.computed_prompt():
                    send_indices.append(idx)
                    seq.token_ids.append(next_tokens[idx])
                if seq.is_finish():
                    free_ids.append(seq.seq_id)
                    self.model_runner.memory_manager.free(seq)
                elif seq.computed_prompt():
                    decode_seqs.append(seq)
                else:
                    prefill_seqs.append(seq)
            self.seqs_to_decode = decode_seqs + self.seqs_to_decode
            self.seqs_to_prefill = prefill_seqs + self.seqs_to_prefill
            
            schedulerOutput = SchedulerOutput([])
            schedulerOutput.free_ids = free_ids
            schedulerOutput.act_schedule_ids = [schedule_seqs[i].seq_id for i in send_indices]
            output_bytes = pickle.dumps((schedulerOutput, [next_tokens[i] for i in send_indices]))
            self.output_socket.send(output_bytes, copy=False)
 
def run_worker(worker: Worker):
    try:
        worker.init()
        logger.info(f'Worker {worker.pp_rank} init')
        while True:
            if worker.pp_rank == 0:
                output = worker.schedule_run()
                worker.process_output(output)   
            else:
                worker.run()
    except KeyboardInterrupt as e:
        logger.info(f'Worker {worker.pp_rank} exit')
        dist.destroy_process_group()
        worker.mp_alive[worker.pp_rank] = -1
    except Exception as e:
        logger.error(f'Worker {worker.pp_rank} \n{e}')
        traceback.print_exc()
        dist.destroy_process_group()
        worker.mp_alive[worker.pp_rank] = -1