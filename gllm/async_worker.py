import torch
import zmq
import pickle
import torch.distributed as dist
import traceback
import time
import random
import logging
import asyncio

from collections import deque
from logger import logger
from typing import List
from functools import reduce

from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner
from gllm.dist_utils import init_dist, send_pp_data, recv_pp_data
from gllm.scheduler import SchedulerOutput
from gllm.utils import make_socket, make_async
from gllm.memory_manager import PrefixMemoryManager


def async_wrapper(func):
    async def wrapper(*args, **kwargs):
        while True:
            await func(*args, **kwargs)
            await asyncio.sleep(0)
    return wrapper

# Used with PipeAsyncLLM
class AsyncWorker:
    
    def __init__(self, model_runner:ModelRunner, pp_rank, pp_size, 
                 master_addr, master_port, schedule_ipc_path, output_ipc_path, token_ipc_path, mp_alive,
                 mp_load_progress):
        self.model_runner = model_runner
        self.pp_rank = pp_rank # pp rank
        self.pp_size = pp_size # pp size
        self.master_addr = master_addr
        self.master_port = master_port
        self.schedule_ipc_path = schedule_ipc_path
        self.output_ipc_path = output_ipc_path
        self.token_ipc_path = token_ipc_path
        self.mp_alive = mp_alive
        self.mp_load_progress = mp_load_progress
    
    def get_pp_next_rank(self):
        # return device_rank of next pp_rank
        return (self.pp_rank + 1) % self.pp_size
        
    def get_pp_last_rank(self):
        # return device_rank of last pp_rank
        return (self.pp_rank - 1 + self.pp_size) % self.pp_size
    
    def init(self):
        self.init_logger()
        init_dist(self.pp_size, self.pp_rank, self.pp_size, self.pp_rank, self.master_addr, self.master_port)
        torch.cuda.set_device(f'cuda:{self.pp_rank}')
        zmq_ctx = zmq.Context()
        if self.pp_rank == 0:
            # main process => rank 0
            self.schedule_socket = make_socket(zmq_ctx, self.schedule_ipc_path, zmq.PULL) 
            # rank 0 => main process
            self.output_socket = make_socket(zmq_ctx, self.output_ipc_path, zmq.PUSH)
            # seqs to schedule
            self.seqs_to_prefill: deque[Sequence] = deque()
            self.seqs_to_decode: deque[Sequence] = deque()
            # running batch 
            self.batch_running = deque()
            # next tokens
            self.next_tokens_queue = deque()
            self.log_time = 0
            # preempt seqs
            self.num_preempt_seqs = 0
            self.log_num_preempt_seqs = 0
            # num wait tokens
            self.num_wait_tokens = 0
            # rank 0 => other ranks : batched seqs
            self.gpu_schedule_socket = []
            for i in range(1,self.pp_size):
                self.gpu_schedule_socket.append(make_socket(zmq_ctx, f'{self.schedule_ipc_path}_{i}',zmq.PUSH))
            if self.pp_size != 1:
                # last rank => rank 0 : next tokens
                self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PULL)
        else:
            # rank 0 => other ranks : batched seqs
            self.gpu_schedule_socket = make_socket(zmq_ctx, f'{self.schedule_ipc_path}_{self.pp_rank}', zmq.PULL)
            # Input data for each rank except 0 
            self.schedule_queue = deque()
            # Input data and intermediate data for rank except 0
            self.run_queue = deque()
        if self.pp_rank == self.pp_size - 1 and self.pp_size != 1:
            # last rank => rank 0 : next tokens
            self.token_socket = make_socket(zmq_ctx, self.token_ipc_path, zmq.PUSH)
        self.model_runner.init(self.mp_load_progress)
        self.dtype = self.model_runner.memory_manager.dtype
        self.hidden_size = self.model_runner.model_loader.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        self.max_decode_seqs = self.model_runner.max_decode_seqs
        self.max_batch_tokens = self.model_runner.max_batch_tokens
        self.page_size = self.model_runner.page_size
        self.ratio_threshold_free_pages = self.model_runner.ratio_threshold_free_pages
        self.num_threshold_free_pages = int(
            self.model_runner.ratio_threshold_free_pages * self.model_runner.memory_manager.get_num_free_pages())
        
        self.mp_alive[self.pp_rank] = 1
    
    def init_logger(self):
        formater = logging.Formatter(f"[%(asctime)s %(filename)s:%(lineno)d Worker {self.pp_rank}] %(levelname)s - %(message)s")
        for handler in logger.handlers:
            handler.setFormatter(formater)
    
    def get_num_free_pages(self):
        return self.model_runner.memory_manager.get_num_free_pages()

    # rank except 0
    @async_wrapper
    async def recv_schedule_seqs(self):
        if self.gpu_schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.gpu_schedule_socket.recv(copy=False)
            seqs = pickle.loads(recv_bytes)
            self.schedule_queue.append(InputData(seqs,self.model_runner.memory_manager))
        
    # rank except 0
    @async_wrapper
    async def recv_intermediate_data(self):
        if len(self.schedule_queue) != 0:
            input_data = self.schedule_queue.popleft()
            intermediate_data = await make_async(recv_pp_data)(
                self.get_pp_last_rank(), self.dtype, 
                [input_data.tokens.shape[0],self.hidden_size], self.ret_residual)
            self.run_queue.append((input_data,intermediate_data))
    
    # rank except 0 
    @async_wrapper
    async def run(self):
        # model forward
        if len(self.run_queue) != 0:
            hidden_states = None
            residual = None
            
            if self.ret_residual:
                input_data, (hidden_states_future, residual_future, hidden_states, residual) = self.run_queue[0]
                if not hidden_states_future.is_completed() or not residual_future.is_completed():
                    return
            else:
                input_data, (hidden_states_future, hidden_states) = self.run_queue[0]
                if not hidden_states_future.is_completed():
                    return
            
            self.run_queue.popleft()
                
            output = self.model_runner.step_once(input_data,hidden_states,residual)
            if self.pp_rank == self.pp_size - 1:
                assert type(output) == list
                token_bytes = pickle.dumps(output)
                self.token_socket.send(token_bytes, copy=False)
            else:
                send_pp_data(output, self.get_pp_next_rank())
    
    # rank 0
    def get_num_decode_seqs(self):
        num_decode_seqs = len(self.seqs_to_decode) + reduce(lambda x,y: x+len(y),self.batch_running,0)
        return num_decode_seqs
    
    # rank 0
    def update_num_wait_tokens(self):
        self.num_wait_tokens = reduce(
            lambda x,y: x + len(y.token_ids) - y.computed_token_num,self.seqs_to_prefill,0)
    
    # rank 0: check if preempt seqs 
    def check_preempt(self,num_decode_tokens):
        preempt_seqs = []
        while self.model_runner.memory_manager.get_num_free_pages() < num_decode_tokens and len(self.seqs_to_decode) != 0:
            seq_to_preempt = self.seqs_to_decode.popleft()
            self.model_runner.memory_manager.free(seq_to_preempt)
            seq_to_preempt.preempt()
            preempt_seqs.append(seq_to_preempt)
            
        self.seqs_to_prefill.extendleft(preempt_seqs)
        
        self.num_preempt_seqs += len(preempt_seqs)
        if self.num_preempt_seqs - self.log_num_preempt_seqs >= 10:
            self.log_num_preempt_seqs = self.num_preempt_seqs
            logger.warning(f'#Preempted seqs: {self.num_preempt_seqs}')
            logger.warning('Try increase --ratio-free-pages or the performance is poor!')
    
    # rank 0: PP schedule 
    def schedule_naive(self):
        schedule_prefill_seqs = []
        schedule_decode_seqs = []
        
        num_tokens_budget = self.max_batch_tokens
        
        self.check_preempt(min(num_tokens_budget,len(self.seqs_to_decode)))
        # decode
        for _ in range(num_tokens_budget):
            if len(self.seqs_to_decode) == 0:
                break
            seq = self.seqs_to_decode.pop()
            seq.to_compute_token_num = 1
            schedule_decode_seqs.append(seq)
            
            
        self.model_runner.memory_manager.pre_allocate_page(schedule_decode_seqs)
        
        num_tokens_budget -= len(schedule_decode_seqs)
        
        # prefill
        prefill_batched_token_nums = 0
        while len(self.seqs_to_prefill) != 0 and num_tokens_budget != 0:
            seq = self.seqs_to_prefill.popleft()
            self.model_runner.memory_manager.pre_allocate_page([seq])
            if len(seq.token_ids)-seq.computed_token_num <= num_tokens_budget:
                seq.to_compute_token_num = len(seq.token_ids) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                num_tokens_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += num_tokens_budget
                seq.to_compute_token_num = num_tokens_budget
                num_tokens_budget = 0
            schedule_prefill_seqs.append(seq)

        if time.time()-self.log_time > 1:
            self.log_time = time.time()
            log_info = '#wait: %4d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%' % (
                            len(self.seqs_to_prefill),
                            self.get_num_decode_seqs(),
                            prefill_batched_token_nums,
                            len(schedule_decode_seqs),
                            '%.2f' % self.model_runner.memory_manager.get_memory_util())
            if isinstance(self.model_runner.memory_manager, PrefixMemoryManager):
                log_info += ' cache_hit_rate: %5s %%' % ('%.2f' % self.model_runner.memory_manager.get_cache_hit_rate())
                logger.info(log_info)
            else:
                logger.info(log_info)
        return schedule_prefill_seqs + schedule_decode_seqs
    
    # rank 0: PP schedule 
    def schedule(self):
        
        schedule_prefill_seqs = []
        schedule_decode_seqs = []
        
        # prefill
        prefill_token_budget = self.page_size * max(self.get_num_free_pages()-self.num_threshold_free_pages,0)
        if self.pp_size > 1 and prefill_token_budget != 0:
            self.update_num_wait_tokens()
            free_ratio = self.model_runner.memory_manager.get_memory_free()
            # a = ratio_threshold_free_pages
            # free_ratio in [1,a] | prefill_ratio in [1,0]
            prefill_ratio = (free_ratio - self.ratio_threshold_free_pages) / (1-self.ratio_threshold_free_pages)
            prefill_ratio = max(prefill_ratio,0)
            prefill_token_budget = min(
                round(prefill_ratio * self.max_batch_tokens),
                prefill_token_budget)
            prefill_token_budget = min(max(self.num_wait_tokens//8,32),prefill_token_budget)
        else:
            prefill_token_budget = min(self.max_batch_tokens, prefill_token_budget)
        prefill_batched_token_nums = 0
        while len(self.seqs_to_prefill) != 0 and prefill_token_budget != 0:
            seq = self.seqs_to_prefill.popleft()
            if isinstance(self.model_runner.memory_manager, PrefixMemoryManager) and seq.computed_token_num == 0:
                self.model_runner.memory_manager.pre_allocate_computed_page([seq])
            if len(seq.token_ids)-seq.computed_token_num <= prefill_token_budget:
                seq.to_compute_token_num = len(seq.token_ids) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                prefill_token_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += prefill_token_budget
                seq.to_compute_token_num = prefill_token_budget
                prefill_token_budget = 0
            schedule_prefill_seqs.append(seq)

        self.model_runner.memory_manager.pre_allocate_page(schedule_prefill_seqs)
        
        # decode
        num_total_decode_seqs = self.get_num_decode_seqs()
        if num_total_decode_seqs < self.pp_size:
            decode_token_budget = num_total_decode_seqs
        else:
            # here we add num_total_decode_seqs to random.randint(0,self.pp_size-1))
            # because we want to solve the situation when #seqs=5 pp_size=4
            decode_token_budget = (num_total_decode_seqs + random.randint(0,self.pp_size-1)) // self.pp_size
        
        self.check_preempt(decode_token_budget)
        
        for _ in range(decode_token_budget):
            if len(self.seqs_to_decode) == 0:
                break
            seq = self.seqs_to_decode.popleft()
            seq.to_compute_token_num = 1
            schedule_decode_seqs.append(seq)
            
        self.model_runner.memory_manager.pre_allocate_page(schedule_decode_seqs)

        if time.time()-self.log_time > 1:
            self.log_time = time.time()
            log_info = '#wait: %4d/%8d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%' % (
                            len(self.seqs_to_prefill),
                            self.num_wait_tokens,
                            num_total_decode_seqs,
                            prefill_batched_token_nums,
                            len(schedule_decode_seqs),
                            '%.2f' % self.model_runner.memory_manager.get_memory_util())
            if isinstance(self.model_runner.memory_manager, PrefixMemoryManager):
                log_info += ' cache_hit_rate: %5s %%' % ('%.2f' % self.model_runner.memory_manager.get_cache_hit_rate())
                logger.info(log_info)
            else:
                logger.info(log_info)
        # with open('log','a') as f:
        #     f.write(f'{prefill_batched_token_nums} {len(schedule_decode_seqs)}\n')
        return schedule_prefill_seqs + schedule_decode_seqs
    
    
    # rank 0
    @async_wrapper
    async def recv_requests(self):
        if self.schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.schedule_socket.recv(copy=False)
            schedulerOutput: SchedulerOutput = pickle.loads(recv_bytes)
            self.seqs_to_prefill.extend(schedulerOutput.schedule_lists)
    
    # rank 0
    @async_wrapper
    async def schedule_run(self):
        if len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0 and len(self.batch_running) < self.pp_size:
            schedule_seqs = self.schedule()
            if len(schedule_seqs) != 0:
                input_data = InputData(schedule_seqs, self.model_runner.memory_manager)
                if self.pp_size > 1:
                    seqs_bytes = pickle.dumps(schedule_seqs)
                    for i in range(1,self.pp_size):
                        self.gpu_schedule_socket[i-1].send(seqs_bytes,copy=False)
                self.batch_running.append(schedule_seqs)
                output = self.model_runner.step_once(input_data)
                
                if type(output) != list:
                    send_pp_data(output, self.get_pp_next_rank())
                else:
                    self.next_tokens_queue.append(output)
    
    # rank 0
    @async_wrapper
    async def recv_next_tokens(self):
        if self.pp_size != 1: # recv tokens from last rank
            recv_bytes = await make_async(self.token_socket.recv)(copy=False)
            next_tokens = pickle.loads(recv_bytes)
            self.next_tokens_queue.append(next_tokens)
    
    # rank 0
    @async_wrapper
    async def process_output(self):
        if len(self.next_tokens_queue) != 0:
            next_tokens = self.next_tokens_queue.popleft()
            
            schedule_seqs:List[Sequence] = self.batch_running.popleft()
            assert len(next_tokens) == len(schedule_seqs)

            send_tokens = []
            schedulerOutput = SchedulerOutput([])
            
            for idx, seq in enumerate(schedule_seqs):
                seq.computed_token_num += seq.to_compute_token_num
                if seq.computed_prompt():
                    schedulerOutput.act_schedule_ids.append(seq.seq_id)
                    send_tokens.append(next_tokens[idx])
                    seq.token_ids.append(next_tokens[idx])
                if seq.is_finish():
                    schedulerOutput.free_ids.append(seq.seq_id)
                    self.model_runner.memory_manager.free(seq)
                elif seq.computed_prompt():
                    self.seqs_to_decode.appendleft(seq)
                else:
                    self.seqs_to_prefill.appendleft(seq)
            
            output_bytes = pickle.dumps((schedulerOutput, send_tokens))
            self.output_socket.send(output_bytes, copy=False)

    def handle_keyboardInterrupt(self):
        logger.info(f'Exit')
        dist.destroy_process_group()
        self.mp_alive[self.pp_rank] = -1
        
    def handle_exception(self,e):
        logger.error(e)
        traceback.print_exc()
        dist.destroy_process_group()
        self.mp_alive[self.pp_rank] = -1

def create_worker_task(func):
    return asyncio.get_event_loop().create_task(func())

async def run_worker_async(worker: AsyncWorker):
    worker.init()
    logger.info(f'Init')
    
    tasks = []
    if worker.pp_rank == 0:
        tasks.append(create_worker_task(worker.recv_requests))
        tasks.append(create_worker_task(worker.schedule_run))
        tasks.append(create_worker_task(worker.process_output))
        tasks.append(create_worker_task(worker.recv_next_tokens))
    else:
        tasks.append(create_worker_task(worker.recv_schedule_seqs))
        tasks.append(create_worker_task(worker.recv_intermediate_data))
        tasks.append(create_worker_task(worker.run))
    await asyncio.gather(*tasks)
    
def run_worker(worker: AsyncWorker):
    try:
        asyncio.run(run_worker_async(worker))
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)