import torch
import torch.distributed as dist
import traceback
import time
import random
import logging

from collections import deque
from logger import logger
from typing import List
from functools import reduce

from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner
from gllm.dist_utils import init_dist, send_pp_data, recv_pp_data
from gllm.scheduler import IPCPackage
from gllm.memory_manager import PrefixMemoryManager
from gllm.zmq_comm import zmqComm

# Used with PipeAsyncLLM
class Worker:

    def __init__(self, model_runner: ModelRunner, pp_rank, pp_size,
                 master_addr, master_port, comm: zmqComm, mp_alive,
                 mp_load_progress, assigned_layers, use_naive_schedule):
        self.model_runner = model_runner
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.comm = comm
        self.mp_alive = mp_alive
        self.mp_load_progress = mp_load_progress
        self.assigned_layers = assigned_layers
        self.use_naive_schedule = use_naive_schedule

    def get_pp_next_rank(self):
        return (self.pp_rank + 1) % self.pp_size

    def get_pp_last_rank(self):
        return (self.pp_rank - 1 + self.pp_size) % self.pp_size

    def get_num_free_pages(self):
        return self.model_runner.memory_manager.get_num_free_pages()

    def get_num_decode_seqs(self):
        num_decode_seqs = len(self.seqs_to_decode) + \
            reduce(lambda x, y: x+len(y), self.batch_running, 0)
        return num_decode_seqs

    def update_num_wait_tokens(self):
        self.num_wait_tokens = reduce(
            lambda x, y: x + len(y.token_ids) - y.computed_token_num, self.seqs_to_prefill, 0)

    def init_logger(self):
        formater = logging.Formatter(
            f"[%(asctime)s %(filename)s:%(lineno)d Worker {self.pp_rank}] %(levelname)s - %(message)s")
        for handler in logger.handlers:
            handler.setFormatter(formater)

    def init(self):
        self.init_logger()
        self.comm.init()
        init_dist(self.pp_size, self.pp_rank, self.pp_size, self.pp_rank,
                  self.master_addr, self.master_port, self.assigned_layers)
        torch.cuda.set_device(f'cuda:{self.pp_rank}')
        if self.pp_rank == 0:
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
        else:
            # Input data for each rank except 0
            self.schedule_queue = deque()
            # Input data and intermediate data for rank except 0
            self.run_queue = deque()
        self.model_runner.init(self.mp_load_progress)
        self.dtype = self.model_runner.memory_manager.dtype
        self.hidden_size = self.model_runner.model_loader.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        self.maxp = self.model_runner.maxp
        self.minp = self.model_runner.minp
        self.iterp = self.model_runner.iterp
        self.page_size = self.model_runner.page_size
        self.kvthresh = self.model_runner.kvthresh
        self.num_kvthresh_pages = int(
            self.model_runner.kvthresh * self.model_runner.memory_manager.get_num_free_pages())

        self.mp_alive[self.pp_rank] = 1

        logger.info(f'Init')

    def recv_schedule_seqs(self):
        seqs = self.comm.recv_schedule()
        if seqs is not None:
            self.schedule_queue.append(
                InputData(seqs, self.model_runner.memory_manager))

    def recv_intermediate_data(self):
        if len(self.schedule_queue) != 0:
            input_data = self.schedule_queue.popleft()
            intermediate_data = recv_pp_data(
                self.get_pp_last_rank(), self.dtype,
                [input_data.tokens.shape[0], self.hidden_size], self.ret_residual)
            self.run_queue.append((input_data, intermediate_data))

    def forward(self):
        if len(self.run_queue) != 0:
            hidden_states = None
            residual = None
            if self.ret_residual:
                input_data, (hidden_states_future, residual_future,
                             hidden_states, residual) = self.run_queue[0]
                if not (hidden_states_future.is_completed() and residual_future.is_completed()):
                    return
            else:
                input_data, (hidden_states_future,
                             hidden_states) = self.run_queue[0]
                if not hidden_states_future.is_completed():
                    return

            self.run_queue.popleft()
            output = self.model_runner.step_once(
                input_data, hidden_states, residual)
            if self.pp_rank == self.pp_size - 1:
                self.comm.send_tokens(output)
            else:
                send_pp_data(output, self.get_pp_next_rank())

    def check_preempt(self, num_decode_tokens):
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
            logger.warning(
                'Try increase --ratio-free-pages or the performance is poor!')

    def schedule_naive(self):
        schedule_prefill_seqs = []
        schedule_decode_seqs = []

        num_tokens_budget = self.maxp

        self.check_preempt(min(num_tokens_budget, len(self.seqs_to_decode)))
        # decode
        for _ in range(num_tokens_budget):
            if len(self.seqs_to_decode) == 0:
                break
            seq = self.seqs_to_decode.pop()
            seq.to_compute_token_num = 1
            schedule_decode_seqs.append(seq)

        self.model_runner.memory_manager.pre_allocate_page(
            schedule_decode_seqs)

        num_tokens_budget -= len(schedule_decode_seqs)

        # prefill
        prefill_batched_token_nums = 0
        while len(self.seqs_to_prefill) != 0 and num_tokens_budget != 0:
            seq = self.seqs_to_prefill.popleft()
            self.model_runner.memory_manager.pre_allocate_page([seq])
            if len(seq.token_ids)-seq.computed_token_num <= num_tokens_budget:
                seq.to_compute_token_num = len(
                    seq.token_ids) - seq.computed_token_num
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
                log_info += ' cache_hit_rate: %5s %%' % (
                    '%.2f' % self.model_runner.memory_manager.get_cache_hit_rate())
                logger.info(log_info)
            else:
                logger.info(log_info)
        return schedule_prefill_seqs + schedule_decode_seqs

    def schedule(self):

        schedule_prefill_seqs = []
        schedule_decode_seqs = []

        # prefill
        prefill_token_budget = self.page_size * \
            max(self.get_num_free_pages()-self.num_kvthresh_pages, 0)
        if self.pp_size > 1 and prefill_token_budget != 0:
            self.update_num_wait_tokens()
            free_ratio = self.model_runner.memory_manager.get_memory_free()
            # a = ratio_threshold_free_pages
            # free_ratio in [1,a] | prefill_ratio in [1,0]
            prefill_ratio = (free_ratio - self.kvthresh) / (1-self.kvthresh)
            prefill_ratio = max(prefill_ratio, 0)
            prefill_token_budget = min(
                round(prefill_ratio * self.maxp),
                prefill_token_budget)
            prefill_token_budget = min(
                max(self.num_wait_tokens//self.iterp, self.minp), prefill_token_budget)
        else:
            prefill_token_budget = min(self.maxp, prefill_token_budget)
        prefill_batched_token_nums = 0
        while len(self.seqs_to_prefill) != 0 and prefill_token_budget != 0:
            seq = self.seqs_to_prefill.popleft()
            if isinstance(self.model_runner.memory_manager, PrefixMemoryManager) and seq.computed_token_num == 0:
                self.model_runner.memory_manager.pre_allocate_computed_page([
                                                                            seq])
            if len(seq.token_ids)-seq.computed_token_num <= prefill_token_budget:
                seq.to_compute_token_num = len(
                    seq.token_ids) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                prefill_token_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += prefill_token_budget
                seq.to_compute_token_num = prefill_token_budget
                prefill_token_budget = 0
            schedule_prefill_seqs.append(seq)

        self.model_runner.memory_manager.pre_allocate_page(
            schedule_prefill_seqs)

        # decode
        num_total_decode_seqs = self.get_num_decode_seqs()
        if num_total_decode_seqs < self.pp_size:
            decode_token_budget = num_total_decode_seqs
        else:
            # here we add num_total_decode_seqs to random.randint(0,self.pp_size-1))
            # because we want to solve the situation when #seqs=5 pp_size=4
            decode_token_budget = (
                num_total_decode_seqs + random.randint(0, self.pp_size-1)) // self.pp_size

        self.check_preempt(decode_token_budget)

        for _ in range(decode_token_budget):
            if len(self.seqs_to_decode) == 0:
                break
            seq = self.seqs_to_decode.popleft()
            seq.to_compute_token_num = 1
            schedule_decode_seqs.append(seq)

        self.model_runner.memory_manager.pre_allocate_page(
            schedule_decode_seqs)

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
                log_info += ' cache_hit_rate: %5s %%' % (
                    '%.2f' % self.model_runner.memory_manager.get_cache_hit_rate())
                logger.info(log_info)
            else:
                logger.info(log_info)
        # with open('log','a') as f:
        #     f.write(f'{prefill_batched_token_nums} {len(schedule_decode_seqs)}\n')
        return schedule_prefill_seqs + schedule_decode_seqs

    def recv_requests(self):
        ipc_package = self.comm.recv_requests()
        if ipc_package is not None:
            self.seqs_to_prefill.extend(ipc_package.schedule_lists)

    def recv_next_tokens(self):
        if self.pp_size != 1:  # recv tokens from last rank
            next_tokens = self.comm.recv_tokens()
            if next_tokens is not None:
                self.next_tokens_queue.append(next_tokens)

    def process_output(self):
        if len(self.next_tokens_queue) != 0:
            next_tokens = self.next_tokens_queue.popleft()

            schedule_seqs: List[Sequence] = self.batch_running.popleft()
            assert len(next_tokens) == len(schedule_seqs)

            send_tokens = []
            ipc_package = IPCPackage([])

            for idx, seq in enumerate(schedule_seqs):
                seq.computed_token_num += seq.to_compute_token_num
                if seq.computed_prompt():
                    ipc_package.act_schedule_ids.append(seq.seq_id)
                    send_tokens.append(next_tokens[idx])
                    seq.token_ids.append(next_tokens[idx])
                if seq.is_finish():
                    ipc_package.free_ids.append(seq.seq_id)
                    self.model_runner.memory_manager.free(seq)
                elif seq.computed_prompt():
                    self.seqs_to_decode.appendleft(seq)
                else:
                    self.seqs_to_prefill.appendleft(seq)

            self.comm.send_output((ipc_package, send_tokens))

    def schedule_forward(self):
        if len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0 and len(self.batch_running) < self.pp_size:
            schedule_seqs = self.schedule() if not self.use_naive_schedule else self.schedule_naive()
            if len(schedule_seqs) != 0:
                input_data = InputData(
                    schedule_seqs, self.model_runner.memory_manager)
                if self.pp_size > 1:
                    self.comm.send_schedule(schedule_seqs)
                self.batch_running.append(schedule_seqs)
                output = self.model_runner.step_once(input_data)

                if type(output) != list:
                    send_pp_data(output, self.get_pp_next_rank())
                else:
                    self.next_tokens_queue.append(output)

    def run_driver(self):
        self.recv_requests()
        self.recv_next_tokens()
        self.schedule_forward()
        self.process_output()

    def run_other(self):
        self.recv_schedule_seqs()
        self.recv_intermediate_data()
        self.forward()

    def handle_keyboardInterrupt(self):
        logger.info(f'Exit')
        dist.destroy_process_group()
        self.mp_alive[self.pp_rank] = -1

    def handle_exception(self, e):
        logger.error(e)
        traceback.print_exc()
        dist.destroy_process_group()
        self.mp_alive[self.pp_rank] = -1


def run_worker(worker: Worker):
    try:
        worker.init()
        while True:
            if worker.pp_rank == 0:
                worker.run_driver()
            else:
                worker.run_other()
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
