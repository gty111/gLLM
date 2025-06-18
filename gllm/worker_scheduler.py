import time
import random

from collections import deque
from functools import reduce
from typing import List
from logger import logger

from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.frontend_scheduler import IPCPackage
from gllm.dist_utils import get_world_size


class WorkerScheduler():
    def __init__(self, pp_size, memory_manager:MemoryManager, use_naive_schedule, maxp, minp, iterp, page_size, kvthresh):
        self.pp_size = pp_size
        self.memory_manager = memory_manager
        self.use_naive_schedule = use_naive_schedule
        self.maxp = maxp
        self.minp = minp
        self.iterp = iterp 
        self.page_size = page_size
        self.kvthresh = kvthresh
        self.num_kvthresh_pages = int(self.kvthresh * self.memory_manager.get_num_free_pages())

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
        # abort ids
        self.abort_ids = set()
        
    def get_num_free_pages(self):
        return self.memory_manager.get_num_free_pages()

    def get_num_decode_seqs(self):
        num_decode_seqs = len(self.seqs_to_decode) + \
            reduce(lambda x, y: x+len(y), self.batch_running, 0)
        return num_decode_seqs

    def update_num_wait_tokens(self):
        self.num_wait_tokens = reduce(
            lambda x, y: x + len(y) - y.computed_token_num, self.seqs_to_prefill, 0)
    
    def add_abort_ids(self, abort_ids):
        self.abort_ids.update(abort_ids)
    
    def add_new_requests(self, seqs):
        self.seqs_to_prefill.extend(seqs)    
    
    def add_next_tokens(self, next_tokens):
        self.next_tokens_queue.append(next_tokens)
        
    def process_output(self):
        if len(self.next_tokens_queue) != 0:
            schedule_seqs: List[Sequence] = self.batch_running.popleft()
            next_tokens = self.next_tokens_queue.popleft()
            send_tokens = []
            ipc_package = IPCPackage([])

            for idx, seq in enumerate(schedule_seqs):
                seq.computed_token_num += seq.to_compute_token_num
                if seq.computed_prompt:
                    ipc_package.act_schedule_ids.append(seq.seq_id)
                    send_tokens.append(next_tokens[idx])
                    seq.append(next_tokens[idx])
                if seq.is_finish:
                    ipc_package.free_ids.append(seq.seq_id)
                    self.memory_manager.free(seq)
                elif seq.computed_prompt:
                    self.seqs_to_decode.appendleft(seq)
                else:
                    self.seqs_to_prefill.appendleft(seq)
            ipc_package.next_tokens = send_tokens
            return ipc_package
        else:
            return None
        
    def check_preempt(self, num_decode_tokens):
        preempt_seqs = []
        while self.get_num_free_pages() < num_decode_tokens and len(self.seqs_to_decode) != 0:
            seq_to_preempt = self.seqs_to_decode.popleft()
            self.memory_manager.free(seq_to_preempt)
            seq_to_preempt.preempt()
            preempt_seqs.append(seq_to_preempt)

        self.seqs_to_prefill.extendleft(preempt_seqs)

        self.num_preempt_seqs += len(preempt_seqs)
        if self.num_preempt_seqs - self.log_num_preempt_seqs >= 10:
            self.log_num_preempt_seqs = self.num_preempt_seqs
            logger.warning(f'#Preempted seqs: {self.num_preempt_seqs}')
            logger.warning(
                'Try increase --kvthresh or the performance is poor!')
    
    def check_abort_seqs_list(self, seqs:deque, ipc_package:IPCPackage):
        for seq in list(seqs):
            if len(self.abort_ids) == 0:
                break
            id = seq.seq_id
            if id in self.abort_ids:
                ipc_package.free_ids.append(id)
                self.memory_manager.free(seq)
                seqs.remove(seq)
                self.abort_ids.remove(id)
    
    def check_abort_seqs(self):
        ipc_package = IPCPackage([])
        self.check_abort_seqs_list(self.seqs_to_prefill, ipc_package)
        self.check_abort_seqs_list(self.seqs_to_decode, ipc_package)
        if len(ipc_package.free_ids) != 0:
            return ipc_package
        else:
            return None
    
    def schedule_once(self):
        if len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0 and len(self.batch_running) < self.pp_size:
            schedule_seqs = self.schedule() if not self.use_naive_schedule else self.schedule_naive()
            if len(schedule_seqs) != 0:
                self.batch_running.append(schedule_seqs)
            return schedule_seqs
        else: 
            return []

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

        self.memory_manager.pre_allocate_page(
            schedule_decode_seqs)

        num_tokens_budget -= len(schedule_decode_seqs)
        
        num_tokens_budget = min(num_tokens_budget, self.page_size * \
            max(self.get_num_free_pages()-self.num_kvthresh_pages, 0))

        # prefill
        prefill_batched_token_nums = 0
        while len(self.seqs_to_prefill) != 0 and num_tokens_budget != 0:
            seq = self.seqs_to_prefill.popleft()
            if len(seq)-seq.computed_token_num <= num_tokens_budget:
                seq.to_compute_token_num = len(seq) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                num_tokens_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += num_tokens_budget
                seq.to_compute_token_num = num_tokens_budget
                num_tokens_budget = 0
            schedule_prefill_seqs.append(seq)
            
        self.memory_manager.pre_allocate_page(
            schedule_prefill_seqs)

        if time.time()-self.log_time > 1:
            self.log_time = time.time()
            log_info = '#wait: %4d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%' % (
                len(self.seqs_to_prefill),
                self.get_num_decode_seqs(),
                prefill_batched_token_nums,
                len(schedule_decode_seqs),
                '%.2f' % self.memory_manager.get_memory_util())
            if isinstance(self.memory_manager, PrefixMemoryManager):
                log_info += ' cache_hit_rate: %5s %%' % (
                    '%.2f' % self.memory_manager.get_cache_hit_rate())
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
        if get_world_size() > 1 and prefill_token_budget != 0:
            self.update_num_wait_tokens()
            free_ratio = self.memory_manager.get_memory_free()
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
            if isinstance(self.memory_manager, PrefixMemoryManager) and seq.computed_token_num == 0:
                self.memory_manager.pre_allocate_computed_page([seq])
            if len(seq)-seq.computed_token_num <= prefill_token_budget:
                seq.to_compute_token_num = len(seq) - seq.computed_token_num
                prefill_batched_token_nums += seq.to_compute_token_num
                prefill_token_budget -= seq.to_compute_token_num
            else:
                prefill_batched_token_nums += prefill_token_budget
                seq.to_compute_token_num = prefill_token_budget
                prefill_token_budget = 0
            schedule_prefill_seqs.append(seq)

        self.memory_manager.pre_allocate_page(
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

        self.memory_manager.pre_allocate_page(
            schedule_decode_seqs)

        if time.time()-self.log_time > 1:
            self.log_time = time.time()
            log_info = '#wait: %4d/%8d #run: %4d #prefill: %4d #decode: %4d memory_util: %5s %%' % (
                len(self.seqs_to_prefill),
                self.num_wait_tokens,
                num_total_decode_seqs,
                prefill_batched_token_nums,
                len(schedule_decode_seqs),
                '%.2f' % self.memory_manager.get_memory_util())
            if isinstance(self.memory_manager, PrefixMemoryManager):
                log_info += ' cache_hit_rate: %5s %%' % (
                    '%.2f' % self.memory_manager.get_cache_hit_rate())
                logger.info(log_info)
            else:
                logger.info(log_info)
        # with open('log','a') as f:
        #     f.write(f'{prefill_batched_token_nums} {len(schedule_decode_seqs)}\n')
        return schedule_prefill_seqs + schedule_decode_seqs
