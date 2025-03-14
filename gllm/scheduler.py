import time

from logger import logger
from typing import List, Dict
from collections import deque

from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager


class SchedulerOutput:
    def __init__(self, schedule_lists: List[Sequence],from_main_process=False, num_batched_tokens=None):
        self.num_batched_tokens = num_batched_tokens
        self.from_main_process = from_main_process
        self.schedule_lists = schedule_lists  # schedule process => gpu process
        self.free_ids = []  # gpu process => schedule process
        self.act_schedule_ids = []  # gpu process => schedule process
        
class PreemptOutput:
    def __init__(self, preempt_ids:List[int]):
        self.preempt_ids = preempt_ids

class Scheduler:
    def __init__(self, max_decode_seqs: int, max_batch_tokens: int, ratio_threshold_free_pages: float,
                 page_size: int, pp_size: int) -> None:
        self.prompt_lists: List[Sequence] = []  # seqs to prefill
        self.decode_lists: List[Sequence] = []  # seqs to decode
        self.finish_lists: List[Sequence] = []  # seqs finished

        self.run_batch:Dict[int,Sequence] = dict()  # seqs under running (seq_id => seq)
        
        self.max_decode_seqs = max_decode_seqs
        self.max_batch_tokens = max_batch_tokens
        self.ratio_threshold_free_pages = ratio_threshold_free_pages
        self.num_threshold_free_pages = None

        self.num_schedule = 0

        self.total_num_free_pages = None
        self.num_free_pages = None
        self.page_size = page_size

        self.log_time = time.time()

        self.pp_size = pp_size
        
        # Since we run multiple prefill schedule, worker may not allocate pages 
        # Use budget to track number of pages for prefill schedule on the fly
        self.total_prefill_budget = 0
        self.prefill_budget = deque()
        
        self.preempt_num_seqs = 0
        self.log_preempt_num_seqs = 0

    def set_total_num_free_pages(self, total_num_free_pages):
        self.num_free_pages = total_num_free_pages
        self.total_num_free_pages = total_num_free_pages
        self.num_threshold_free_pages = int(
            total_num_free_pages * self.ratio_threshold_free_pages)

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self, num_free_pages: int, log: bool = False, delta: bool = False):
        # log
        cur_time = time.time()
        if log and cur_time - self.log_time > 1:
            self.log_time = cur_time
            logger.info(
                '#wait: %4d #run: %4d memory_util: %2.2f %%'
                % (len(self.prompt_lists),
                   len(self.decode_lists) + len(self.run_batch),
                   self.get_memory_util()))

        self.num_free_pages = num_free_pages

        cur_prefill_budget = 0
        # prompt
        if delta:
            if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages) and self.num_schedule <= self.pp_size + 1:
                prefill_schedule_lists: List[Sequence] = []
                cu_seqs_len = 0
                for seq in self.prompt_lists:
                    num_page = (len(seq.token_ids)+self.page_size-1) // self.page_size
                    # dynamic adjust prefill batch tokens number according to memory utilization
                    if cu_seqs_len + len(seq.token_ids) <= self.max_batch_tokens*self.get_memory_free() and (
                        self.num_free_pages - num_page - cur_prefill_budget - self.total_prefill_budget > self.num_threshold_free_pages):
                        cu_seqs_len += len(seq.token_ids)
                        prefill_schedule_lists.append(seq)
                        cur_prefill_budget += num_page
                    else:
                        break
            
                if len(prefill_schedule_lists) != 0:
                    for seq in prefill_schedule_lists:
                        self.prompt_lists.remove(seq)
                        self.run_batch[seq.seq_id] = seq
                    self.prefill_budget.append(cur_prefill_budget)
                    self.total_prefill_budget += cur_prefill_budget
                    self.num_schedule += 1
                    return SchedulerOutput(prefill_schedule_lists, True, cu_seqs_len)
        else:
            if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages):
                prefill_schedule_lists: List[Sequence] = []
                cu_seqs_len = 0
                for seq in self.prompt_lists:
                    num_page = (len(seq.token_ids)+self.page_size-1) // self.page_size
                    if cu_seqs_len + len(seq.token_ids) <= self.max_batch_tokens and (
                        self.num_free_pages - num_page - cur_prefill_budget > self.num_threshold_free_pages):
                        cu_seqs_len += len(seq.token_ids)
                        prefill_schedule_lists.append(seq)
                        cur_prefill_budget += num_page
                    else:
                        break
                
                if len(prefill_schedule_lists) != 0:
                    for seq in prefill_schedule_lists:
                        self.prompt_lists.remove(seq)
                    return SchedulerOutput(prefill_schedule_lists)

        # decode
        if delta:
            # this scheduleOutput will not sent to worker
            # since worker can schedule by self
            return SchedulerOutput([])
        else:
            decode_batch_size = min(
                self.max_decode_seqs, self.num_free_pages*self.page_size, len(self.decode_lists))
            schedule_lists = self.decode_lists[:decode_batch_size]
            self.decode_lists = self.decode_lists[decode_batch_size:]
            return SchedulerOutput(schedule_lists)

    def update_seqs(self, schedulerOutput: SchedulerOutput, next_tokens: List[int] = None, 
                    delta=False, memory_manager: MemoryManager=None):
        if not delta:
            assert memory_manager is not None
            for idx, seq in enumerate(schedulerOutput.schedule_lists):
                seq.token_ids.append(next_tokens[idx])
                seq.computed_token_num += seq.to_compute_token_num
                seq.to_compute_token_num = 1
                if seq.is_finish():
                    memory_manager.free(seq)
                    self.finish_lists.append(seq.seq_id)
                else:
                    self.decode_lists.append(seq)
        else:
            if schedulerOutput.from_main_process:
                self.num_schedule -= 1
                self.total_prefill_budget -= self.prefill_budget.popleft()
            for idx, id in enumerate(schedulerOutput.act_schedule_ids):
                seq: Sequence = self.run_batch[id]
                if id in schedulerOutput.free_ids:
                    self.run_batch.pop(id)
                else:
                    seq.token_ids.append(next_tokens[idx])
            self.finish_lists.extend(schedulerOutput.free_ids)

    def process_preempt(self, preempt_ids:List[int]=None, preempt_seqs:List[Sequence]=None):
        _preempt_seqs = []
        if preempt_ids is not None:
            for id in preempt_ids:
                seq = self.run_batch.pop(id)
                _preempt_seqs.append(seq)
        else:
            assert preempt_seqs is not None
            _preempt_seqs = preempt_seqs
        for seq in _preempt_seqs:
            seq.preempt()
        self.prompt_lists = _preempt_seqs + self.prompt_lists
        self.preempt_num_seqs += len(_preempt_seqs)
        if self.preempt_num_seqs - self.log_preempt_num_seqs > 10:
            logger.warning(f'#Preempted seqs: {self.preempt_num_seqs}')
            logger.warning('Try increase --ratio-free-pages or the performance is poor!')
            self.log_preempt_num_seqs = self.preempt_num_seqs

    def can_schedule_prefill(self):
        return len(self.prompt_lists) != 0 and self.num_free_pages > self.num_threshold_free_pages

    def has_scheduled_seqs(self):
        return self.can_schedule_prefill() or len(self.decode_lists) + len(self.run_batch) != 0

    def has_seqs(self):
        return len(self.prompt_lists) + len(self.decode_lists) + len(self.run_batch) != 0 or self.num_schedule != 0

    def get_memory_util(self):
        return round((self.total_num_free_pages - self.num_free_pages)*100 / self.total_num_free_pages, 2)

    def get_memory_free(self):
        return self.num_free_pages / self.total_num_free_pages