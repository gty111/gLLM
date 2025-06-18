import time

from logger import logger
from typing import List

from gllm.sequence import Sequence
from gllm.memory_manager import MemoryManager, PrefixMemoryManager


class IPCPackage:
    def __init__(self, schedule_lists: List[Sequence]):
        # front-end => worker
        self.schedule_lists = schedule_lists
        self.abort_ids = [] # seq_ids to abort
        # worker => front-end
        self.free_ids = [] # seq_ids to free
        self.act_schedule_ids = []
        self.next_tokens = []

# Only used for LLM or AsyncLLM
class FrontendScheduler:
    def __init__(self, maxd: int, maxp: int, kvthresh: float,
                 page_size: int) -> None:
        self.prompt_lists: List[Sequence] = []  # seqs to prefill
        self.decode_lists: List[Sequence] = []  # seqs to decode
        self.finish_ids: List[Sequence] = []  # ids of finished seq

        self.max_decode_seqs = maxd
        self.max_batch_tokens = maxp
        self.kvthresh = kvthresh
        self.num_kvthresh_pages = None

        self.total_num_free_pages = None
        self.num_free_pages = None
        self.page_size = page_size

        self.log_time = time.time()

        self.preempt_num_seqs = 0
        self.log_preempt_num_seqs = 0

    def set_total_num_free_pages(self, total_num_free_pages):
        self.num_free_pages = total_num_free_pages
        self.total_num_free_pages = total_num_free_pages
        self.num_kvthresh_pages = int(
            total_num_free_pages * self.kvthresh)

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self, memory_manager: MemoryManager, log: bool = False):
        num_free_pages = memory_manager.get_num_free_pages()
        self.num_free_pages = num_free_pages

        # log
        cur_time = time.time()
        if log and cur_time - self.log_time > 1:
            self.log_time = cur_time
            logger.info(
                '#wait: %4d #run: %4d memory_util: %2.2f %%'
                % (len(self.prompt_lists),
                   len(self.decode_lists),
                   self.get_memory_util()))

        prefill_schedule_lists: List[Sequence] = []
        decode_schedule_lists: List[Sequence] = []

        # prompt
        cur_prefill_budget = len(decode_schedule_lists)
        if self.num_free_pages > self.num_kvthresh_pages:
            cu_seqs_len = 0
            for seq in self.prompt_lists:
                num_page = (len(seq) +
                            self.page_size-1) // self.page_size
                if cu_seqs_len + len(seq) <= self.max_batch_tokens and (
                        self.num_free_pages - num_page - cur_prefill_budget > self.num_kvthresh_pages):
                    cu_seqs_len += len(seq)
                    if isinstance(memory_manager, PrefixMemoryManager):
                        memory_manager.pre_allocate_computed_page([seq])
                    seq.to_compute_token_num = len(seq) - seq.computed_token_num
                    memory_manager.pre_allocate_page([seq])
                    prefill_schedule_lists.append(seq)
                    cur_prefill_budget += num_page
                else:
                    break
            for seq in prefill_schedule_lists:
                self.prompt_lists.remove(seq)

        # decode
        if len(prefill_schedule_lists) == 0:
            self.check_preempt_seqs(memory_manager)
            decode_batch_size = min(
                self.max_decode_seqs, self.num_free_pages*self.page_size, len(self.decode_lists))
            decode_schedule_lists = self.decode_lists[:decode_batch_size]
            self.decode_lists = self.decode_lists[decode_batch_size:]
            for seq in decode_schedule_lists:
                seq.to_compute_token_num = 1
            memory_manager.pre_allocate_page(decode_schedule_lists)

        return IPCPackage(prefill_schedule_lists+decode_schedule_lists)

    def update_seqs(self, ipc_package: IPCPackage, next_tokens: List[int], memory_manager: MemoryManager):
        for idx, seq in enumerate(ipc_package.schedule_lists):
            seq.append(next_tokens[idx])
            seq.computed_token_num += seq.to_compute_token_num
            if seq.is_finish:
                memory_manager.free(seq)
                self.finish_ids.append(seq.seq_id)
            else:
                self.decode_lists.append(seq)

    def check_preempt_seqs(self, memory_manager: MemoryManager):
        preempt_seqs = []
        while memory_manager.get_num_free_pages() < len(self.decode_lists):
            seq = self.decode_lists.pop()
            memory_manager.free(seq)
            preempt_seqs.append(seq)
        self.process_preempt(preempt_seqs)

    def process_preempt(self, preempt_seqs: List[Sequence] = None):
        for seq in preempt_seqs:
            seq.preempt()
        self.prompt_lists = preempt_seqs + self.prompt_lists
        self.preempt_num_seqs += len(preempt_seqs)
        if self.preempt_num_seqs - self.log_preempt_num_seqs > 10:
            logger.warning(f'#Preempted seqs: {self.preempt_num_seqs}')
            logger.warning(
                'Try increase --kvthresh or the performance is poor!')
            self.log_preempt_num_seqs = self.preempt_num_seqs

    def get_finish_ids(self):
        finish_ids = self.finish_ids
        self.finish_ids = []
        return finish_ids

    def has_seqs(self):
        return len(self.prompt_lists) + len(self.decode_lists) != 0

    def get_memory_util(self):
        return round((self.total_num_free_pages - self.num_free_pages)*100 / self.total_num_free_pages, 2)
