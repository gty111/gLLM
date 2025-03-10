import time

from logger import logger
from typing import List

from gllm.sequence import Sequence


class SchedulerOutput:
    def __init__(self, schedule_lists: List[Sequence]):
        self.schedule_lists = schedule_lists # schedule process => gpu process
        self.free_ids = [] # gpu process => schedule process
        self.act_schedule_ids = [] # gpu process => schedule process

# Only used for decode
class DeltaSchedulerOutput:
    def __init__(self, free_ids: List[int]):
        self.free_ids = free_ids # gpu process => schedule process
        self.act_schedule_ids = [] # gpu process => schedule process

class Scheduler:
    def __init__(self, max_decode_seqs: int, max_batch_tokens: int, ratio_threshold_free_pages: float,
                 page_size: int, pp_size:int) -> None:
        self.prompt_lists: List[Sequence] = []  # seqs to prefill
        self.decode_lists: List[Sequence] = []  # seqs to decode
        self.finish_lists: List[Sequence] = []  # seqs finished

        self.prefill_batch = dict() # seqs under prefill (seq_id => seq)
        self.decode_batch = dict() # seqs under decoding (seq_id => seq)

        self.max_decode_seqs = max_decode_seqs
        self.max_batch_tokens = max_batch_tokens
        self.ratio_threshold_free_pages = ratio_threshold_free_pages
        self.num_threshold_free_pages = None

        # only used for delta
        self.num_schedule_prefill = 0
        
        self.total_num_free_pages = None
        self.num_free_pages = None
        self.page_size = page_size

        self.log_time = time.time()
        
        self.pp_size = pp_size
        
    def set_total_num_free_pages(self, total_num_free_pages):
        self.num_free_pages = total_num_free_pages
        self.total_num_free_pages = total_num_free_pages
        self.num_threshold_free_pages = int(total_num_free_pages * self.ratio_threshold_free_pages)

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self, num_free_pages: int, log: bool = False, delta: bool = False):
        # log
        cur_time = time.time()
        if log and cur_time - self.log_time > 1:
            self.log_time = cur_time
            logger.info(
                '#wait: %4d #decode: %4d memory_util: %2.2f %%'
                % (len(self.prompt_lists),
                   len(self.decode_lists) + len(self.decode_batch),
                   self.get_memory_util()))

        self.num_free_pages = num_free_pages

        # prompt
        if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages) and (
                    not delta or delta and self.num_schedule_prefill <= self.pp_size + 1): # plus 1 can overlap schedule and execution
            prefill_schedule_lists: List[Sequence] = []
            cu_seqs_len = 0
            for seq in self.prompt_lists:
                if cu_seqs_len + len(
                        seq.token_ids) <= self.max_batch_tokens and self.num_free_pages > self.num_threshold_free_pages:
                    cu_seqs_len += len(seq.token_ids)
                    prefill_schedule_lists.append(seq)
                    self.num_free_pages -= len(seq.token_ids) // self.page_size
            for seq in prefill_schedule_lists:
                self.prompt_lists.remove(seq)
                self.prefill_batch[seq.seq_id] = seq
            self.num_schedule_prefill += 1
            return SchedulerOutput(prefill_schedule_lists)

        # decode
        if not delta:
            decode_batch_size = min(
                self.max_decode_seqs, self.num_free_pages*self.page_size, len(self.decode_lists))
            schedule_lists = self.decode_lists[:decode_batch_size]
            self.decode_lists = self.decode_lists[decode_batch_size:]
            return SchedulerOutput(schedule_lists)
        else:
            return DeltaSchedulerOutput([])


    def update_seqs(self, schedulerOutput:SchedulerOutput, next_tokens: List[int]=None, delta=False):
        if not delta:
            for idx,seq in enumerate(schedulerOutput.schedule_lists):
                seq.token_ids.append(next_tokens[idx])
                if not seq.computed_prompt:
                    seq.computed_prompt = True
                if seq.is_finish():
                    self.finish_lists.append(seq)
                else:
                    self.decode_lists.append(seq)
        else:
            if isinstance(schedulerOutput, SchedulerOutput):  # prefill
                for idx, id in enumerate(schedulerOutput.act_schedule_ids):
                    seq:Sequence = self.prefill_batch.pop(id)
                    seq.token_ids.append(next_tokens[idx])
                    seq.computed_prompt = True
                    if id not in schedulerOutput.free_ids:
                        self.decode_batch[id] = seq
                self.num_schedule_prefill -= 1
            elif isinstance(schedulerOutput, DeltaSchedulerOutput):  # decode
                for i in schedulerOutput.free_ids:
                    self.decode_batch.pop(i)
                for idx,i in enumerate(schedulerOutput.act_schedule_ids):
                    if i in self.decode_batch:
                        self.decode_batch[i].token_ids.append(next_tokens[idx])
            else:
                assert 0
            self.finish_lists.extend(schedulerOutput.free_ids)

    def can_schedule_prefill(self):
        return len(self.prompt_lists) != 0 and self.num_free_pages > self.num_threshold_free_pages

    def has_scheduled_seqs(self):
        return self.can_schedule_prefill() or len(self.decode_lists) + len(self.decode_batch) != 0

    def has_seqs(self):
        return len(self.prompt_lists) + len(self.decode_lists) + len(self.decode_batch) != 0 or self.num_schedule_prefill != 0

    def get_memory_util(self):
        return round((self.total_num_free_pages - self.num_free_pages)*100 / self.total_num_free_pages, 2)
