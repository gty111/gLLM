import time

from logger import logger
from typing import List

from gllm.sequence import Sequence


class SchedulerOutput:
    def __init__(self, schedule_lists: List[Sequence]):
        if not len(schedule_lists) == 0:
            self.computed_prompt = schedule_lists[0].computed_prompt
        else:
            self.computed_prompt = True
        self.schedule_lists = schedule_lists

# Only used for decode
class DeltaSchedulerOutput:
    def __init__(self, free_indices: List[int], delta_schedule_list: List[int]):
        self.free_indices = free_indices # gpu process => schedule process
        self.delta_schedule_list = delta_schedule_list # schedule process => gpu process


class Scheduler:
    def __init__(self, max_decode_seqs: int, max_batch_tokens: int, ratio_threshold_free_pages: float,
                 page_size: int) -> None:
        self.prompt_lists: List[Sequence] = []  # seqs to prefill
        self.decode_lists: List[Sequence] = []  # seqs to decode
        self.finish_lists: List[Sequence] = []  # seqs finished

        self.decode_batch = SchedulerOutput([])
        self.delta_scheduler_output = DeltaSchedulerOutput([], [])

        self.max_decode_seqs = max_decode_seqs
        self.max_batch_tokens = max_batch_tokens
        self.ratio_threshold_free_pages = ratio_threshold_free_pages
        self.num_threshold_free_pages = None

        # only used for delta
        self.num_schedule_prefill = 0
        self.schedule_decode = True
        
        self.total_num_free_pages = None
        self.num_free_pages = None
        self.page_size = page_size

        self.log_time = time.time()
        
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
                   len(self.decode_lists) + len(self.decode_batch.schedule_lists),
                   self.get_memory_util()))

        self.num_free_pages = num_free_pages

        # prompt
        if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages) and (not delta or delta and self.num_schedule_prefill <= 1):
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
            if not self.schedule_decode:
                return DeltaSchedulerOutput([],[])
            assert self.num_free_pages*self.page_size > len(self.decode_batch.schedule_lists)

            delta_batch_size = min(self.num_free_pages*self.page_size, self.max_decode_seqs -
                                len(self.decode_batch.schedule_lists), len(self.decode_lists))
            self.delta_scheduler_output.delta_schedule_list = self.decode_lists[:delta_batch_size]
            self.decode_batch.schedule_lists.extend(
                self.delta_scheduler_output.delta_schedule_list)
            self.decode_lists = self.decode_lists[delta_batch_size:]

            assert (len(self.decode_batch.schedule_lists) +
                    delta_batch_size != 0 or self.num_schedule_prefill) and "Try to increase ratio_threshold_free_pages"
            # we only schedule decode when GPU has stepped once on last decode batch
            if not len(self.delta_scheduler_output.delta_schedule_list) == 0:
                self.schedule_decode = False 
            return self.delta_scheduler_output


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
                for idx in schedulerOutput.free_indices:
                    self.finish_lists.append(schedulerOutput.schedule_lists[idx])
                keep_indices = list(set(range(len(schedulerOutput.schedule_lists)))-set(schedulerOutput.free_indices))
                for idx in keep_indices:
                    self.decode_lists.append(schedulerOutput.schedule_lists[idx])
                self.num_schedule_prefill -= 1
            elif isinstance(schedulerOutput, DeltaSchedulerOutput):  # decode
                self.schedule_decode = True
                for i in schedulerOutput.free_indices:
                    self.finish_lists.append(self.decode_batch.schedule_lists[i])
                keep_indices = list(set(range(len(self.decode_batch.schedule_lists)))-set(schedulerOutput.free_indices))
                self.decode_batch.schedule_lists = [
                    self.decode_batch.schedule_lists[i] for i in keep_indices]
                keep_indices_token = list(set(range(len(next_tokens)))-set(schedulerOutput.free_indices))
                for i,idx in enumerate(keep_indices_token):
                    self.decode_batch.schedule_lists[i].token_ids.append(next_tokens[idx])

    def can_schedule_prefill(self):
        return len(self.prompt_lists) != 0 and self.num_free_pages > self.num_threshold_free_pages

    def has_scheduled_seqs(self):
        return self.can_schedule_prefill() or len(self.decode_lists) + len(self.decode_batch.schedule_lists) != 0

    def has_seqs(self):
        return len(self.prompt_lists) + len(self.decode_lists) + len(self.decode_batch.schedule_lists) != 0 or self.num_schedule_prefill != 0

    def get_memory_util(self):
        return round((self.total_num_free_pages - self.num_free_pages)*100 / self.total_num_free_pages, 2)
