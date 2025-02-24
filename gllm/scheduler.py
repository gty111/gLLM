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
    def __init__(self, free_indices: List[int], keep_indices: List[int], delta_schedule_list: List[int]):
        self.free_indices = free_indices
        self.keep_indices = keep_indices
        self.delta_schedule_list = delta_schedule_list


class Scheduler:
    def __init__(self, max_decode_seqs: int, max_batch_tokens: int, ratio_threshold_free_pages: float,
                 total_num_free_pages: int, finish_tokens: List[int], page_size: int) -> None:
        self.prompt_lists: List[Sequence] = []  # seqs to prefill
        self.decode_lists: List[Sequence] = []  # seqs to decode
        self.finish_lists: List[Sequence] = []  # seqs finished

        self.decode_batch = SchedulerOutput([])
        self.delta_scheduler_output = DeltaSchedulerOutput([], [], [])

        self.max_decode_seqs = max_decode_seqs
        self.max_batch_tokens = max_batch_tokens
        self.num_threshold_free_pages = int(
            total_num_free_pages * ratio_threshold_free_pages)

        self.num_schedule_prefill = 0
        self.num_schedule_decode = 0

        self.total_num_free_pages = total_num_free_pages
        self.num_free_pages = total_num_free_pages
        self.page_size = page_size

        self.finish_tokens = finish_tokens

        self.log_time = time.time()

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self, num_free_pages: int, log: bool = False, delta: bool = False):
        # log
        cur_time = time.time()
        if log and cur_time - self.log_time > 1:
            self.log_time = cur_time
            print(
                '#prompt: %4d #decode: %4d memory_util: %2.2f %%'
                % (len(self.prompt_lists),
                   len(self.decode_lists) + len(self.decode_batch.schedule_lists),
                   self.get_memory_util()))

        self.num_free_pages = num_free_pages

        # prompt
        if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages):
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
        self.num_schedule_decode += 1
        if not delta:
            decode_batch_size = min(
                self.max_decode_seqs, self.num_free_pages, len(self.decode_lists))
            schedule_lists = self.decode_lists[:decode_batch_size]
            self.decode_lists = self.decode_lists[decode_batch_size:]
            return SchedulerOutput(schedule_lists)
        else:
            assert self.num_free_pages > len(self.decode_batch.schedule_lists)

            delta_batch_size = min(self.num_free_pages, self.max_decode_seqs -
                                len(self.decode_batch.schedule_lists), len(self.decode_lists))
            self.delta_scheduler_output.delta_schedule_list = self.decode_lists[:delta_batch_size]
            self.decode_batch.schedule_lists.extend(
                self.delta_scheduler_output.delta_schedule_list)
            self.decode_lists = self.decode_lists[delta_batch_size:]

            assert (len(self.decode_batch.schedule_lists) +
                    delta_batch_size) != 0 and "Try to increase ratio_threshold_free_pages"
            return self.delta_scheduler_output


    def update_seqs(self, schedulerOutput:SchedulerOutput, next_tokens: List[int]=None, delta=False):
        if not delta:
            for idx,seq in enumerate(schedulerOutput.schedule_lists):
                if seq.is_finish():
                    self.finish_lists.append(seq)
                else:
                    self.decode_lists.append(seq)
            if schedulerOutput.computed_prompt:
                self.num_schedule_decode -= 1
            else:
                self.num_schedule_prefill -= 1
        else:
            if isinstance(schedulerOutput, SchedulerOutput):  # prefill
                self.num_schedule_prefill -= 1
                for idx in schedulerOutput.free_indices:
                    self.finish_lists.append(schedulerOutput.schedule_lists[idx])
                for idx in schedulerOutput.keep_indices:
                    self.decode_lists.append(schedulerOutput.schedule_lists[idx])
            elif isinstance(schedulerOutput, DeltaSchedulerOutput):  # decode
                self.num_schedule_decode -= 1
                for i in schedulerOutput.free_indices:
                    self.finish_lists.append(self.decode_batch.schedule_lists[i])
                self.decode_batch.schedule_lists = [
                    self.decode_batch.schedule_lists[i] for i in schedulerOutput.keep_indices]
                for i,idx in enumerate(schedulerOutput.keep_indices):
                    self.decode_batch.schedule_lists[i].token_ids.append(next_tokens[idx])

    def can_schedule_prefill(self):
        return len(self.prompt_lists) != 0 and self.num_free_pages > self.num_threshold_free_pages

    def has_scheduled_seqs(self):
        return self.can_schedule_prefill() or len(self.decode_lists) + len(self.decode_batch.schedule_lists) != 0

    def has_seqs(self):
        return len(self.prompt_lists) + len(self.decode_lists) + len(self.decode_batch.schedule_lists) != 0 or (
            self.num_schedule_decode + self.num_schedule_prefill) != 0

    def get_memory_util(self):
        return round((self.total_num_free_pages - self.num_free_pages)*100 / self.total_num_free_pages, 2)
