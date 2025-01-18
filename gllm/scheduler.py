import time
from logger import logger
from typing import List

from gllm.sequence import Sequence


class SchedulerOutput:
    def __init__(self, schedule_lists: List[Sequence]):
        self.computed_prompt = schedule_lists[0].computed_prompt
        self.schedule_lists = schedule_lists


class Scheduler:
    def __init__(self, max_decode_seqs: int, max_batch_tokens: int, ratio_threshold_free_pages: float,
                 total_num_free_pages: int, finish_tokens: List[int], page_size: int) -> None:
        self.prompt_lists: List[Sequence] = []  # seqs to prefill
        self.decode_lists: List[Sequence] = []  # seqs to decode
        self.finish_lists: List[Sequence] = []  # seqs finished

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

    def schedule(self, num_free_pages: int, iflog: bool = False):
        self.num_free_pages = num_free_pages
        schedule_lists: List[Sequence] = []

        # prompt
        if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages):
            cu_seqs_len = 0
            for seq in self.prompt_lists:
                if cu_seqs_len + len(
                        seq.token_ids) <= self.max_batch_tokens and self.num_free_pages > self.num_threshold_free_pages:
                    cu_seqs_len += len(seq.token_ids)
                    schedule_lists.append(seq)
                    self.num_free_pages -= len(seq.token_ids) // self.page_size
            for seq in schedule_lists:
                self.prompt_lists.remove(seq)
            self.num_schedule_prefill += 1

        # decode
        if len(schedule_lists) == 0:
            decode_batch_size = min(
                self.max_decode_seqs, self.num_free_pages, len(self.decode_lists))
            schedule_lists = self.decode_lists[:decode_batch_size]
            self.decode_lists = self.decode_lists[decode_batch_size:]

            self.num_schedule_decode += 1

        assert len(
            schedule_lists) != 0 and "Try to increase ratio_threshold_free_pages"

        cur_time = time.time()
        if iflog and cur_time - self.log_time > 2:
            self.log_time = cur_time
            if not schedule_lists[0].computed_prompt:
                print('prefill: ', end='')
            else:
                print('decode : ', end='')
            print(
                '#schedule: %4d #prompt: %4d #decode: %4d memory_util: %2.2f %%'
                % (len(schedule_lists),
                   len(self.prompt_lists),
                   len(self.decode_lists),
                   self.get_memory_util()))
        # if not schedule_lists[0].computed_prompt:
        #     print('prefill: ', end='')
        # else:
        #     print('decode : ', end='')
        return SchedulerOutput(schedule_lists)

    def update_seqs(self, scheduleOutput: SchedulerOutput):
        for seq in scheduleOutput.schedule_lists:
            if seq.is_finish():
                self.finish_lists.append(seq)
            else:
                self.decode_lists.append(seq)
        if not scheduleOutput.computed_prompt:
            self.num_schedule_prefill -= 1
        else:
            self.num_schedule_decode -= 1

    def delay_schedule(self):
        '''
        Used for situations :
        One schedule 128 seqs and the other schedule 8 seqs
        we can merge two schedule to one which contains 136 seqs
        '''
        return self.num_schedule_decode != 0 and not self.can_schedule_prefill() and len(self.decode_lists) < 32

    def can_schedule_prefill(self):
        return len(self.prompt_lists) != 0 and self.num_free_pages > self.num_threshold_free_pages

    def has_scheduled_seqs(self):
        return self.can_schedule_prefill() or len(self.decode_lists) != 0

    def has_seqs(self):
        return len(self.prompt_lists) + len(self.decode_lists) != 0 or (self.num_schedule_decode + self.num_schedule_prefill) != 0

    def get_memory_util(self):
        return round((self.total_num_free_pages - self.num_free_pages)*100 / self.total_num_free_pages, 2)
