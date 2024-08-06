from typing import List, Dict
from collections import deque

from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner


class Scheduler:
    def __init__(self, model_runner: ModelRunner, max_decode_seqs: int, max_batch_tokens: int, ratio_threshold_free_pages: float) -> None:
        self.model_runner = model_runner
        self.prompt_lists: List[Sequence] = [] # seqs to prefill
        self.decode_lists: deque[Sequence] = deque() # seqs to decode
        self.finish_lists: List[Sequence] = [] # seqs finished

        self.max_decode_seqs = max_decode_seqs
        self.max_batch_tokens = max_batch_tokens
        self.num_threshold_free_pages = int(
            self.model_runner.memory_manager.get_num_free_pages() * ratio_threshold_free_pages)
        
        self.num_schedule_running = 0
        self.total_num_free_pages = self.model_runner.memory_manager.get_num_free_pages()
        self.num_free_pages = self.model_runner.memory_manager.get_num_free_pages()
        self.page_size = self.model_runner.memory_manager.page_size

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self):
        schedule_lists: List[Sequence] = []

        # prompt
        if len(self.prompt_lists) != 0 and (
                self.num_free_pages > self.num_threshold_free_pages and len(schedule_lists) + len(self.decode_lists) < self.max_decode_seqs):
            cu_seqs_len = 0
            for seq in self.prompt_lists:
                if cu_seqs_len + len(
                    seq.token_ids) <= self.max_batch_tokens and self.num_free_pages > self.num_threshold_free_pages:
                    cu_seqs_len += len(seq.token_ids)
                    schedule_lists.append(seq)
                    self.num_free_pages -= (len(seq.token_ids)+self.page_size - 1) // self.page_size
            for seq in schedule_lists:
                self.prompt_lists.remove(seq)

        # decode
        if len(schedule_lists) == 0:
            decode_batch_size = min(
                self.max_decode_seqs, self.num_free_pages, len(self.decode_lists))
            for i in range(decode_batch_size):
                seq = self.decode_lists.popleft()
                if (len(seq.token_ids) - 1) % self.page_size == 0:
                    self.num_free_pages -= 1
                schedule_lists.append(seq)
            
        self.num_schedule_running += 1

        assert len(schedule_lists) != 0 and "Try to increase ratio_threshold_free_pages"

        # print(
        #     f'#schedule:{len(schedule_lists)} '
        #     f'#prompt:{len(self.prompt_lists)} '
        #     f'#decode:{len(self.decode_lists)} '
        #     f'#finish:{len(self.finish_lists)} '
        #     f'free_memory_util:{self.get_memory_util()} %')
        return schedule_lists

    def update_seqs(self,seqs:List[Sequence],next_tokens:List[int]):
        # append next token to each seq
        for i in range(len(next_tokens)):
            seqs[i].token_ids.append(next_tokens[i])
            if not seqs[i].computed_prompt:
                seqs[i].computed_prompt = True
        # check finished seqs
        for seq in seqs:
            if seq.token_ids[-1] in self.model_runner.model.finish_tokens or len(seq.token_ids) - seq.prompt_len >= seq.output_len:
                self.finish_lists.append(seq)
                self.num_free_pages += (len(seq.token_ids)+self.page_size-1) // self.page_size
            else:
                self.decode_lists.appendleft(seq)
        self.num_schedule_running -= 1

    def has_seqs_cur(self):
        return len(self.prompt_lists) + len(self.decode_lists) != 0 

    def has_seqs(self):
        return self.has_seqs_cur() or self.num_schedule_running == 1
    
    def get_memory_util(self):
        return round((self.total_num_free_pages-self.num_free_pages)*100 / self.total_num_free_pages,2)