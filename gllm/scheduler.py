from typing import List, Dict

from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner


class Scheduler:
    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.prompt_lists: List[Sequence] = []
        self.decode_lists: List[Sequence] = []
        self.finish_lists: Dict[int, Sequence] = {}
        
        self.max_decode_seqs = 256

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self):
        schedule_lists: List[Sequence] = []

        # prompt
        if len(self.prompt_lists) != 0 and (
                self.model_runner.memory_manager.get_num_free_pages() > 1024 and len(self.decode_lists) < self.max_decode_seqs):
            cu_seqs_len = 0
            for seq in self.prompt_lists:
                if cu_seqs_len + len(seq.token_ids) <= self.model_runner.model.max_model_len:
                    cu_seqs_len += len(seq.token_ids)
                    self.decode_lists.append(seq)
                    schedule_lists.append(seq)
            for seq in schedule_lists:
                self.prompt_lists.remove(seq)

        # decode
        if len(schedule_lists) == 0:
            # set max batch size
            for seq in self.decode_lists[:self.max_decode_seqs]:
                schedule_lists.append(seq)

        # print(
        #     f'#schedule:{len(schedule_lists)} '
        #     f'#prompt:{len(self.prompt_lists)} '
        #     f'#decode:{len(self.decode_lists)} '
        #     f'#finish:{len(self.finish_lists)} '
        #     f'memory_util:{self.model_runner.memory_manager.get_memory_util()} %')
        return schedule_lists

    def update_finish_seqs(self):
        # check finished seqs
        finish_lists = []
        for seq in self.decode_lists:
            if seq.token_ids[-1] in self.model_runner.model.finish_tokens or len(seq.token_ids) - seq.prompt_len >= seq.output_len:
                finish_lists.append(seq)
                self.finish_lists[seq.seq_id] = seq
                self.model_runner.free_kv_cache(seq)
        for seq in finish_lists:
            self.decode_lists.remove(seq)
