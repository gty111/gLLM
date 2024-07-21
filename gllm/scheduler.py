from typing import List, Dict

from gllm.sequence import Sequence
from gllm.model_runner import ModelRunner


class Scheduler:
    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.prompt_lists: List[Sequence] = []
        self.decode_lists: List[Sequence] = []
        self.finish_lists: Dict[int, Sequence] = {}

    def add_requests(self, requests: List[Sequence]):
        self.prompt_lists.extend(requests)

    def schedule(self):
        # print(
        #     f'#prompt:{len(self.prompt_lists)} #decode:{len(self.decode_lists)}  #finish:{len(self.finish_lists)}')
        finish_lists_each = []
        for seq in self.decode_lists:
            if seq.token_ids[-1] in [128001, 128009]:
                finish_lists_each.append(seq)
                self.finish_lists[seq.seq_id] = seq
                self.model_runner.model.free_kv_cache(seq)
        for seq in finish_lists_each:
            self.decode_lists.remove(seq)

        schedule_lists: List[Sequence] = []

        if len(self.prompt_lists) != 0:
            cu_seqs_len = 0
            for seq in self.prompt_lists:
                if cu_seqs_len + len(seq.token_ids) <= 4096:
                    cu_seqs_len += len(seq.token_ids)
                    self.decode_lists.append(seq)
                    schedule_lists.append(seq)
            for seq in schedule_lists:
                self.prompt_lists.remove(seq)
            return schedule_lists

        for seq in self.decode_lists:
            schedule_lists.append(seq)
        return schedule_lists
