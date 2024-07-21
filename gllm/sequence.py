from typing import List

class Sequence():
    def __init__(self, seq_id, token_ids):
        self.seq_id = seq_id
        self.token_ids: List[int] = token_ids
        self.prompt_len = len(token_ids)
        self.segment_id = 0
        self.page_table = []
        self.computed_prompt = False
        self.prompt = ''
        self.output = ''