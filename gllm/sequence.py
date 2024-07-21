from typing import List

class Sequence():
    def __init__(self, seq_id, token_ids, output_len=None):
        self.seq_id = seq_id
        self.token_ids: List[int] = token_ids
        self.prompt_len = len(token_ids)
        self.segment_id = 0
        self.page_table = []
        self.computed_prompt = False
        self.prompt = ''
        self.output = ''
        # maximum output length
        if output_len is None:
            self.output_len = 4096 
        else:
            self.output_len = output_len