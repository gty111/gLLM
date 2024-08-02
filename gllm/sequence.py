from typing import List, Optional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


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
        # used for detokenize
        self.cur_length = self.prompt_len

    def detokenize_inc(self, tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast]):
        added_space = ' ' if ' ' in tokenizer.decode(
            self.token_ids[self.cur_length-1:self.cur_length+1], True, True).strip() else ''
        delta_text = tokenizer.decode(
            self.token_ids[self.cur_length:], True, True)
        if delta_text.endswith('�'):
            return ''
        if len(delta_text) > 0 and delta_text[0] != ' ':
            delta_text = added_space + delta_text
        self.cur_length = len(self.token_ids)
        return delta_text
