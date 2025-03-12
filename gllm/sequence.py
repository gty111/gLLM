from typing import List, Optional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class Sequence():
    def __init__(self, seq_id, token_ids, finish_tokens, output_len=None, ignore_eos=False,
                 temperature=0.6, top_p=0.9, top_k=10):
        self.seq_id = seq_id
        self.token_ids: List[int] = token_ids
        self.prompt_len = len(token_ids)
        self.page_table = []
        self.prompt = ''
        self.output = ''
        self.ignore_eos = ignore_eos
        self.finish_tokens: List[int] = finish_tokens
        # maximum output length
        if output_len is None:
            self.output_len = 4096
        else:
            self.output_len = output_len
        # used for detokenize
        self.cur_length = self.prompt_len
        # used for sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # used for prefix cache and chunked prefill
        self.computed_token_num = 0
        self.to_compute_token_num = 0

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

    def is_finish(self):
        if (not self.ignore_eos and self.token_ids[-1] in self.finish_tokens
                ) or len(self.token_ids) - self.prompt_len >= self.output_len:
            return True
        else:
            return False
