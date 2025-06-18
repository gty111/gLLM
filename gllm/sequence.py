from typing import List, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class Sequence():
    def __init__(self, seq_id, token_ids, finish_tokens, output_len=None, ignore_eos=False,
                 temperature=0.6, top_p=0.9, top_k=10, repetition_penalty=1.0):
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
        self.repetition_penalty = repetition_penalty
        # used for prefix cache and chunked prefill
        self.computed_token_num = 0
        self.to_compute_token_num = 0
        # used for abort
        self.is_abort = False
        
    def __len__(self):
        return len(self.token_ids)
    
    def __getitem__(self, key):
        return self.token_ids[key]
    
    def append(self, token_id):
        self.token_ids.append(token_id)

    def detokenize_inc(self, tokenizer: Union[PreTrainedTokenizer | PreTrainedTokenizerFast]):
        added_space = ' ' if ' ' in tokenizer.decode(
            self[self.cur_length-1:self.cur_length+1], True, True).strip() else ''
        delta_text = tokenizer.decode(
            self[self.cur_length:], True, True)
        if delta_text.endswith('�'):
            return ''
        if len(delta_text) > 0 and delta_text[0] != ' ':
            delta_text = added_space + delta_text
        self.cur_length = len(self)
        return delta_text

    @property
    def is_finish(self):
        return (not self.ignore_eos and self[-1] in self.finish_tokens
                    ) or len(self) - self.prompt_len >= self.output_len
        
    def preempt(self):
        self.computed_token_num = 0
        self.page_table = []

    @property
    def computed_prompt(self):
        return self.computed_token_num >= self.prompt_len
    
    @property
    def seq_len(self):
        return self.computed_token_num + self.to_compute_token_num