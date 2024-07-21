import torch
import time
from transformers import AutoTokenizer
from typing import List

from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.memory_manager import MemoryManager

class ModelRunner():
    def __init__(self, model_path: str):
        model_loader = ModelLoader(model_path)

        self.model = model_loader.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.memory_manager = MemoryManager(32)

    def step_once(self,
                  seqs: List[Sequence]):
        if len(seqs) == 0:
            return
        with torch.no_grad():
            input_data = InputData(seqs, self.memory_manager)
            hidden_states = self.model(input_data)
            logits = self.model.compute_logits(input_data, hidden_states)
            next_tokens = self.model.sample(logits)
            assert len(next_tokens) == len(seqs)
            for i in range(len(next_tokens)):
                seqs[i].token_ids.append(next_tokens[i])
                if not input_data.computed_prompt:
                    seqs[i].computed_prompt = True
    
    def free_kv_cache(self, seq: Sequence):
        self.memory_manager.free(seq)

    def stream_inference(self, seq: Sequence):
        output_tokens = []
        # -------prefill------
        prefill_start = time.time()
        self.step_once([seq])
        seq.computed_prompt = True
        prefill_end = time.time()
        # ----prefill end-----

        # ------decode-------
        decode_start = time.time()
        while True:
            next_token = seq.token_ids[-1]
            output_tokens.append(next_token)
            print(self.tokenizer.decode(
                [next_token], True, True), end='', flush=True)
            if next_token in [128001, 128009]:
                break
            self.step_once([seq])
        print('\n')
        self.free_kv_cache(seq)
        decode_end = time.time()
        # ------decode end-------
        # ------metric---------
        elapsed_decode_time = decode_end - decode_start
        elapsed_prefill_time = prefill_end - prefill_start
        elapsed_time = round(elapsed_decode_time+elapsed_prefill_time, 2)
        prefill_rate = round(
            seq.prompt_len / elapsed_prefill_time, 2)
        decode_rate = round(
            len(output_tokens) / elapsed_decode_time, 2)
        print(
            f"#input: {seq.prompt_len} #output: {len(output_tokens)} elapsed time: {elapsed_time} s rate(prefill/decode): {prefill_rate}/{decode_rate} toks/s")
        # -----metric end--------
        return self.tokenizer.decode(output_tokens, True, True)
