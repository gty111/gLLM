import torch
import time
from transformers import AutoTokenizer
from typing import List

from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.memory_manager import MemoryManager


class ModelRunner():
    def __init__(self, model_path: str, gpu_memory_utilization, page_size):
        model_loader = ModelLoader(model_path)

        self.model = model_loader.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.memory_manager = MemoryManager(
            gpu_memory_utilization, self.model.num_layers, self.model.dtype, page_size, self.model.num_kv_heads, self.model.head_dim)

    @torch.inference_mode()
    def step_once(self,
                  seqs: List[Sequence], temperature, top_p):
        input_data = InputData(seqs, self.memory_manager)
        hidden_states = self.model(input_data)
        logits = self.model.compute_logits(input_data, hidden_states)
        next_tokens = self.model.sample(
            logits, temperature, top_p)
        for idx,seq in enumerate(seqs):
            if not seq.computed_prompt:
                seq.computed_prompt = True
            seq.token_ids.append(next_tokens[idx])
            if seq.is_finish():
                self.free_kv_cache(seq)
        assert len(next_tokens) == len(seqs)
            

    def free_kv_cache(self, seq: Sequence):
        self.memory_manager.free(seq)

    def stream_inference(self, seq: Sequence, temperature: float, top_p: float):
        # -------prefill------
        prefill_start = time.time()
        self.step_once([seq], temperature, top_p)
        prefill_end = time.time()
        # ----prefill end-----

        # ------decode-------
        decode_start = time.time()
        while True:
            print(seq.detokenize_inc(self.tokenizer), end='', flush=True)
            if seq.is_finish():
                break
            self.step_once([seq], temperature, top_p)
        print("\n")
        decode_end = time.time()
        # ------decode end-------
        # ------metric---------
        elapsed_decode_time = decode_end - decode_start
        elapsed_prefill_time = prefill_end - prefill_start
        elapsed_time = round(elapsed_decode_time+elapsed_prefill_time, 2)
        prefill_rate = round(
            seq.prompt_len / elapsed_prefill_time, 2)
        decode_rate = round(
            len(seq.token_ids[seq.prompt_len:]) / elapsed_decode_time, 2)
        print(
            f"#input: {seq.prompt_len} #output: {len(seq.token_ids[seq.prompt_len:])} elapsed time: {elapsed_time} s rate(prefill/decode): {prefill_rate}/{decode_rate} toks/s")
        # -----metric end--------
        return self.tokenizer.decode(seq.token_ids[seq.prompt_len:], True, True)
