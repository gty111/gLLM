import torch
import time
from logger import logger

from transformers import AutoTokenizer

from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.dist_utils import get_pp_rank, get_pp_size


class ModelRunner():
    def __init__(self, load_format:str, model_path: str, gpu_memory_util:float, page_size:int, 
                 enable_prefix_caching:bool):
        self.model_path = model_path
        self.model_loader = ModelLoader(load_format, model_path)
        self.enable_prefix_caching = enable_prefix_caching
        if self.enable_prefix_caching:
            logger.info('Enable prefix caching')
        self.gpu_memory_util = gpu_memory_util
        self.page_size = page_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        
        # lazy init
        self.model = None
        self.memory_manager = None
    
    def init(self):
        self.model = self.model_loader.load_model()
        memory_manager_cls = PrefixMemoryManager if self.enable_prefix_caching else MemoryManager
        self.memory_manager = memory_manager_cls(
            gpu_memory_util=self.gpu_memory_util, num_layers=self.model.num_layers, 
            dtype=self.model.dtype, page_size=self.page_size, kv_head_num=self.model.num_kv_heads, 
            kv_head_dim=self.model.head_dim, vocab_size=self.model_loader.vocab_size)

    def tokenize(self, content, chat:bool=False):
        if chat:
            return self.tokenizer.apply_chat_template(content, add_generation_prompt=True)
        else:
            return self.tokenizer.encode(content)

    @torch.inference_mode()
    def step_once(self, input_data:InputData=None, hidden_states=None, residual=None):
        output = self.model(input_data, hidden_states, residual)
        if get_pp_rank() == get_pp_size() - 1:
            logits = self.model.compute_logits(input_data, output)
            next_tokens = self.model.sample(input_data, logits)
            return next_tokens
        else:
            return output

    def free_kv_cache(self, seq: Sequence):
        self.memory_manager.free(seq)

    def stream_inference(self, seq: Sequence):
        # -------prefill------
        prefill_start = time.time()
        next_token = self.step_once(InputData([seq],self.memory_manager))[0]
        seq.token_ids.append(next_token)
        seq.computed_prompt = True
        prefill_end = time.time()
        # ----prefill end-----

        # ------decode-------
        decode_start = time.time()
        while True:
            print(seq.detokenize_inc(self.tokenizer), end='', flush=True)
            if seq.is_finish():
                break
            next_token = self.step_once(InputData([seq], self.memory_manager))[0]
            seq.token_ids.append(next_token)
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
