import torch
import time

from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.dist_utils import is_output_rank, get_tp_size, is_last_pp_rank
from gllm.layers.sampler import Sampler


class ModelRunner():
    def __init__(self, load_format: str, model_path: str, gpu_memory_util: float, page_size: int,
                 enable_prefix_caching: bool, use_thinking: bool, maxp, maxd, kvthresh, minp, iterp):
        self.model_path = model_path
        self.model_loader = ModelLoader(load_format, model_path)
        self.enable_prefix_caching = enable_prefix_caching
        self.use_thinking = use_thinking
        self.gpu_memory_util = gpu_memory_util
        self.page_size = page_size
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.maxp = maxp
        self.maxd = maxd
        self.kvthresh = kvthresh
        self.minp = minp
        self.iterp = iterp
        self.sampler = Sampler()

        # lazy init
        self.model = None
        self.memory_manager = None

    def init(self, mp_load_progress=None):
        self.model = self.model_loader.load_model(mp_load_progress)
        memory_manager_cls = PrefixMemoryManager if self.enable_prefix_caching else MemoryManager
        self.memory_manager = memory_manager_cls(
            gpu_memory_util=self.gpu_memory_util, num_layers=self.model.num_layers,
            dtype=self.model_loader.dtype, page_size=self.page_size, kv_head_num=self.model.num_kv_heads//get_tp_size(),
            kv_head_dim=self.model.head_dim, vocab_size=self.model_loader.vocab_size)

    def encode(self, content, chat: bool = False):
        if chat:
            return self.tokenizer.apply_chat_template(content, add_generation_prompt=True, enable_thinking=self.use_thinking)
        else:
            return self.tokenizer.encode(content)
        
    def decode(self, content):
        return self.tokenizer.decode(content, True, True)

    @torch.inference_mode()
    def step_once(self, input_data: InputData = None, hidden_states=None, residual=None):
        output = self.model(input_data, hidden_states, residual)
        if is_last_pp_rank():
            logits = self.model.compute_logits(input_data, output)
            if is_output_rank():
                next_tokens = self.sampler.forward(logits, input_data)
                return next_tokens
            return output
        else:
            return output

    def free_kv_cache(self, seq: Sequence):
        self.memory_manager.free(seq)

    def stream_inference(self, seq: Sequence):
        # -------prefill------
        prefill_start = time.time()
        if isinstance(self.memory_manager, PrefixMemoryManager):
            self.memory_manager.pre_allocate_computed_page([seq])
        seq.to_compute_token_num = len(seq) - seq.computed_token_num
        self.memory_manager.pre_allocate_page([seq])
        next_token = self.step_once(InputData([seq], self.memory_manager))[0]
        seq.append(next_token)
        seq.computed_token_num += seq.to_compute_token_num
        seq.to_compute_token_num = 1
        prefill_end = time.time()
        # ----prefill end-----

        # ------decode-------
        decode_start = time.time()
        while True:
            print(seq.detokenize_inc(self.tokenizer), end='', flush=True)
            if seq.is_finish:
                break
            self.memory_manager.pre_allocate_page([seq])
            next_token = self.step_once(
                InputData([seq], self.memory_manager))[0]
            seq.computed_token_num += seq.to_compute_token_num
            seq.append(next_token)
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
            len(seq[seq.prompt_len:]) / elapsed_decode_time, 2)
        print(
            f'#input: {seq.prompt_len} #output: {len(seq[seq.prompt_len:])} '
            f'elapsed time: {elapsed_time} s rate(prefill/decode): {prefill_rate}/{decode_rate} toks/s\n')
        # -----metric end--------
        return self.decode(seq[seq.prompt_len:])
