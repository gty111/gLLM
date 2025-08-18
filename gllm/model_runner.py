import torch

from typing import Union, Dict
from transformers import (AutoTokenizer, PreTrainedTokenizer, 
                          PreTrainedTokenizerFast, AutoProcessor,
                          AutoImageProcessor)

from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.dist_utils import (is_output_rank, get_tp_size, is_last_pp_rank,
                             is_first_pp_rank)
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
        
        self.use_mm = self.model_loader.use_mm
        
        if self.use_mm:
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            self.image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)

        # lazy init
        self.model = None
        self.memory_manager = None

    def init(self, mp_load_progress=None):
        self.model = self.model_loader.load_model(mp_load_progress)
        memory_manager_cls = PrefixMemoryManager if self.enable_prefix_caching else MemoryManager
        self.memory_manager = memory_manager_cls(
            gpu_memory_util=self.gpu_memory_util, num_layers=self.model.num_layers,
            dtype=self.model_loader.dtype, page_size=self.page_size, kv_head_num=self.model.num_kv_heads//get_tp_size(),
            kv_head_dim=self.model.head_dim, vocab_size=self.model_loader.vocab_size,
            use_mla=self.model_loader.use_mla)

    def encode(self, messages, chat: bool = False, has_mm: bool = False):
        if chat:
            if not self.use_mm or not has_mm:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    enable_thinking=self.use_thinking)
            else:
                return self.processor.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True)[0]
        else:
            return self.tokenizer.encode(messages)
        
    def decode(self, content):
        return self.tokenizer.decode(content, True, True)
        
    @torch.inference_mode()
    def step_once(self, input_data: InputData = None, hidden_states=None, 
                  residual=None, input_embeddings=None):
        output = self.model(input_data, hidden_states, residual, input_embeddings)
        if is_last_pp_rank():
            logits = self.model.compute_logits(input_data, output)
            if is_output_rank():
                next_tokens = self.sampler.forward(logits, input_data)
                return next_tokens
            return output
        else:
            return output

    def free(self, seq: Sequence):
        self.memory_manager.free(seq)
