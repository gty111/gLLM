import torch

from attr import dataclass
from typing import Union, Dict
from transformers import (AutoTokenizer, PreTrainedTokenizer, 
                          PreTrainedTokenizerFast, AutoProcessor,
                          AutoImageProcessor)
from transformers.image_utils import load_images

from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.dist_utils import (is_output_rank, get_tp_size, is_last_pp_rank,
                             is_first_pp_rank)
from gllm.layers.sampler import Sampler
from gllm.layers.rotary_embedding import MRotaryEmbedding

@dataclass
class EmbeddingInfo:
    embedding: torch.Tensor = None
    positions: torch.Tensor = None
    mrope_position_delta:int = None

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
        
        # embedding cache: seq_id => embedding
        self.embedding_cache:Dict[int,EmbeddingInfo] = {}

    def init(self, mp_load_progress=None):
        self.model = self.model_loader.load_model(mp_load_progress)
        memory_manager_cls = PrefixMemoryManager if self.enable_prefix_caching else MemoryManager
        self.memory_manager = memory_manager_cls(
            gpu_memory_util=self.gpu_memory_util, num_layers=self.model.num_layers,
            dtype=self.model_loader.dtype, page_size=self.page_size, kv_head_num=self.model.num_kv_heads//get_tp_size(),
            kv_head_dim=self.model.head_dim, vocab_size=self.model_loader.vocab_size,
            use_mla=self.model_loader.use_mla)

    def encode(self, messages, chat: bool = False):
        if chat:
            if not self.use_mm:
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
    
    def extract_modify_mm(self, messages:Dict):
        mm_contents = []
        for message in messages:
            contents = message['content']
            for content in contents:
                if content['type'] == 'image':
                    mm_contents.append(content['image']) 
                elif content['type'] == 'image_url':
                    content['type'] = 'image'
                    data = content['image_url']
                    del content['image_url']
                    if type(data) == dict:
                        data = data['url']
                    content['image'] = data
                    mm_contents.append(data)
        return mm_contents if len(mm_contents) != 0 else None
    
    @torch.inference_mode()
    def mm_prepare_inputs(self, input_data: InputData):
        # Calculate the embedding and positions of pic
        batch_embeddings = []
        batch_positions = []
        for seq in input_data.seqs:
            embedding = None
            position = None
            if seq.computed_prompt:
                embedding_info = self.embedding_cache[seq.seq_id]
                position = MRotaryEmbedding.get_next_input_positions(
                    embedding_info.mrope_position_delta,
                    seq.computed_token_num,
                    seq.seq_len
                )
                position = torch.tensor(position)
                embedding = self.model.get_input_embeddings(
                    torch.tensor(seq.token_ids[seq.computed_token_num:seq.seq_len]))
            else:
                if seq.computed_token_num == 0 or seq.seq_id not in self.embedding_cache:
                    mm_embeddings = None
                    image_grid_thw = None
                    if seq.mm_contents is not None:
                        images = load_images(seq.mm_contents)
                        images_input = self.image_processor(images=images)
                        image_grid_thw = images_input['image_grid_thw']
                        mm_embeddings = self.model.get_multimodal_embeddings(**images_input)
                    prompt_embeddings = self.model.get_input_embeddings(
                        torch.tensor(seq.token_ids), mm_embeddings)
                    prompt_positions, mrope_position_delta = MRotaryEmbedding.get_input_positions(
                        input_tokens=seq.token_ids,
                        hf_config=self.model.config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=None,
                        second_per_grid_ts=None,
                    )
                    self.embedding_cache[seq.seq_id] = EmbeddingInfo(
                        prompt_embeddings, prompt_positions, mrope_position_delta)
                    position = prompt_positions[
                        :, seq.computed_token_num:seq.seq_len]
                    embedding = prompt_embeddings[
                        seq.computed_token_num:seq.seq_len, :]
                else:
                    embedding_info = self.embedding_cache[seq.seq_id]
                    position = embedding_info.positions[
                        :, seq.computed_token_num:seq.seq_len]
                    embedding = embedding_info.embedding[
                        seq.computed_token_num:seq.seq_len, :]
            batch_embeddings.append(embedding)
            batch_positions.append(position)
        input_embeddings = torch.concat(batch_embeddings)
        positions = torch.concat(batch_positions, dim=1).to(torch.long)
        return input_embeddings, positions
        
    @torch.inference_mode()
    def step_once(self, input_data: InputData = None, hidden_states=None, residual=None):
        if is_first_pp_rank() and self.use_mm:
            input_embeddings, input_data.positions = self.mm_prepare_inputs(input_data)
            output = self.model(input_data, None, None, input_embeddings)
        else:
            output = self.model(input_data, hidden_states, residual)
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
