import torch

from attr import dataclass
from typing import Union, Dict, List
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
    prompt_positions: torch.Tensor = None
    mrope_position_delta: torch.Tensor = None
    stale: bool = False

class ModelRunner():
    def __init__(self, load_format: str, model_path: str, gpu_memory_util: float, page_size: int,
                 enable_prefix_caching: bool, use_thinking: bool, maxp, maxd, kvthresh, minp, iterp,
                 use_cp_schedule: bool):
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
        self.use_cp_schedule = use_cp_schedule
        self.sampler = Sampler()
        
        self.use_mm = self.model_loader.use_mm
        self.hidden_size = self.model_loader.hidden_size
        
        if self.use_mm:
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            self.image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)

        # lazy init
        self.model = None
        self.memory_manager = None
        self.input_data = None
        self.input_hidden_states = None
        self.input_residual = None
        self.output_hidden_states = None
        self.output_residual = None
        
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
        # Input buffer
        self.input_data = InputData(
            max_running_seqs=self.maxp if self.use_cp_schedule else self.maxd, 
            max_seq_length=self.tokenizer.model_max_length,
            memory_manager=self.memory_manager,
            use_buffer=True,
        )
        max_tokens_ret = self.maxp if self.use_cp_schedule else self.maxp + self.maxd
        self.input_hidden_states = torch.zeros((max_tokens_ret, self.hidden_size))
        self.input_residual = torch.zeros((max_tokens_ret, self.hidden_size))
        # Output buffer
        self.output_hidden_states = torch.zeros((max_tokens_ret, self.hidden_size))
        self.output_residual = torch.zeros((max_tokens_ret, self.hidden_size))
        # Profile run
        self.profile_run()
        # Init KV cache at last
        self.memory_manager.init()

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
    
    def extract_modify_mm(self, messages:Dict):
        mm_contents = []
        for message in messages:
            contents = message['content']
            if type(contents) != list:
                continue
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
    def mm_prepare_inputs(self, seqs: List[Sequence]):
        # Calculate the embedding and positions of pic
        batch_embeddings = []
        batch_positions = []
        for seq in seqs:
            embedding = None
            position = None
            if seq.computed_prompt:
                embedding_info = self.embedding_cache[seq.seq_id]
                assert embedding_info.stale
                embedding = self.model.get_input_embeddings(
                    torch.tensor(seq.token_ids[seq.computed_token_num:seq.seq_len]))
                position = MRotaryEmbedding.get_next_input_positions(
                    embedding_info.mrope_position_delta,
                    seq.computed_token_num,
                    seq.seq_len,
                )
                position = torch.tensor(position)
            else:
                embedding_info = None
                if seq.seq_id not in self.embedding_cache or self.embedding_cache[seq.seq_id].stale:
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
                    embedding_info = EmbeddingInfo(prompt_embeddings, prompt_positions, mrope_position_delta)
                    self.embedding_cache[seq.seq_id] = embedding_info
                    embedding = prompt_embeddings[
                        seq.computed_token_num:seq.seq_len, :]
                    position = prompt_positions[:, seq.computed_token_num:seq.seq_len]
                else:
                    embedding_info = self.embedding_cache[seq.seq_id]
                    embedding = embedding_info.embedding[
                        seq.computed_token_num:seq.seq_len, :]
                    position = embedding_info.prompt_positions[:, seq.computed_token_num:seq.seq_len]
                if seq.seq_len == seq.prompt_len:
                    # invalidate embedding_cache
                    embedding_info.stale = True
                    embedding_info.embedding = None
            batch_embeddings.append(embedding)
            batch_positions.append(position)
        input_embeddings = torch.concat(batch_embeddings)
        positions = torch.concat(batch_positions,dim=1)
        return input_embeddings, positions
    
    def prepare_hidden_states(self, hidden_states=None):
        if hidden_states is not None:
            assert is_first_pp_rank()
            self.input_hidden_states[:hidden_states.shape[0]] = hidden_states
            self.input_data.embedding_size = hidden_states.shape[0]
    
    def cal_input(self, seqs:List[Sequence], hidden_states=None):
        self.input_data.cal_and_set_input(seqs)
        self.prepare_hidden_states(hidden_states)
    
    def set_input(self, input_data:InputData, hidden_states=None):
        self.input_data.set_input_from_prebuilt(input_data)
        self.prepare_hidden_states(hidden_states)
    
    def set_mrope_positions(self, mrope_postions):
        self.input_data.set_mrope_position(mrope_postions)
    
    @torch.inference_mode()
    def profile_run(self):
        seqs = [Sequence(idx, [1], [0], output_len=1) for idx in 
                range(self.maxp if self.use_cp_schedule else self.maxd)]
        for seq in seqs:
            seq.page_table.append(seq.seq_id)
            seq.computed_token_num = 0
            seq.to_compute_token_num = 1
        self.cal_input(seqs)
        num_cal_toknes = self.input_data.tokens_cpu.shape[0]
        if is_first_pp_rank():
            self.model(self.input_data)
        else:
            self.model(
                self.input_data, 
                self.input_hidden_states[:num_cal_toknes], 
                self.input_residual[:num_cal_toknes]
            )
    
    @torch.inference_mode()
    def step_once(self):
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        if is_first_pp_rank() and self.use_mm:
            output = self.model(
                self.input_data, 
                self.input_hidden_states[:self.input_data.embedding_size] 
                if self.input_data.embedding_size > 0 else None
            )
        elif is_first_pp_rank():
            output = self.model(self.input_data)
        else:
            output = self.model(
                self.input_data, 
                self.input_hidden_states[:num_cal_tokens], 
                self.input_residual[:num_cal_tokens]
            )
        if isinstance(output, tuple):
            assert len(output) == 2
            self.output_hidden_states[:num_cal_tokens], self.output_residual[:num_cal_tokens] = output
        else:
            assert isinstance(output, torch.Tensor)
            self.output_hidden_states[:num_cal_tokens] = output
        if is_last_pp_rank():
            logits = self.model.compute_logits(self.input_data, self.output_hidden_states[:num_cal_tokens])
            if is_output_rank():
                self.input_data.prepare_sample()
                next_tokens = self.sampler.forward(logits, self.input_data)
                return next_tokens
        return self.output_hidden_states[:num_cal_tokens], self.output_residual[:num_cal_tokens]

    def free(self, seq: Sequence):
        self.memory_manager.free(seq)
