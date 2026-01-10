from typing import Dict, List, Union

import torch
from attr import dataclass
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.image_utils import load_images

from gllm.dist_utils import (
    get_local_rank,
    get_tp_size,
    is_first_pp_rank,
    is_last_pp_rank,
    is_output_rank,
)
from gllm.input_data import InputData
from gllm.layers.rotary_embedding import MRotaryEmbedding
from gllm.layers.sampler import Sampler
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.model_loader import ModelLoader
from gllm.sequence import Sequence


@dataclass
class EmbeddingInfo:
    embedding: torch.Tensor = None
    prompt_positions: torch.Tensor = None
    mrope_position_delta: torch.Tensor = None
    stale: bool = False


class ModelRunner:
    def __init__(
        self,
        load_format: str,
        model_path: str,
        gpu_memory_util: float,
        page_size: int,
        enable_prefix_caching: bool,
        use_thinking: bool,
        maxp,
        maxd,
        kvthresh,
        minp,
        iterp,
        schedule_method: str,
        enable_cuda_graph: bool,
        max_cuda_graph_bs: int,
    ):
        self.model_path = model_path
        self.model_loader = ModelLoader(load_format, model_path)
        self.enable_prefix_caching = enable_prefix_caching
        self.use_thinking = use_thinking
        self.gpu_memory_util = gpu_memory_util
        self.page_size = page_size
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
            AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        )
        self.maxp = maxp
        self.maxd = maxd
        self.kvthresh = kvthresh
        self.minp = minp
        self.iterp = iterp
        self.schedule_method = schedule_method
        self.sampler = Sampler()

        self.use_mm = self.model_loader.use_mm
        self.use_mla = self.model_loader.use_mla
        self.hidden_size = self.model_loader.hidden_size

        if self.use_mm:
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_path, use_fast=True
            )

        # lazy init
        self.model: torch.nn.Module = None
        self.memory_manager: MemoryManager = None
        self.input_data: InputData = None
        self.input_hidden_states: torch.Tensor = None
        self.input_residual: torch.Tensor = None
        self.output_hidden_states: torch.Tensor = None
        self.output_residual: torch.Tensor = None

        # embedding cache: seq_id => embedding
        self.embedding_cache: Dict[int, EmbeddingInfo] = {}

        # cuda graph
        self.enable_cuda_graph = enable_cuda_graph
        self.max_cuda_graph_bs = max_cuda_graph_bs
        self.size_to_graph: Dict[int, torch.cuda.CUDAGraph] = dict()
        self.capture_sizes = list(range(self.max_cuda_graph_bs, 0, -1))

    def init(self, mp_load_progress=None):
        self.model = self.model_loader.load_model(mp_load_progress)
        memory_manager_cls = (
            PrefixMemoryManager if self.enable_prefix_caching else MemoryManager
        )
        self.memory_manager = memory_manager_cls(
            gpu_memory_util=self.gpu_memory_util,
            num_layers=self.model.num_layers,
            dtype=self.model_loader.dtype,
            page_size=self.page_size,
            kv_head_num=self.model.num_kv_heads // get_tp_size(),
            kv_head_dim=self.model.head_dim,
            vocab_size=self.model_loader.vocab_size,
            use_mla=self.model_loader.use_mla,
        )
        # Input buffer
        self.input_data = InputData(
            max_running_seqs=(
                self.maxp
                if self.schedule_method in ["chunked_prefill", "split_pd"]
                else self.maxd
            ),
            max_seq_length=self.tokenizer.model_max_length,
            memory_manager=self.memory_manager,
            use_buffer=True,
        )
        max_tokens_ret = (
            self.maxp
            if self.schedule_method in ["chunked_prefill", "split_pd"]
            else self.maxp + self.maxd
        )
        self.input_hidden_states = torch.zeros((max_tokens_ret, self.hidden_size))
        self.input_residual = torch.zeros((max_tokens_ret, self.hidden_size))
        # Output buffer
        self.output_hidden_states = torch.zeros((max_tokens_ret, self.hidden_size))
        self.output_residual = torch.zeros((max_tokens_ret, self.hidden_size))
        # Profile run
        self.profile_run()
        # Init KV cache at last
        self.memory_manager.init()

        if self.enable_cuda_graph:
            self.capture_graph()

    def encode(self, messages, chat: bool = False, has_mm: bool = False):
        if chat:
            if not self.use_mm or not has_mm:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=self.use_thinking,
                )
            else:
                return self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )[0]
        else:
            return self.tokenizer.encode(messages)

    def decode(self, content):
        return self.tokenizer.decode(content, True, True)

    def extract_modify_mm(self, messages: Dict):
        mm_contents = []
        for message in messages:
            contents = message["content"]
            if type(contents) != list:
                continue
            for content in contents:
                if content["type"] == "image":
                    mm_contents.append(content["image"])
                elif content["type"] == "image_url":
                    content["type"] = "image"
                    data = content["image_url"]
                    del content["image_url"]
                    if type(data) == dict:
                        data = data["url"]
                    content["image"] = data
                    mm_contents.append(data)
        return mm_contents if len(mm_contents) != 0 else None

    @torch.inference_mode()
    def mm_prepare_inputs(self, seqs: List[Sequence]):
        # Calculate the embedding (on cuda) and positions (on cpu) of pic
        batch_embeddings = []
        batch_positions = []
        for seq in seqs:
            embedding = None
            position = None
            if seq.computed_prompt:
                embedding_info = self.embedding_cache[seq.seq_id]
                assert embedding_info.stale
                embedding = self.model.get_input_embeddings(
                    torch.tensor(seq.to_compute_tokens)
                )
                position = MRotaryEmbedding.get_next_input_positions(
                    embedding_info.mrope_position_delta,
                    seq.computed_token_num,
                    seq.seq_len,
                )
                position = torch.tensor(position, device="cpu")
            else:
                embedding_info = None
                if (
                    seq.seq_id not in self.embedding_cache
                    or self.embedding_cache[seq.seq_id].stale
                ):
                    mm_embeddings = None
                    image_grid_thw: torch.Tensor = None
                    if seq.mm_contents is not None:
                        images = load_images(seq.mm_contents)
                        images_input = self.image_processor(images=images)
                        image_grid_thw = images_input["image_grid_thw"]
                        mm_embeddings = self.model.get_multimodal_embeddings(
                            **images_input
                        )
                    prompt_embeddings = self.model.get_input_embeddings(
                        torch.tensor(seq.token_ids), mm_embeddings
                    )
                    if image_grid_thw is not None:
                        image_grid_thw = image_grid_thw.cpu()
                    prompt_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=seq.token_ids,
                            hf_config=self.model.config,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=None,
                            second_per_grid_ts=None,
                        )
                    )
                    embedding_info = EmbeddingInfo(
                        prompt_embeddings, prompt_positions, mrope_position_delta
                    )
                    self.embedding_cache[seq.seq_id] = embedding_info
                    embedding = prompt_embeddings[
                        seq.computed_token_num : seq.seq_len, :
                    ]
                    position = prompt_positions[:, seq.computed_token_num : seq.seq_len]
                else:
                    embedding_info = self.embedding_cache[seq.seq_id]
                    embedding = embedding_info.embedding[
                        seq.computed_token_num : seq.seq_len, :
                    ]
                    position = embedding_info.prompt_positions[
                        :, seq.computed_token_num : seq.seq_len
                    ]
                if seq.seq_len == seq.prompt_len:
                    # invalidate embedding_cache
                    embedding_info.stale = True
                    embedding_info.embedding = None
            batch_embeddings.append(embedding)
            batch_positions.append(position)
        input_embeddings = torch.concat(batch_embeddings)
        positions = torch.concat(batch_positions, dim=1)
        return input_embeddings, positions

    def prepare_input_embeddings(self, hidden_states=None):
        if hidden_states is not None:
            assert is_first_pp_rank()
            self.input_hidden_states[: hidden_states.shape[0]] = hidden_states
            self.input_data.embedding_size = hidden_states.shape[0]

    def prepare_input(self, seqs: List[Sequence] = None, input_data: InputData = None):
        if input_data is not None:
            self.input_data.set_input_from_prebuilt(input_data)
        else:
            assert seqs is not None
            self.input_data.cal_and_set_input(seqs)
        if self.use_mm and is_first_pp_rank():
            input_embeddings, mrope_positions = self.mm_prepare_inputs(
                self.input_data.seqs
            )
            self.input_data.set_mrope_position(mrope_positions)
            self.prepare_input_embeddings(input_embeddings)

    def create_dummy_seqs(self, size):
        seqs = [Sequence(idx, [1, 2], [], output_len=1) for idx in range(size)]
        for seq in seqs:
            seq.page_table.append(seq.seq_id)
            seq.prompt_len = 1
            seq.computed_token_num = 1
            seq.to_compute_token_num = 1
            seq.to_compute_tokens = [2]
        return seqs

    @torch.inference_mode()
    def profile_run(self):
        seqs = self.create_dummy_seqs(
            self.maxp if self.schedule_method == "chunked_prefill" else self.maxd
        )
        self.input_data.cal_and_set_input(seqs)
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        if self.use_mm:
            self.input_data.set_mrope_position(
                torch.zeros((3, num_cal_tokens), device="cpu")
            )
        if is_first_pp_rank():
            self.model(self.input_data)
        else:
            self.model(
                self.input_data,
                self.input_hidden_states[:num_cal_tokens],
                self.input_residual[:num_cal_tokens],
            )

    @torch.inference_mode()
    def capture_graph(self):
        iterator = self.capture_sizes
        if get_local_rank() == 0:
            # logger.info(f"Capturing cuda graph for sizes {self.capture_sizes}")
            iterator = tqdm(self.capture_sizes, desc="Capturing CUDA Graphs", ncols=100)
        memory_pool = torch.cuda.graph_pool_handle()
        for size in iterator:
            seqs = self.create_dummy_seqs(size)
            self.input_data.cal_and_set_input(seqs=seqs)
            if self.use_mm:
                self.input_data.set_mrope_position(torch.zeros((3, size), device="cpu"))
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph=g, pool=memory_pool):
                self.forward()
            self.size_to_graph[size] = g
        if torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    @torch.inference_mode()
    def forward(self):
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        if is_first_pp_rank() and self.use_mm:
            output = self.model(
                self.input_data,
                (
                    self.input_hidden_states[: self.input_data.embedding_size]
                    if self.input_data.embedding_size > 0
                    else None
                ),
            )
        elif is_first_pp_rank():
            output = self.model(self.input_data)
        else:
            output = self.model(
                self.input_data,
                self.input_hidden_states[:num_cal_tokens],
                self.input_residual[:num_cal_tokens],
            )
        if isinstance(output, tuple):
            assert len(output) == 2
            (
                self.output_hidden_states[:num_cal_tokens],
                self.output_residual[:num_cal_tokens],
            ) = output
        else:
            assert isinstance(output, torch.Tensor)
            self.output_hidden_states[:num_cal_tokens] = output

    def check_decode_batch(self):
        # Since the scheduler put prefill seqs at the end
        # we only check the last seq
        return self.input_data.seqs[-1].computed_prompt

    @torch.inference_mode()
    def step_once(self):
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        # Only decode batch use cuda graph
        if self.check_decode_batch() and num_cal_tokens in self.size_to_graph:
            self.size_to_graph[num_cal_tokens].replay()
        else:
            self.forward()
        if is_last_pp_rank():
            logits = self.model.compute_logits(
                self.input_data, self.output_hidden_states[:num_cal_tokens]
            )
            if is_output_rank():
                self.input_data.prepare_sample()
                next_tokens = self.sampler.forward(logits, self.input_data)
                return next_tokens
        return (
            self.output_hidden_states[:num_cal_tokens],
            self.output_residual[:num_cal_tokens],
        )

    def free(self, seq: Sequence):
        self.memory_manager.free(seq)
