from typing import Dict, List, Union, Optional

import torch
from attr import dataclass
from logger import logger
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.image_utils import load_images
from transformers.video_utils import load_video
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

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
from gllm.utils import unify_decode
from gllm.async_utils import AsyncSchedulerContext, FutureMap


@dataclass
class EmbeddingInfo:
    embedding: torch.Tensor = None
    prompt_positions: torch.Tensor = None
    mrope_position_delta: torch.Tensor = None

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
        disable_cuda_graph: bool,
        max_cuda_graph_bs: int,
        model_max_length: int,
        mm_processor_min_pixels: int = None,
        mm_processor_max_pixels: int = None,
    ):
        
        self.max_num_batched_tokens = (
            maxp
            if schedule_method in ["chunked_prefill", "split_pd"]
            else maxp + maxd
        )
        
        self.max_running_seqs = (
            maxp
            if schedule_method in ["chunked_prefill", "split_pd"]
            else maxd
        )
        
        self.model_path = model_path
        self.model_loader = ModelLoader(load_format, model_path, self.max_num_batched_tokens)
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
            self.image_processor = self.processor.image_processor
            self.video_processor = self.processor.video_processor
            if mm_processor_min_pixels is not None:
                self.image_processor.min_pixels = mm_processor_min_pixels
                self.video_processor.min_pixels = mm_processor_min_pixels
                self.image_processor.size["shortest_edge"] = mm_processor_min_pixels
                self.video_processor.size["shortest_edge"] = mm_processor_min_pixels
                logger.info(f"Min pixels: {mm_processor_min_pixels}")
            if mm_processor_max_pixels is not None:
                self.image_processor.max_pixels = mm_processor_max_pixels
                self.video_processor.max_pixels = mm_processor_max_pixels
                self.image_processor.size["longest_edge"] = mm_processor_max_pixels
                self.video_processor.size["longest_edge"] = mm_processor_max_pixels
                logger.info(f"Max pixels: {mm_processor_max_pixels}")
            

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
        self.disable_cuda_graph = disable_cuda_graph
        self.max_cuda_graph_bs = max_cuda_graph_bs
        self.size_to_graph: Dict[int, torch.cuda.CUDAGraph] = dict()
        # Use power-of-two bucket sizes to reduce the number of captured graphs.
        # At runtime the actual batch is padded up to the nearest bucket.
        self.capture_sizes = self._build_capture_sizes(self.max_cuda_graph_bs)

        # max length
        self.model_max_length = self.resolve_model_max_length(model_max_length)

    def resolve_model_max_length(self, model_max_length):
        if self.tokenizer.model_max_length != VERY_LARGE_INTEGER:
            model_max_length = self.tokenizer.model_max_length
        if self.model_loader.generation_config.max_length != 20:
            model_max_length = self.model_loader.generation_config.max_length
        if model_max_length is None:
            model_max_length = 8192
        logger.info(f"Model max length: {model_max_length}")
        return model_max_length

    @staticmethod
    def _build_capture_sizes(max_bs: int):
        """Return power-of-two bucket sizes up to max_bs, in descending order.

        For example, max_bs=20 → [20, 16, 8, 4, 2, 1].
        We always include 1 as a floor bucket.
        """
        if max_bs <= 0:
            return []
        sizes = []
        s = 1
        while s <= max_bs:
            sizes.append(s)
            s *= 2
        # If max_bs is not itself a power of two, add it as the top bucket so
        # that batches of exactly max_bs can still use CUDA graph.
        if sizes[-1] != max_bs:
            sizes.append(max_bs)
        return list(reversed(sizes))

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
            max_running_seqs=self.max_running_seqs,
            max_seq_length=self.model_max_length,
            memory_manager=self.memory_manager,
            use_buffer=True,
        )
        self.input_hidden_states = torch.zeros((self.max_num_batched_tokens, self.hidden_size))
        self.input_residual = torch.zeros((self.max_num_batched_tokens, self.hidden_size))
        # Output buffer
        self.output_hidden_states = torch.zeros((self.max_num_batched_tokens, self.hidden_size))
        self.output_residual = torch.zeros((self.max_num_batched_tokens, self.hidden_size))
        # Profile run
        self.profile_run()
        # Init KV cache at last; only reserve the dummy page when CUDA graphs
        # are actually enabled so we don't waste memory otherwise.
        self.memory_manager.init(reserve_dummy_page=not self.disable_cuda_graph)

        if not self.disable_cuda_graph:
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

    def decode(self, token_ids):
        return unify_decode(self.tokenizer, token_ids)

    def extract_modify_mm(self, messages: Dict):
        mm_contents = {"image": [], "video": []}
        for message in messages:
            contents = message["content"]
            if type(contents) != list:
                continue
            for content in contents:
                if content["type"] == "image":
                    mm_contents["image"].append(content["image"])
                elif content["type"] == "video":
                    mm_contents["video"].append(content["video"])
                elif content["type"] == "image_url":
                    content["type"] = "image"
                    data = content["image_url"]
                    del content["image_url"]
                    if type(data) == dict:
                        data = data["url"]
                    content["image"] = data
                    mm_contents["image"].append(data)
                elif content["type"] == "video_url":
                    content["type"] = "video"
                    data = content["video_url"]
                    del content["video_url"]
                    if type(data) == dict:
                        data = data["url"]
                    content["video"] = data
                    mm_contents["video"].append(data)
        return mm_contents if len(mm_contents["image"]) + len(mm_contents["video"]) != 0 else None

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
                # In async scheduling, decode tokens may contain negative
                # FutureMap placeholder values (e.g. -42).  These will be
                # resolved on the GPU side via resolve_future(), but here we
                # need a valid token ID for the embedding lookup.  Clamp to 0
                # (the actual token value is irrelevant since the embedding is
                # only used as the model forward's input hidden state, and the
                # real token ID is resolved on GPU before any KV-cache write).
                to_compute = torch.tensor(seq.to_compute_tokens)
                if (to_compute < 0).any():
                    to_compute = to_compute.clamp(min=0)
                embedding = self.model.embed_input_ids(to_compute)
                position = MRotaryEmbedding.get_next_input_positions(
                    embedding_info.mrope_position_delta,
                    seq.computed_token_num,
                    seq.seq_len,
                )
                position = torch.tensor(position, device="cpu")
            else:
                embedding_info = None
                if seq.seq_id not in self.embedding_cache:
                    mm_embeddings = None
                    image_grid_thw: torch.Tensor = None
                    video_grid_thw: torch.Tensor = None
                    if seq.mm_contents is not None:
                        mm_input = {}
                        if len(seq.mm_contents["image"]) != 0:
                            images = load_images(seq.mm_contents["image"])
                            images_input = self.image_processor(images=images)
                            mm_input.update(images_input)
                            image_grid_thw = images_input["image_grid_thw"]
                        if len(seq.mm_contents["video"]) != 0:
                            videos = []
                            video_metadata = []
                            for video_content in seq.mm_contents["video"]:
                                video_data, metadata = load_video(video_content)
                                videos.append(video_data)
                                video_metadata.append(metadata)
                            videos_input = self.video_processor(
                                videos=videos,
                                video_metadata=video_metadata,
                            )
                            mm_input.update(videos_input)
                            video_grid_thw = videos_input["video_grid_thw"]
                        mm_embeddings = self.model.embed_multimodal(
                            **mm_input
                        )
                    
                    input_ids = torch.tensor(seq.token_ids)
                    placeholder_token_id = torch.tensor(self.model.get_mm_placeholder_token_ids())
                    prompt_embeddings = self.model.embed_input_ids(
                        input_ids, 
                        mm_embeddings,
                        torch.isin(input_ids, placeholder_token_id)
                    )
                    if image_grid_thw is not None:
                        image_grid_thw = image_grid_thw.cpu()
                    if video_grid_thw is not None:
                        video_grid_thw = video_grid_thw.cpu()
                    prompt_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=seq.token_ids,
                            hf_config=self.model.config,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
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
            # Check if VL embeddings were pre-computed (async path).
            mm_embeddings = getattr(input_data, '_mm_embeddings', None) if input_data is not None else None
            if mm_embeddings is not None:
                # Use pre-computed results from _build_prefetched_input.
                self.input_data.set_mrope_position(input_data._mm_positions)
                self.prepare_input_embeddings(mm_embeddings)
            else:
                # Sync path or no prebuilt data — compute on the spot.
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
        return seqs

    @torch.inference_mode()
    def profile_run(self):
        seqs = self.create_dummy_seqs(self.max_running_seqs)
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
            logger.info(f"Capturing CUDA graphs for bucket sizes: {list(reversed(self.capture_sizes))}")
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
        # Only pure decode batches use CUDA graph.
        if self.check_decode_batch():
            # Find the smallest captured bucket >= actual batch size.
            padded_size = None
            for bucket in self.capture_sizes:
                if bucket >= num_cal_tokens:
                    padded_size = bucket
            if padded_size is not None and padded_size in self.size_to_graph:
                # Pad input buffers to the bucket size with dummy values, then
                # replay the pre-captured graph.
                num_real_tokens = self.input_data.pad_for_cuda_graph(padded_size)
                self.size_to_graph[padded_size].replay()
                # After replay, use only the real-token slice for logits.
                num_cal_tokens = num_real_tokens
            else:
                self.forward()
        else:
            self.forward()
        if is_last_pp_rank():
            logits = self.model.compute_logits(
                self.input_data, self.output_hidden_states[:num_cal_tokens]
            )
            self.input_data.prepare_sample()
            next_tokens = self.sampler.forward(logits, self.input_data)
            return next_tokens
        return (
            self.output_hidden_states[:num_cal_tokens],
            self.output_residual[:num_cal_tokens],
        )

    def free(self, seq: Sequence):
        self.memory_manager.free(seq)
        if self.use_mm and is_first_pp_rank():
            self.embedding_cache.pop(seq.seq_id)
            
class AsyncModelRunner(ModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def init_async(self, num_prefill_chunks: int = 256):
        """Initialize async scheduling infrastructure.
        
        Sets up CUDA streams and FutureMap circular buffer for non-blocking
        async scheduling.
        
        Args:
            num_prefill_chunks: Size of prefill chunks for buffer calculation
        """
        
        # Async scheduling: double-buffered pinned CPU buffer for D2H token copy.
        # Two buffers alternate so batch N's collect can safely read buffer[i]
        # while batch N+1's D2H writes to buffer[1-i].
        self._next_tokens_bufs = [
            torch.zeros(self.max_running_seqs, dtype=torch.long, pin_memory=True, device='cpu'),
            torch.zeros(self.max_running_seqs, dtype=torch.long, pin_memory=True, device='cpu'),
        ]
        self._next_tokens_buf_idx = 0
        # Dedicated CUDA stream for D2H token copies so they run concurrently
        # with the next batch's GPU forward pass on the default stream.
        self.copy_stream = torch.cuda.Stream()
        # Event recorded on forward_stream after the model forward completes
        # (input buffers are no longer being read). The default stream waits
        # on this before overwriting input buffers with the next H2D copy.
        self._input_consumed_event = torch.cuda.Event()
        
        self.future_map: Optional[FutureMap] = None
        self.async_context: Optional[AsyncSchedulerContext] = None
        self.schedule_stream: Optional[torch.cuda.Stream] = None
        self.forward_stream: Optional[torch.cuda.Stream] = None
        
        if self.future_map is not None:
            logger.warning("Async already initialized, skipping re-initialization")
            return
        
        # Create async scheduler context with high-priority schedule stream
        self.async_context = AsyncSchedulerContext(
            schedule_stream_priority=0,
            device=torch.device(f"cuda:{get_local_rank()}"),
        )
        
        # Store stream references for convenient access
        self.schedule_stream = self.async_context.schedule_stream
        self.forward_stream = self.async_context.forward_stream
        
        # Initialize FutureMap circular buffer for token ID buffering
        self.future_map = FutureMap(
            max_running_requests=self.max_running_seqs,
            context_len=self.model_max_length,
            chunked_prefill_size=num_prefill_chunks,
            device=torch.device(f"cuda:{get_local_rank()}"),
        )
        
        logger.info(
            f"Async scheduling initialized: "
            f"future_limit={self.future_map.future_limit}, "
            f"buffer_size={self.future_map.future_buffer_len * 8 / 1024:.1f} KB"
        )
        
    @torch.inference_mode()
    def run_batch_async(self) -> tuple:
        """Run the full batch pipeline on forward_stream and return immediately.

        All GPU work — resolve_future, forward, logits, sample, store_to_map,
        async D2H — is submitted to forward_stream in order.  The CPU returns
        as soon as the kernel launches are queued (non-blocking).

        Returns:
            (event, batch_size, future_indices, buf_idx) for step_collect_async.
            batch_size is saved here because self.input_data will be
            overwritten by the next batch's prepare_input before collect.
        """
        if self.future_map is None:
            raise RuntimeError("Call init_async() before using async methods")

        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        batch_size = len(self.input_data.seqs)

        # Pick the current pinned buffer and flip for next call
        buf_idx = self._next_tokens_buf_idx
        self._next_tokens_buf_idx = 1 - buf_idx
        next_tokens_cpu = self._next_tokens_bufs[buf_idx]

        # All GPU work on forward_stream
        with torch.cuda.stream(self.forward_stream):
            # Allocate future indices for this batch's output
            future_indices = self.future_map.alloc_future_indices(batch_size)

            # Wait for default stream (H2D copies from prepare_input)
            self.forward_stream.wait_stream(torch.cuda.default_stream())

            # D2D resolve: replace negative placeholders with real tokens
            self.future_map.resolve_future(
                self.input_data.tokens[:num_cal_tokens]
            )

            # VL fixup: after resolve_future, decode tokens in
            # input_data.tokens now hold real token IDs.  But VL's
            # mm_prepare_inputs already wrote embeddings for these
            # positions using placeholder (clamped to 0) values.
            # Re-compute decode-token embeddings from the resolved
            # GPU tokens so the model sees correct hidden states.
            if self.use_mm and is_first_pp_rank() and self.input_data.embedding_size > 0:
                num_decode_tokens = sum(
                    1 for s in self.input_data.seqs if s.computed_prompt
                )
                if num_decode_tokens > 0:
                    decode_embeds = self.model.language_model.model.embed_tokens(
                        self.input_data.tokens[:num_decode_tokens]
                    )
                    self.input_hidden_states[:num_decode_tokens] = decode_embeds

            # Model forward (CUDA graph or eager)
            if self.check_decode_batch():
                padded_size = None
                for bucket in self.capture_sizes:
                    if bucket >= num_cal_tokens:
                        padded_size = bucket
                if padded_size is not None and padded_size in self.size_to_graph:
                    num_real_tokens = self.input_data.pad_for_cuda_graph(padded_size)
                    self.size_to_graph[padded_size].replay()
                    num_cal_tokens = num_real_tokens
                else:
                    self.forward()
            else:
                self.forward()

            # --- logits + sample + store + D2H (all still on forward_stream) ---
            if is_output_rank():
                logits = self.model.compute_logits(
                    self.input_data, self.output_hidden_states[:num_cal_tokens]
                )
                self.input_data.prepare_sample()
                next_tokens_gpu = self.sampler.forward_gpu(logits, self.input_data)
                n = next_tokens_gpu.shape[0]

                # Store tokens in FutureMap so next batch can D2D resolve them
                self.future_map.store_to_map(future_indices, next_tokens_gpu)

                # Record event AFTER all shared buffer reads are done.
                # compute_logits reads query_start_loc (GPU), prepare_sample
                # reads seqs (CPU list, safe), forward reads tokens/positions/
                # seq_lens/block_table.  All of these are now finished.
                # The next prepare_input waits on this event before overwriting
                # shared GPU buffers.  Only D2H copy follows, which uses
                # next_tokens_gpu (private) and pinned buffer (double-buffered).
                self._input_consumed_event.record(self.forward_stream)

                # Async D2H on copy_stream into THIS iteration's pinned buffer
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_stream(self.forward_stream)
                    next_tokens_gpu.record_stream(self.copy_stream)
                    next_tokens_cpu[:n].copy_(next_tokens_gpu, non_blocking=True)
                    copy_done_event = torch.cuda.Event()
                    copy_done_event.record()

                return copy_done_event, batch_size, num_cal_tokens, future_indices, buf_idx

            # Non-output ranks: record after forward finishes reading inputs
            self._input_consumed_event.record(self.forward_stream)
            event = torch.cuda.Event()
            event.record()
            return event, batch_size, num_cal_tokens, future_indices, buf_idx

    @torch.inference_mode()
    def step_collect_async(self, event, batch_size, num_cal_tokens, buf_idx):
        """Collect results from run_batch_async after D2H completes.

        Uses the saved batch_size, num_cal_tokens and buf_idx (not
        self.input_data which may have been overwritten by a later batch's
        prepare_input).
        """
        event.synchronize()
        if is_last_pp_rank() and is_output_rank():
            return self._next_tokens_bufs[buf_idx][:batch_size].numpy().tolist()
        return (
            self.output_hidden_states[:num_cal_tokens],
            self.output_residual[:num_cal_tokens],
        )

    def sync_before_next_prepare(self):
        """Ensure forward_stream has finished reading shared input buffers.

        Waits on _input_consumed_event (recorded after logits/sample/store
        but BEFORE D2H copy). This allows the next batch's H2D copies to
        overlap with the D2H copy of the current batch:

          forward_stream: [resolve(N)] [forward(N)] [logits(N)] [sample(N)] [store(N)] ★event
          copy_stream:                                                                  [D2H(N)]
          default_stream:                                                       ↑wait   [H2D(N+1)]
                                                                                ↑ overlap D2H!
        """
        torch.cuda.current_stream().wait_event(self._input_consumed_event)
