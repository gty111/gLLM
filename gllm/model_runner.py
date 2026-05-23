from contextlib import nullcontext as _nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
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

from gllm.async_utils import FutureMap, OverlapRuntime
from gllm.dist_utils import (
    get_local_rank,
    get_output_rank,
    get_tp_group,
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
        if model_max_length is None:
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
    def _mm_prepare_cpu(self, seqs: List[Sequence]) -> Dict:
        """CPU phase of :meth:`mm_prepare_inputs`.

        Computes mrope positions and collects per-seq prefill work to run in
        :meth:`_mm_prepare_gpu`. Decode seqs (``seq.computed_prompt``) only
        contribute positions and a token count: their embedding rows are
        re-written in one fused call by
        :meth:`OverlapModelRunner._fixup_vl_decode_embeddings` on the forward
        stream, so we skip the per-seq ``embed_input_ids`` launch (and the
        attendant ``aten::any`` / ``aten::clamp`` sync points) entirely.

        Returning a context dict (instead of going straight to GPU work) lets
        the overlap scheduler run this phase concurrently with the previous
        batch's GPU forward.
        """
        batch_positions: List[torch.Tensor] = []
        prefill_works: List[Dict] = []
        num_decode_tokens = 0
        in_decode = True

        for seq in seqs:
            if seq.computed_prompt:
                # Decode token: positions only; embed is deferred to fixup.
                # The scheduler places decode seqs before prefill seqs, so
                # the contiguous decode block always sits at the front.
                assert in_decode, (
                    "scheduler invariant violated: decode seqs must precede "
                    "prefill seqs within a batch"
                )
                embedding_info = self.embedding_cache[seq.seq_id]
                position = MRotaryEmbedding.get_next_input_positions(
                    embedding_info.mrope_position_delta,
                    seq.computed_token_num,
                    seq.seq_len,
                )
                batch_positions.append(torch.tensor(position, device="cpu"))
                num_decode_tokens += seq.to_compute_token_num
                continue

            in_decode = False
            if seq.seq_id not in self.embedding_cache:
                mm_input: Dict = {}
                image_grid_thw: Optional[torch.Tensor] = None
                video_grid_thw: Optional[torch.Tensor] = None
                if seq.mm_contents is not None:
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

                # ``get_input_positions`` does per-element Python indexing on
                # the grid tensors; if a fast image processor handed us CUDA
                # tensors here, that loop would issue a D2H sync per element.
                # Force CPU once, cheaply, so the CPU phase stays CPU-only.
                if isinstance(image_grid_thw, torch.Tensor):
                    image_grid_thw = image_grid_thw.cpu()
                if isinstance(video_grid_thw, torch.Tensor):
                    video_grid_thw = video_grid_thw.cpu()

                # Explicitly pin these tensors to CPU. The repo sets the
                # default device to CUDA via ``ModelLoader``, so a bare
                # ``torch.tensor(...)`` would silently allocate on GPU and
                # the ``torch.isin`` below would launch a kernel on the
                # default stream -- defeating overlap with the previous
                # batch's forward. Materialization to CUDA is deferred to
                # ``_mm_prepare_gpu`` right before ``embed_input_ids``.
                input_ids_cpu = torch.tensor(seq.token_ids, device="cpu")
                placeholder_token_id_cpu = torch.tensor(
                    self.model.get_mm_placeholder_token_ids(), device="cpu"
                )
                is_multimodal_cpu = torch.isin(
                    input_ids_cpu, placeholder_token_id_cpu
                )
                prompt_positions, mrope_position_delta = (
                    MRotaryEmbedding.get_input_positions(
                        input_tokens=seq.token_ids,
                        hf_config=self.model.config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=None,
                    )
                )
                batch_positions.append(
                    prompt_positions[:, seq.computed_token_num : seq.seq_len]
                )
                prefill_works.append(
                    {
                        "kind": "uncached",
                        "seq": seq,
                        "input_ids_cpu": input_ids_cpu,
                        "is_multimodal_cpu": is_multimodal_cpu,
                        "mm_input": mm_input,
                        "prompt_positions": prompt_positions,
                        "mrope_position_delta": mrope_position_delta,
                    }
                )
            else:
                embedding_info = self.embedding_cache[seq.seq_id]
                batch_positions.append(
                    embedding_info.prompt_positions[
                        :, seq.computed_token_num : seq.seq_len
                    ]
                )
                prefill_works.append(
                    {
                        "kind": "cached",
                        "seq": seq,
                        "embedding_info": embedding_info,
                    }
                )

        mrope_positions = torch.concat(batch_positions, dim=1)
        return {
            "prefill_works": prefill_works,
            "mrope_positions": mrope_positions,
            "num_decode_tokens": num_decode_tokens,
        }

    @torch.inference_mode()
    def _mm_prepare_gpu(self, ctx: Dict) -> Optional[torch.Tensor]:
        """GPU phase of :meth:`mm_prepare_inputs`.

        Runs each prefill seq's multimodal+text embed and produces a single
        ``input_embeddings`` tensor laid out as ``[decode_rows, prefill_rows]``.
        Decode rows are an uninitialized placeholder; they will be overwritten
        by :meth:`OverlapModelRunner._fixup_vl_decode_embeddings` on the
        forward stream right before the model runs, so the placeholder content
        is irrelevant.
        """
        device = self.input_hidden_states.device
        batch_embeddings: List[torch.Tensor] = []
        for work in ctx["prefill_works"]:
            seq = work["seq"]
            if work["kind"] == "uncached":
                mm_embeddings = None
                mm_input = work["mm_input"]
                if mm_input:
                    mm_embeddings = self.model.embed_multimodal(**mm_input)

                # Materialize CPU tensors built in ``_mm_prepare_cpu`` onto
                # the model device for the embed kernels. Sources are small
                # (one prompt's worth of ids) so a synchronous H2D is cheap
                # and avoids the pageable-memory caveats of non_blocking.
                input_ids = work["input_ids_cpu"].to(device, non_blocking=True)
                is_multimodal = work["is_multimodal_cpu"].to(
                    device, non_blocking=True
                )
                prompt_embeddings = self.model.embed_input_ids(
                    input_ids,
                    mm_embeddings,
                    is_multimodal,
                )

                embedding_info = EmbeddingInfo(
                    prompt_embeddings,
                    work["prompt_positions"],
                    work["mrope_position_delta"],
                )
                self.embedding_cache[seq.seq_id] = embedding_info
                embedding = prompt_embeddings[
                    seq.computed_token_num : seq.seq_len, :
                ]
            else:
                embedding_info = work["embedding_info"]
                embedding = embedding_info.embedding[
                    seq.computed_token_num : seq.seq_len, :
                ]

            if seq.seq_len == seq.prompt_len:
                # Prefill just finished; drop the cached embedding tensor to
                # free memory. We still keep mrope_position_delta around for
                # future decode-position calculations.
                embedding_info.embedding = None

            batch_embeddings.append(embedding)

        num_decode_tokens = ctx["num_decode_tokens"]
        if num_decode_tokens > 0:
            # Placeholder rows; ``_fixup_vl_decode_embeddings`` re-embeds these
            # token positions in a single fused launch on the forward stream
            # after future-token resolution. ``empty`` is fine since the
            # contents are dead-on-arrival.
            placeholder = torch.empty(
                (num_decode_tokens, self.hidden_size),
                device=self.input_hidden_states.device,
                dtype=self.input_hidden_states.dtype,
            )
            batch_embeddings.insert(0, placeholder)

        if not batch_embeddings:
            return None
        return torch.concat(batch_embeddings)

    @torch.inference_mode()
    def mm_prepare_inputs(self, seqs: List[Sequence]):
        """Single-shot wrapper kept for the non-overlap worker path."""
        ctx = self._mm_prepare_cpu(seqs)
        input_embeddings = self._mm_prepare_gpu(ctx)
        return input_embeddings, ctx["mrope_positions"]

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
    def capture_graph(self, stream: Optional[torch.cuda.Stream] = None):
        """Capture per-bucket decode CUDA graphs.

        ``stream`` controls which CUDA stream the graph is captured on.
        ``torch.cuda.graph`` otherwise allocates a brand-new private stream
        each call, which is fine for kernels but interacts poorly with
        captured NCCL ops if replay later happens on a *different* stream
        (the symptom we hit in TP+overlap runs was gradual KV-cache drift
        between TP ranks surfacing as repetition loops). Subclasses that
        replay on a known stream (e.g. ``OverlapModelRunner.forward_stream``)
        should pass that same stream here so capture and replay agree.
        """
        iterator = self.capture_sizes
        if get_local_rank() == 0:
            logger.info(f"Capturing CUDA graphs for bucket sizes: {list(reversed(self.capture_sizes))}")
            iterator = tqdm(self.capture_sizes, desc="Capturing CUDA Graphs", ncols=100)
        memory_pool = torch.cuda.graph_pool_handle()

        # If the custom NVLink-P2P all-reduce is active, wrap the whole
        # capture in its ``capture()`` context so that, after all buckets
        # are captured, it broadcasts the per-rank IPC handles for the
        # buffers that ended up baked into the graphs. Without this,
        # graph replay on any rank-N>0 would try to dereference a local
        # pointer baked at capture time on rank 0 and crash. With NCCL
        # AR there's nothing to do (NCCL kernels handle their own IPC
        # internally), so a missing/disabled custom AR is a no-op.
        from gllm.distributed import get_custom_allreduce

        car = get_custom_allreduce()
        capture_ctx = car.capture() if car is not None else _nullcontext()
        with capture_ctx:
            for size in iterator:
                seqs = self.create_dummy_seqs(size)
                self.input_data.cal_and_set_input(seqs=seqs)
                if self.use_mm:
                    self.input_data.set_mrope_position(torch.zeros((3, size), device="cpu"))
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
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
            self.embedding_cache.pop(seq.seq_id, None)

    def free_follower_state(self, seq_id: int) -> None:
        """Drop per-seq cache on a TP/PP follower; does **not** touch pages.

        KV-page allocation is centralized on rank-0 (the only place that
        runs the scheduler / memory manager), so followers must not
        re-free pages -- doing so would push the page back into the
        ID allocator while rank-0 still considers it allocated, and
        the next ``pre_allocate_page`` would happily re-hand it to a
        different seq mid-flight.

        What followers *do* need to release on free is the
        ``embedding_cache`` row (VL only, first PP rank only) -- the
        existing code path never reached this because the follower
        was stateless about seq lifetimes pre-refactor, which leaked
        a multimodal-embedding tensor per finished VL request.
        """
        if self.use_mm and is_first_pp_rank():
            self.embedding_cache.pop(seq_id, None)


class OverlapModelRunner(ModelRunner):
    """ModelRunner with FutureMap-based overlap scheduling (TP, pp_size=1 only)."""

    def init(self, mp_load_progress=None):
        # Create the overlap CUDA streams BEFORE ``super().init()`` so that
        # ``capture_graph`` (invoked from inside ``super().init()``) can use
        # ``forward_stream`` as the capture stream. Capturing on the same
        # stream that ``run_batch_async`` replays on keeps the NCCL kernels
        # baked into the graph tied to a single CUDA stream across capture
        # and replay -- mismatch had caused TP ranks to subtly disagree
        # after many decode steps and surface as repetition loops in long
        # generations.
        device = torch.device(f"cuda:{get_local_rank()}")
        self.overlap_runtime = OverlapRuntime(device)
        self.forward_stream = self.overlap_runtime.forward_stream
        self.copy_stream = self.overlap_runtime.copy_stream
        super().init(mp_load_progress)
        self._init_overlap_buffers()

    def capture_graph(self, stream: Optional[torch.cuda.Stream] = None):
        # Capture on ``forward_stream`` so capture stream == replay stream.
        # NCCL kernels (e.g. ``embed_tokens`` all_reduce, layer all_reduces)
        # baked into the graph stay tied to the same CUDA stream across
        # capture and replay. Without this they were captured on a fresh
        # private stream that ``torch.cuda.graph`` allocates by default,
        # then replayed on ``forward_stream`` -- the resulting NCCL/stream
        # mismatch was letting TP ranks subtly drift over many decode
        # iterations and produce the long-generation repetition loops.
        super().capture_graph(stream=self.forward_stream)

    def _init_overlap_buffers(self, num_prefill_chunks: int = 256) -> None:
        device = self.forward_stream.device
        self.future_map = FutureMap(
            max_running_requests=self.max_running_seqs,
            context_len=self.model_max_length,
            chunked_prefill_size=num_prefill_chunks,
            device=device,
        )
        self._next_tokens_bufs = [
            torch.zeros(
                self.max_running_seqs,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            ),
            torch.zeros(
                self.max_running_seqs,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            ),
        ]
        self._next_tokens_buf_idx = 0
        # Holds the context produced by ``_mm_prepare_cpu`` between the CPU
        # and GPU phases of input prep when the overlap worker drives us.
        self._pending_mm_ctx: Optional[Dict] = None
        logger.info(
            "Overlap scheduling enabled: future_limit=%s tp_size=%s",
            self.future_map.future_limit,
            get_tp_size(),
        )

    def sync_before_next_prepare(self) -> None:
        """Deprecated host-side barrier; kept for backward compatibility.

        The fully-async overlap pipeline expresses the
        "previous forward has released the input buffers" dependency on the
        GPU side via ``input_consumed_event`` and ``prep_stream``, so callers
        should no longer need this. The method is intentionally a no-op now;
        we keep the symbol so that any external user gets a smooth migration.
        """
        return

    def prepare_input_cpu(self, input_data: InputData) -> None:
        """CPU-only portion of input prep.

        Safe to invoke while the previous batch's forward is still consuming
        the shared GPU input buffers — this only touches Python attributes
        and CPU tensors. The companion :meth:`prepare_input_gpu` issues the
        actual H2D and embed work on ``prep_stream``, which itself
        GPU-waits for the previous forward via ``input_consumed_event``.
        """
        self.input_data.set_input_from_prebuilt_cpu(input_data)
        if self.use_mm and is_first_pp_rank():
            assert self._pending_mm_ctx is None, (
                "prepare_input_cpu called twice without an intervening "
                "prepare_input_gpu"
            )
            self._pending_mm_ctx = self._mm_prepare_cpu(self.input_data.seqs)
        else:
            self._pending_mm_ctx = None

    def prepare_input_gpu(self) -> None:
        """GPU/H2D portion of input prep, fully async.

        All work (H2D copies into the shared input buffers, deferred
        multimodal embed for prefill seqs, scattering decode embeddings) is
        enqueued on ``prep_stream``. ``prep_stream`` first GPU-waits on
        ``input_consumed_event`` so the writes can't clobber input buffers
        that the previous batch's forward is still reading. After the work
        is queued we record ``input_ready_event`` so that
        :meth:`run_batch_async` can have ``forward_stream`` GPU-wait on it.

        The host thread never blocks here: the ``cudaEventSynchronize`` that
        used to serialize batches has been replaced by GPU-side
        ``cudaStreamWaitEvent`` and stream events.
        """
        rt = self.overlap_runtime
        # GPU-side wait: prep_stream blocks until the previous forward has
        # finished reading the input buffers. ``wait_event`` on an unrecorded
        # event is a no-op (CUDA semantics), so this is safe on the very
        # first iteration.
        rt.prep_stream.wait_event(rt.input_consumed_event)
        with torch.cuda.stream(rt.prep_stream):
            self.input_data.copy_to_input_buffer()
            if self._pending_mm_ctx is not None:
                ctx = self._pending_mm_ctx
                self._pending_mm_ctx = None
                input_embeddings = self._mm_prepare_gpu(ctx)
                self.input_data.set_mrope_position(ctx["mrope_positions"])
                self.prepare_input_embeddings(input_embeddings)
            rt.input_ready_event.record(rt.prep_stream)

    def _run_forward_on_stream(self, num_cal_tokens: int) -> int:
        if self.check_decode_batch():
            padded_size = None
            for bucket in self.capture_sizes:
                if bucket >= num_cal_tokens:
                    padded_size = bucket
            if padded_size is not None and padded_size in self.size_to_graph:
                num_cal_tokens = self.input_data.pad_for_cuda_graph(padded_size)
                self.size_to_graph[padded_size].replay()
            else:
                self.forward()
        else:
            self.forward()
        return num_cal_tokens

    def _fixup_vl_decode_embeddings(self, num_decode_tokens: int) -> None:
        if (
            self.use_mm
            and is_first_pp_rank()
            and self.input_data.embedding_size > 0
            and num_decode_tokens > 0
        ):
            decode_embeds = self.model.language_model.model.embed_tokens(
                self.input_data.tokens[:num_decode_tokens]
            )
            self.input_hidden_states[:num_decode_tokens] = decode_embeds

    @torch.inference_mode()
    def run_batch_async(self) -> Tuple[torch.cuda.Event, int, List[int], int]:
        """Launch forward + sample on forward_stream (pp_size=1 only)."""
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        batch_size = len(self.input_data.seqs)
        buf_idx = self._next_tokens_buf_idx
        self._next_tokens_buf_idx = 1 - buf_idx
        next_tokens_cpu = self._next_tokens_bufs[buf_idx]

        # ``future_slot_ids`` is purely a CPU concept (used by the scheduler
        # for deferred output finalize). Derive it from the allocator's CPU
        # state instead of materializing a GPU tensor and yanking it back
        # via ``.cpu().tolist()`` -- that round-trip used to insert a hidden
        # ``cudaStreamSynchronize`` on every batch.
        future_indices = self.future_map.alloc_future_indices(batch_size)
        future_slot_ids = list(
            range(future_indices.interval.start, future_indices.interval.stop)
        )

        # ``prepare_input_gpu`` enqueued all H2D + (VL) embed work on
        # ``prep_stream`` and recorded ``input_ready_event``. ``forward_stream``
        # GPU-waits on that event before reading the shared input buffers, so
        # the pipeline is fully async (no host-side ``cudaEventSynchronize``).
        # We additionally wait_stream(default_stream) defensively in case any
        # incidental work landed on the worker thread's default stream
        # (e.g. user-issued ops outside the overlap path); that's a no-op in
        # the steady state.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.forward_stream):
            self.forward_stream.wait_event(self.overlap_runtime.input_ready_event)
            self.forward_stream.wait_stream(default_stream)
            self.future_map.resolve_future(
                self.input_data.tokens[:num_cal_tokens]
            )

            num_decode_tokens = sum(
                1 for s in self.input_data.seqs if s.computed_prompt
            )
            self._fixup_vl_decode_embeddings(num_decode_tokens)

            num_cal_tokens = self._run_forward_on_stream(num_cal_tokens)

            logits = self.model.compute_logits(
                self.input_data, self.output_hidden_states[:num_cal_tokens]
            )
            self.input_data.prepare_sample()
            next_tokens_gpu = None
            if is_output_rank():
                next_tokens_gpu = self.sampler.forward_gpu(logits, self.input_data)
            if get_tp_size() > 1:
                if next_tokens_gpu is None:
                    next_tokens_gpu = torch.empty(
                        batch_size,
                        dtype=torch.long,
                        device=self.input_data.tokens.device,
                    )
                # Use the TP group so that this broadcast goes through the
                # same NCCL communicator as the model's all_reduces. Sharing
                # one communicator means NCCL's per-communicator FIFO
                # ordering implicitly serializes broadcast vs all_reduce
                # within a rank, removing a class of cross-communicator
                # ordering hazards that were occasionally letting TP ranks
                # store stale tokens into ``token_ids_buf`` and surface as
                # repetition loops in long generations.
                dist.broadcast(
                    next_tokens_gpu,
                    src=get_output_rank(),
                    group=get_tp_group(),
                )
            self.future_map.store_to_map(future_indices, next_tokens_gpu)
            # Every PP-0 TP rank D2H-copies the broadcast tokens into
            # its own pinned ``_next_tokens_bufs`` slot. Pre-refactor
            # only ``output_rank`` did this because rank-0 was the
            # sole consumer of the integer token list; with the
            # column-driver design every TP rank's local scheduler
            # needs to ``process_output_finalize`` against the same
            # tokens, so we issue ``tp_size`` independent D2H copies
            # off the same already-broadcast GPU tensor. The copies
            # all run on the per-rank ``copy_stream`` (one per worker
            # process), so there's no inter-rank serialization.
            if get_tp_size() > 1 or is_output_rank():
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_stream(self.forward_stream)
                    next_tokens_cpu[:batch_size].copy_(
                        next_tokens_gpu, non_blocking=True
                    )

            self.overlap_runtime.input_consumed_event.record(self.forward_stream)

        copy_done = torch.cuda.Event()
        if get_tp_size() > 1 or is_output_rank():
            copy_done.record(self.copy_stream)
        else:
            copy_done.record(self.forward_stream)
        return copy_done, batch_size, future_slot_ids, buf_idx

    @torch.inference_mode()
    def step_collect_async(
        self,
        copy_done: torch.cuda.Event,
        batch_size: int,
        buf_idx: int,
    ) -> Union[list[int], Tuple[torch.Tensor, torch.Tensor]]:
        copy_done.synchronize()
        if is_output_rank():
            return self._next_tokens_bufs[buf_idx][:batch_size].tolist()
        return None
