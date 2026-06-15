import hashlib
from collections import OrderedDict
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
    # Per-prompt deepstack residual (shape ``[L, N, hidden]``). Cached
    # alongside ``embedding`` so chunked prefill / prefix-cache re-runs can
    # re-slice it the same way ``embedding`` is sliced and feed the model
    # buffer the chunk that matches ``hidden_states``. ``None`` for
    # text-only prompts and for non-deepstack VL models.
    deepstack_embedding: torch.Tensor = None
    # Encoder-disaggregation overlap (design §6.2): for a seq whose visual
    # embeddings are still arriving, ``embedding`` only covers the span-aligned
    # *ready prefix* ``[0, coverage_len)``. When the scheduler later advances
    # past ``coverage_len`` (more items became ready), the embed is rebuilt
    # over the larger prefix. ``None`` => full-prompt embedding (monolith and
    # fully-ready disagg seqs), i.e. no coverage limit.
    coverage_len: Optional[int] = None


@dataclass
class DisaggSeqState:
    """Per-seq encoder-disaggregation overlap state (design §6.2).

    Owned by the :class:`ModelRunner` (keyed by ``seq_id``) so it is immune to
    the scheduler's chunked-prefill ``deepcopy`` of the :class:`Sequence`. The
    LM disagg manager fills ``item_embed[i]`` (and flips ``item_ready[i]``) as
    each item's visual embedding lands over NIXL; the scheduler reads
    ``item_ready`` for the two-layer prefill gate and the model runner reads
    ``item_embed`` to embed the ready prefix.

    Items are stored in **image-then-video order** (the order
    ``model.embed_multimodal`` returns its tuple in, which is what the merge
    expects). Each carries its ``[span_start, span_end)`` in the *expanded*
    token sequence so gate B and the ready-prefix embed can be computed.
    """

    num_items: int
    item_span: List[Tuple[int, int]]          # ordered: (start, end) in tokens
    item_modality: List[str]
    item_ready: List[bool]
    item_embed: List[Optional[torch.Tensor]]  # ordered, filled on NIXL notif
    image_grid_thw: Optional[torch.Tensor]
    video_grid_thw: Optional[torch.Tensor]
    input_ids_cpu: torch.Tensor               # full expanded prompt ids (cpu)
    is_multimodal_cpu: torch.Tensor           # full mask (cpu)
    prompt_positions: torch.Tensor            # full-prompt mrope positions
    mrope_position_delta: torch.Tensor
    prompt_len: int

    @property
    def all_ready(self) -> bool:
        return all(self.item_ready)


# High-id offset for the synthetic ``pad_id``s spliced into the prefix-cache
# key. The flag bit ``1 << 30`` keeps these well above any real vocab id (the
# largest model in this repo, Qwen3.5, tops out around 250k) and below the
# default ``int64`` tokenizer ceiling. The low 30 bits carry 30 bits of the
# multimodal content hash so two distinct images produce different pad ids
# with overwhelming probability.
_MM_PAD_ID_BASE = 1 << 30
_MM_PAD_ID_MASK = _MM_PAD_ID_BASE - 1


def _mm_pad_id_from_hash(mm_hash: bytes) -> int:
    return _MM_PAD_ID_BASE | (int.from_bytes(mm_hash[:4], "big") & _MM_PAD_ID_MASK)


def _hash_tensor_bytes(*tensors: torch.Tensor) -> bytes:
    """Stable digest over the concatenated raw bytes of one or more tensors.

    Vision-tower inputs (pixel_values, grid_thw, timestamps, ...) are CPU-
    side when this runs (forced by ``_mm_prepare_cpu``), so we can lift the
    underlying storage directly without an extra D2H copy.
    """
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"\x00")
            continue
        if t.device.type != "cpu":
            t = t.detach().cpu()
        t = t.contiguous()
        # Mix dtype + shape so two tensors with identical bytes but
        # different reinterpretations can't collide.
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(memoryview(t.numpy().tobytes()))
    return h.digest()


def _build_item_content_hash(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> bytes:
    """Per-item content hash, byte-identical to the monolith's i-th item hash.

    The monolith (:meth:`ModelRunner._build_mm_content_hashes`) computes each
    item's digest as ``_hash_tensor_bytes(pixel_chunk_i, image_grid_thw[i])``
    where ``pixel_chunk_i`` is this item's slice of the concatenated
    ``pixel_values`` and ``image_grid_thw[i]`` is the 1-D ``[3]`` grid row.

    The encoder runs the processor on a *single* image, so its ``pixel_values``
    already equals ``pixel_chunk_i`` and its ``grid_thw`` is ``[1, 3]``; we take
    row 0 to match the monolith's 1-D grid tensor exactly. This determinism is
    what lets the LM's prefix-cache pad ids agree across the two paths
    (design §5.4.4).
    """
    if isinstance(grid_thw, torch.Tensor) and grid_thw.ndim == 2:
        thw = grid_thw[0]
    else:
        thw = grid_thw
    return _hash_tensor_bytes(pixel_values, thw)


class MultiModalEmbeddingCache:
    """LRU cache over ``model.embed_multimodal(**mm_input)`` outputs.

    Key is the prompt-level digest of all of a sequence's multimodal items
    (concatenation of per-item sha256s, computed once in
    :meth:`_mm_prepare_cpu`). Value is the per-item embedding tuple that
    ``embed_multimodal`` returns — i.e. the same shape the model expects to
    splice back into the input embeddings.

    Eviction is byte-aware so a single huge ViT output can't squat on the
    pool indefinitely; once the running total exceeds ``max_bytes`` we evict
    LRU until back under the cap.
    """

    def __init__(self, max_entries: int = 64, max_mb: float = 256.0):
        self._cache: "OrderedDict[bytes, tuple]" = OrderedDict()
        self.max_entries = max_entries
        self.max_bytes = int(max_mb * 1024 * 1024)
        self._cur_bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _size_of(value) -> int:
        if value is None:
            return 0
        total = 0
        for t in value:
            if isinstance(t, torch.Tensor):
                total += t.element_size() * t.numel()
        return total

    def get(self, key: Optional[bytes]):
        if key is None:
            return None
        v = self._cache.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        self._cache.move_to_end(key)
        return v

    def put(self, key: Optional[bytes], value) -> None:
        if key is None or value is None:
            return
        sz = self._size_of(value)
        if sz > self.max_bytes:
            # Don't even try to cache something that wouldn't fit; the
            # eviction loop would just thrash.
            return
        if key in self._cache:
            self._cur_bytes -= self._size_of(self._cache[key])
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._cur_bytes += sz
        # Evict by entry count first, then by byte budget.
        while len(self._cache) > self.max_entries or self._cur_bytes > self.max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cur_bytes -= self._size_of(evicted)


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
        skip_visual: bool = False,
        skip_language: bool = False,
        mla_decode_backend: str = "flashmla",
    ):
        
        self.max_num_batched_tokens = (
            maxp
            if schedule_method in ["chunked_prefill", "split_pd"]
            else maxp + maxd
        )
        
        # Concurrent decode slots (SSM working pool, input buffers, CUDA
        # graph capture). Bounded by ``maxd`` for all schedule methods.
        self.max_running_seqs = maxd
        
        self.model_path = model_path
        self.model_loader = ModelLoader(
            load_format,
            model_path,
            self.max_num_batched_tokens,
            skip_visual=skip_visual,
            skip_language=skip_language,
        )
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
        # Kimi-K2.5 is multimodal but its DeepSeek-V3 language backbone uses
        # ordinary 1-D RoPE, NOT the 3-D mrope that the Qwen-VL family uses.
        # ``uses_mrope`` gates the 3-row position machinery so Kimi flows
        # through the multimodal *embedding-merge* path while keeping plain
        # 1-D positions.
        self.is_kimi_mm = (
            self.model_loader.architecture == "KimiK25ForConditionalGeneration"
        )
        self.uses_mrope = self.use_mm and not self.is_kimi_mm

        # Resolve the MLA decode backend at this (upper) layer and thread the
        # decision down to the attention layers through the model config. The
        # default preference is FlashMLA, which requires a KV page size of 64;
        # bump page_size automatically so the requirement is satisfied without
        # the user having to pass --page-size 64. The attention layer performs
        # the final availability check and falls back to Triton if the
        # FlashMLA kernel cannot actually run on this build/hardware.
        # 64 == required FlashMLA block size (kept as a literal to avoid
        # importing the CUDA-heavy attention module in the parent process).
        _FLASHMLA_PAGE_SIZE = 64
        self.mla_decode_backend = (mla_decode_backend or "flashmla").lower()
        if self.mla_decode_backend not in ("triton", "flashmla"):
            raise ValueError(
                "mla_decode_backend must be 'triton' or 'flashmla', "
                f"got {self.mla_decode_backend!r}."
            )
        if self.use_mla and self.mla_decode_backend == "flashmla":
            if self.page_size != _FLASHMLA_PAGE_SIZE:
                logger.info(
                    f"MLA FlashMLA decode backend requires page_size="
                    f"{_FLASHMLA_PAGE_SIZE}; overriding page_size "
                    f"{self.page_size} -> {_FLASHMLA_PAGE_SIZE}."
                )
                self.page_size = _FLASHMLA_PAGE_SIZE
        # Stamp the resolved preference + final page size onto the model config
        # so ``MLAAttention`` can pick them up at construction time.
        self.model_loader.config.mla_decode_backend = (
            self.mla_decode_backend if self.use_mla else None
        )
        self.model_loader.config.page_size = self.page_size

        # Kimi-K2.5 ships a bespoke processor (``KimiK25Processor``) whose API
        # and outputs diverge from the Qwen-VL ``AutoProcessor`` contract:
        # output keys are ``pixel_values``/``grid_thws`` (not
        # ``image_grid_thw``), no separate ``image_processor``/
        # ``video_processor`` split, and the chat template emits a single
        # ``<|media_pad|>`` per image that must be expanded downstream.
        if self.use_mm and self.is_kimi_mm:
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True
            )
            self.image_processor = None
            self.video_processor = None
        elif self.use_mm:
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

        # Encoder-disaggregation overlap (design §6.2): seq_id => per-item
        # readiness + embeddings for seqs admitted before all their visual
        # embeddings arrived. Populated by the LM disagg manager; consumed by
        # the scheduler (gate B) and the embed path. Empty for the monolith.
        self.disagg_embeds: Dict[int, DisaggSeqState] = {}

        # Multimodal vision-tower output cache, keyed by the content hash of
        # the prompt's MM items. Hits skip ``model.embed_multimodal``
        # entirely. Independent of ``self.embedding_cache`` (which is per
        # seq_id) so it survives across requests. Disabled cheaply when the
        # model isn't multimodal: the put/get paths are guarded by
        # ``self.use_mm`` callers.
        self.mm_embed_cache = MultiModalEmbeddingCache(
            max_entries=64, max_mb=256.0
        )

        # cuda graph
        self.disable_cuda_graph = disable_cuda_graph
        # ``max_cuda_graph_bs`` cannot exceed ``maxd`` — the runtime decode
        # batch is hard-capped at ``maxd`` (scheduler) and several device
        # buffers (e.g. ``InputData.block_table``, ``slot_mapping``) are
        # sized at ``maxd`` rows. Capturing a larger graph would either
        # overflow those buffers during capture or produce graphs that are
        # never replayed at runtime. Clamp here so the user can keep the
        # ``--max-cuda-graph-bs`` default without manually matching ``maxd``.
        if max_cuda_graph_bs > maxd:
            logger.warning(
                f"max_cuda_graph_bs={max_cuda_graph_bs} exceeds maxd={maxd}; "
                f"clamping to {maxd} (decode batch is bounded by maxd)."
            )
            max_cuda_graph_bs = maxd
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
        # Hybrid models (Qwen3.5 GDN) advertise a ready-to-use SSM cache
        # config via ``model.ssm_cache_config``. ``num_layers`` for the KV
        # path must then be the count of *full-attention* layers only.
        ssm_cache_config = getattr(self.model, "ssm_cache_config", None)
        kv_num_layers = getattr(self.model, "num_kv_layers", self.model.num_layers)
        self.memory_manager = memory_manager_cls(
            gpu_memory_util=self.gpu_memory_util,
            num_layers=kv_num_layers,
            dtype=self.model_loader.dtype,
            page_size=self.page_size,
            # ``num_kv_heads / tp_size`` rounded *up* to 1: when the model
            # has fewer kv heads than TP ranks (Qwen3.5-MoE has 2 kv heads
            # with TP=4) each kv head is broadcast across multiple ranks,
            # and every rank still owns one effective slot of KV cache per
            # token. Integer division would zero out the page size and the
            # KV budget computation downstream.
            kv_head_num=max(1, self.model.num_kv_heads // get_tp_size()),
            kv_head_dim=self.model.head_dim,
            vocab_size=self.model_loader.vocab_size,
            use_mla=self.model_loader.use_mla,
            ssm_cache_config=ssm_cache_config,
            max_working_ssm_slots=self.max_running_seqs if ssm_cache_config else 0,
            max_snapshot_ssm_slots=(
                4 * self.max_running_seqs
                if ssm_cache_config and self.enable_prefix_caching
                else 0
            ),
            max_running_seqs=self.max_running_seqs,
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
                # Different chat templates gate "thinking" mode via different
                # variable names: Qwen3/3.5 read ``enable_thinking`` while
                # Kimi-K2.5 reads ``thinking`` (rendering ``<think></think>``
                # when false vs an open ``<think>`` when true). Jinja silently
                # ignores undefined template variables, so passing both keeps
                # every model honoring ``use_thinking`` without per-model
                # branching.
                out = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=self.use_thinking,
                    thinking=self.use_thinking,
                )
            elif self.is_kimi_mm:
                # Kimi's chat template renders one ``<|media_pad|>`` per image
                # and a ``<|kimi_k25_video_placeholder|>`` per video, neither of
                # which its processor expands (unlike Qwen-VL). Render the text,
                # then ``build_kimi_input_ids`` splices video placeholders into
                # per-chunk prompts and expands every ``<|media_pad|>`` to the
                # exact per-item embedding count, so the downstream
                # ``is_multimodal`` mask has one True per produced vision
                # embedding. Counts come from the processor's own calculator,
                # guaranteeing they match the grids the vision tower will emit.
                out = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=self.use_thinking,
                    thinking=self.use_thinking,
                    tokenize=False,
                )
                if isinstance(out, (list, tuple)):
                    out = out[0]
                from gllm.models.kimi_k25 import build_kimi_input_ids

                return build_kimi_input_ids(
                    out,
                    messages,
                    self.processor,
                    self.tokenizer,
                    self.model_loader.config.media_placeholder_token_id,
                )
            else:
                out = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )[0]
        else:
            out = self.tokenizer.encode(messages)
        # transformers >= 5.x ``apply_chat_template`` returns a
        # ``BatchEncoding`` (dict-like) when ``return_dict`` defaults to
        # True; older versions returned a flat ``List[int]``. Normalize
        # here so downstream code can always treat the result as a token
        # id list.
        if hasattr(out, "input_ids"):
            out = out.input_ids
        elif isinstance(out, dict) and "input_ids" in out:
            out = out["input_ids"]
        elif isinstance(out, str):
            # Some custom tokenizers (e.g. Kimi-K2.5's tiktoken-based
            # ``TikTokenTokenizer``) default ``apply_chat_template`` to
            # ``tokenize=False`` and hand back the *rendered prompt string*
            # rather than token ids. Encode it here so the chat path yields
            # ids like every other model. The rendered string already
            # carries the chat special tokens, so a plain ``encode`` round
            # -trips to the same ids as ``tokenize=True``.
            out = self.tokenizer.encode(out)
        return out

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

    def extract_mm_items_ordered(self, messages: List[Dict]):
        """Return the mm items as an ordered ``[(modality, content), ...]`` list.

        Encoder disaggregation needs the items in *prompt order* (matching the
        skeleton's sentinel order) so the LM can pair the i-th sentinel with the
        i-th encoder job. Call *after* :meth:`extract_modify_mm`, which has
        already normalized ``image_url``/``video_url`` -> ``image``/``video``.
        """
        items = []
        for message in messages:
            contents = message["content"]
            if type(contents) != list:
                continue
            for content in contents:
                if content["type"] == "image":
                    items.append(("image", content["image"]))
                elif content["type"] == "video":
                    items.append(("video", content["video"]))
        return items

    def encode_skeleton(self, messages):
        """Text-only tokenization with one sentinel per mm item (design §5.4).

        Used by the disaggregated LM frontend instead of the multimodal
        ``processor.apply_chat_template``: no pixels are opened or processed
        here, and each image/video collapses to a single placeholder id that
        the LM PP0 later expands to ``N_vis_i`` tokens. Returns the skeleton
        token-id list.
        """
        from gllm.mm_common import tokenize_text_only

        cfg = self.model_loader.config
        skel = tokenize_text_only(
            self.tokenizer,
            messages,
            image_token_id=int(cfg.image_token_id),
            video_token_id=int(cfg.video_token_id),
            add_generation_prompt=True,
        )
        return skel.token_ids

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
                if self.uses_mrope:
                    position = MRotaryEmbedding.get_next_input_positions(
                        embedding_info.mrope_position_delta,
                        seq.computed_token_num,
                        seq.seq_len,
                    )
                    batch_positions.append(torch.tensor(position, device="cpu"))
                else:
                    # Kimi: plain 1-D positions. These are discarded by the
                    # caller (``set_mrope_position`` is skipped for Kimi; the
                    # real positions come from ``cal_and_set_input``), but we
                    # still append a correctly-shaped tensor to keep the
                    # downstream ``torch.concat`` happy.
                    batch_positions.append(
                        torch.arange(
                            seq.computed_token_num, seq.seq_len, device="cpu"
                        )
                    )
                num_decode_tokens += seq.to_compute_token_num
                continue

            in_decode = False
            if seq.seq_id in self.disagg_embeds:
                # Encoder-disaggregation overlap (design §6.2): this seq was
                # admitted before all its visual embeddings landed. Embed only
                # the span-aligned *ready prefix*; rebuild when more items land.
                self._mm_disagg_collect(
                    seq,
                    self.disagg_embeds[seq.seq_id],
                    prefill_works,
                    batch_positions,
                )
                continue
            if seq.seq_id not in self.embedding_cache:
                # If the scheduler already ran ``_mm_precompute_hash`` for
                # this seq (required for multimodal prefix-cache correctness
                # -- see that method's docstring), reuse the cached
                # image_processor output and is_multimodal mask. Otherwise
                # build them now (text-only seqs, non-prefix-cache configs,
                # and the never-cached scheduler in tests all land here).
                pre = getattr(seq, "_mm_precomputed", None)
                if pre is not None:
                    mm_input = pre["mm_input"]
                    image_grid_thw = pre["image_grid_thw"]
                    video_grid_thw = pre["video_grid_thw"]
                    input_ids_cpu = pre["input_ids_cpu"]
                    is_multimodal_cpu = pre["is_multimodal_cpu"]
                    mm_bundle_key = pre["mm_bundle_key"]
                    # Encoder-disaggregation (design §5.3): the per-item visual
                    # embeddings were produced on the encoder and NIXL-written
                    # into the LM slot pool, then cloned into this tuple by the
                    # LM disagg manager. When present, ``_mm_prepare_gpu`` uses
                    # them verbatim instead of running the (absent) local ViT.
                    mm_embeddings = pre.get("mm_embeddings")
                    # Single-use: drop the stash so a re-scheduled seq
                    # (preempt + resume) doesn't accidentally read stale
                    # tensors. The work it represents is now folded into
                    # ``embedding_cache[seq.seq_id]`` below.
                    seq._mm_precomputed = None
                else:
                    mm_embeddings = None
                    mm_input, image_grid_thw, video_grid_thw = (
                        self._mm_run_processor(seq)
                    )
                    input_ids_cpu, is_multimodal_cpu = (
                        self._mm_build_is_multimodal_cpu(seq)
                    )
                    mm_bundle_key, item_hashes = (
                        self._build_mm_content_hashes(
                            mm_input, image_grid_thw, video_grid_thw
                        )
                    )
                    if item_hashes:
                        seq.hash_token_ids = self._splice_mm_pad_ids(
                            seq.token_ids,
                            is_multimodal_cpu,
                            item_hashes,
                        )
                    else:
                        seq.hash_token_ids = None

                if self.uses_mrope:
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
                else:
                    # Kimi: plain 1-D positions over the full prompt. Stored in
                    # EmbeddingInfo so decode can extrapolate; ``mrope_position
                    # _delta`` is unused (decode uses ``torch.arange``).
                    prompt_positions = torch.arange(
                        len(seq.token_ids), device="cpu"
                    )
                    mrope_position_delta = None
                    batch_positions.append(
                        prompt_positions[seq.computed_token_num : seq.seq_len]
                    )

                prefill_works.append(
                    {
                        "kind": "uncached",
                        "seq": seq,
                        "input_ids_cpu": input_ids_cpu,
                        "is_multimodal_cpu": is_multimodal_cpu,
                        "mm_input": mm_input,
                        "mm_embeddings": mm_embeddings,
                        "prompt_positions": prompt_positions,
                        "mrope_position_delta": mrope_position_delta,
                        "mm_bundle_key": mm_bundle_key,
                    }
                )
            else:
                embedding_info = self.embedding_cache[seq.seq_id]
                if self.uses_mrope:
                    batch_positions.append(
                        embedding_info.prompt_positions[
                            :, seq.computed_token_num : seq.seq_len
                        ]
                    )
                else:
                    batch_positions.append(
                        embedding_info.prompt_positions[
                            seq.computed_token_num : seq.seq_len
                        ]
                    )
                prefill_works.append(
                    {
                        "kind": "cached",
                        "seq": seq,
                        "embedding_info": embedding_info,
                    }
                )

        # Qwen-VL packs positions as (3, N) and concatenates on the token axis
        # (dim=1); Kimi uses 1-D positions (dim=0). Kimi's result is discarded
        # by callers (``set_mrope_position`` is skipped) but we still build a
        # well-formed tensor.
        if self.uses_mrope:
            mrope_positions = torch.concat(batch_positions, dim=1)
        elif batch_positions:
            mrope_positions = torch.concat(batch_positions, dim=0)
        else:
            mrope_positions = None
        return {
            "prefill_works": prefill_works,
            "mrope_positions": mrope_positions,
            "num_decode_tokens": num_decode_tokens,
        }

    @staticmethod
    def _disagg_ready_len(st: "DisaggSeqState") -> int:
        """Length of the span-aligned ready prefix ``[0, ready_len)``.

        Stops at the first *not-yet-ready* image span start (in token order),
        regardless of whether a later item happens to be ready: a prefix that
        spanned an unready item would have more ``is_multimodal`` positions
        than gathered embeddings and the merge would misalign.
        """
        rl = st.prompt_len
        for i in range(st.num_items):
            if not st.item_ready[i]:
                rl = min(rl, st.item_span[i][0])
        return rl

    def _mm_disagg_collect(
        self,
        seq: Sequence,
        st: "DisaggSeqState",
        prefill_works: List[Dict],
        batch_positions: List[torch.Tensor],
    ) -> None:
        """Build the prefill work for an overlap disagg seq (design §6.2).

        Positions come from the full-prompt mrope grid (all grids known once
        meta arrived). The embedding covers the ready prefix; it is rebuilt
        (kind ``uncached``) whenever the scheduler advances past the cached
        ``coverage_len`` because more items became ready, otherwise the cached
        prefix is re-sliced (kind ``cached``).
        """
        batch_positions.append(
            st.prompt_positions[:, seq.computed_token_num : seq.seq_len]
        )
        info = self.embedding_cache.get(seq.seq_id)
        need_build = info is None or (
            info.coverage_len is not None and seq.seq_len > info.coverage_len
        )
        if not need_build:
            prefill_works.append(
                {"kind": "cached", "seq": seq, "embedding_info": info}
            )
            return
        ready_len = self._disagg_ready_len(st)
        # Gather the ready-prefix items in token-span order so the concatenated
        # embeddings line up 1-1 with the ``is_multimodal`` True positions.
        ready_items = [
            i for i in range(st.num_items) if st.item_span[i][1] <= ready_len
        ]
        ready_items.sort(key=lambda i: st.item_span[i][0])
        ready_embeds = tuple(st.item_embed[i] for i in ready_items)
        prefill_works.append(
            {
                "kind": "uncached",
                "seq": seq,
                "input_ids_cpu": st.input_ids_cpu[:ready_len],
                "is_multimodal_cpu": st.is_multimodal_cpu[:ready_len],
                "mm_input": {},
                "mm_embeddings": ready_embeds if ready_embeds else None,
                "prompt_positions": st.prompt_positions,
                "mrope_position_delta": st.mrope_position_delta,
                "mm_bundle_key": None,
                "coverage_len": ready_len,
            }
        )

    def _mm_run_processor(
        self, seq: Sequence
    ) -> Tuple[Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run image/video processors for ``seq.mm_contents``.

        Returns ``(mm_input, image_grid_thw, video_grid_thw)``. The
        grid tensors are forced to CPU because ``get_input_positions``
        (and our content hashing) does per-element Python indexing on
        them; leaving CUDA tensors there would trigger a D2H sync per
        element and serialize the prepare-input stage against the
        previous batch's forward.
        """
        mm_input: Dict = {}
        image_grid_thw: Optional[torch.Tensor] = None
        video_grid_thw: Optional[torch.Tensor] = None
        if seq.mm_contents is not None and self.is_kimi_mm:
            return self._mm_run_processor_kimi(seq)
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
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.cpu()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.cpu()
        return mm_input, image_grid_thw, video_grid_thw

    def _mm_run_processor_kimi(
        self, seq: Sequence
    ) -> Tuple[Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Kimi-K2.5 image + video preprocessing.

        ``KimiK25VisionProcessor.preprocess`` takes a list of media dicts
        (``{"type":"image",...}`` or ``{"type":"video_chunk",...}``) and returns
        ``pixel_values`` (patchified, ``[sum(t*h*w), 3, ps, ps]``) plus
        ``grid_thws`` (``[num_items, 3]``; video chunks have ``t>1``). We build
        one combined media list in embed order -- all images first, then every
        video's temporal chunks -- matching ``build_kimi_input_ids``'s
        placeholder order and ``embed_multimodal``'s iteration. ``grid_thws`` is
        surfaced as ``image_grid_thw`` so the generic content-hashing path
        (``prod(dim=-1)`` + ``split``) covers every item, while ``grid_thws``
        stays in ``mm_input`` for ``embed_multimodal``.
        """
        from PIL import Image as _PILImage
        from transformers.image_utils import load_image as _hf_load_image

        from gllm.models.kimi_k25_vision import split_video_chunks

        medias = []
        for img_ref in seq.mm_contents["image"]:
            pil = (
                img_ref
                if isinstance(img_ref, _PILImage.Image)
                else _hf_load_image(img_ref)
            )
            medias.append({"type": "image", "image": pil})
        cfg = self.processor.media_processor.media_proc_cfg
        for vid_ref in seq.mm_contents["video"]:
            for chunk in split_video_chunks(vid_ref, cfg):
                medias.append(chunk)

        mm_input: Dict = {}
        image_grid_thw: Optional[torch.Tensor] = None
        if medias:
            preprocessed = self.processor.media_processor.preprocess(
                medias, return_tensors="pt"
            )
            mm_input["pixel_values"] = preprocessed["pixel_values"]
            mm_input["grid_thws"] = preprocessed["grid_thws"]
            image_grid_thw = preprocessed["grid_thws"]
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.cpu()
        return mm_input, image_grid_thw, None

    def _mm_build_is_multimodal_cpu(
        self, seq: Sequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (input_ids_cpu, is_multimodal_cpu) for ``seq``.

        Explicitly CPU-side: the repo sets the default device to CUDA
        via ``ModelLoader``, so a bare ``torch.tensor(...)`` would
        silently allocate on GPU and the ``torch.isin`` below would
        launch a kernel on the default stream -- defeating overlap with
        the previous batch's forward.
        """
        input_ids_cpu = torch.tensor(seq.token_ids, device="cpu")
        placeholder_token_id_cpu = torch.tensor(
            self.model.get_mm_placeholder_token_ids(), device="cpu"
        )
        is_multimodal_cpu = torch.isin(
            input_ids_cpu, placeholder_token_id_cpu
        )
        return input_ids_cpu, is_multimodal_cpu

    def _mm_precompute_hash(self, seq: Sequence) -> None:
        """Pre-build ``seq.hash_token_ids`` before the scheduler's prefix
        cache lookup, so distinct multimodal items don't collide on the
        raw ``<|image_pad|>`` placeholder id.

        The scheduler calls ``pre_allocate_computed_page`` for every new
        seq *before* ``_mm_prepare_cpu`` runs. The default cache key
        function ``_default_cache_key_fn`` reads ``seq.hash_token_ids``
        if present and otherwise falls back to ``seq.token_ids``;
        before this hook existed, that fallback meant every image-bearing
        request used the same placeholder ids at the image span and the
        second request would silently reuse the first request's KV
        pages at the image positions -- producing answers about the
        wrong image (the symptom that previously forced
        ``--no-enable-prefix-caching`` for VL).

        Side effects:
            * ``seq.hash_token_ids`` is populated when the prompt has
              at least one mm item, ``None`` otherwise.
            * ``seq._mm_precomputed`` stashes the heavy outputs of the
              image/video processor + the cpu masks so the later
              ``_mm_prepare_cpu`` pass does not repeat the work.
        """
        if not self.use_mm:
            return
        if seq.mm_contents is None:
            return
        if seq.hash_token_ids is not None or getattr(
            seq, "_mm_precomputed", None
        ) is not None:
            return  # already built (e.g. preemption + re-schedule).

        mm_input, image_grid_thw, video_grid_thw = self._mm_run_processor(seq)
        input_ids_cpu, is_multimodal_cpu = self._mm_build_is_multimodal_cpu(
            seq
        )
        mm_bundle_key, item_hashes = self._build_mm_content_hashes(
            mm_input, image_grid_thw, video_grid_thw
        )
        if item_hashes:
            seq.hash_token_ids = self._splice_mm_pad_ids(
                seq.token_ids, is_multimodal_cpu, item_hashes
            )
        else:
            seq.hash_token_ids = None

        seq._mm_precomputed = {
            "mm_input": mm_input,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "input_ids_cpu": input_ids_cpu,
            "is_multimodal_cpu": is_multimodal_cpu,
            "mm_bundle_key": mm_bundle_key,
        }

    @staticmethod
    def _build_mm_content_hashes(
        mm_input: Dict,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> Tuple[Optional[bytes], List[bytes]]:
        """Hash each MM item's content, return (prompt-level key, per-item).

        Per-item hash mixes pixel bytes + grid shape so two crops of the
        same image with different processor settings still differ. The
        prompt-level key is the concatenation of all per-item digests in
        the order they appear in ``mm_input`` (image items, then video
        items, mirroring :meth:`embed_multimodal`'s iteration). ``None`` is
        returned when there's nothing multimodal in the prompt — that
        signals downstream that the seq is text-only and falls back to the
        cheap ``token_ids`` cache key.
        """
        item_hashes: List[bytes] = []

        pixel_values = mm_input.get("pixel_values")
        if pixel_values is not None and image_grid_thw is not None:
            sizes = image_grid_thw.prod(dim=-1).tolist()
            if isinstance(pixel_values, torch.Tensor):
                chunks = pixel_values.split(sizes, dim=0)
            else:
                chunks = pixel_values
            for chunk, thw in zip(chunks, image_grid_thw):
                item_hashes.append(_hash_tensor_bytes(chunk, thw))

        pixel_values_videos = mm_input.get("pixel_values_videos")
        if pixel_values_videos is not None and video_grid_thw is not None:
            sizes = video_grid_thw.prod(dim=-1).tolist()
            if isinstance(pixel_values_videos, torch.Tensor):
                chunks = pixel_values_videos.split(sizes, dim=0)
            else:
                chunks = pixel_values_videos
            for chunk, thw in zip(chunks, video_grid_thw):
                item_hashes.append(_hash_tensor_bytes(chunk, thw))

        if not item_hashes:
            return None, []
        bundle = hashlib.sha256()
        for h in item_hashes:
            bundle.update(h)
        return bundle.digest(), item_hashes

    @staticmethod
    def _splice_mm_pad_ids(
        token_ids: List[int],
        is_multimodal_cpu: torch.Tensor,
        item_hashes: List[bytes],
    ) -> List[int]:
        """Return a copy of ``token_ids`` with placeholder spans rewritten.

        Each contiguous run of multimodal placeholders is replaced by a
        single ``pad_id`` derived from the next item's content hash, so the
        downstream :class:`PrefixSegment` key naturally diverges between
        prompts whose only difference is the image content. Mirrors
        sglang's ``pad_input_tokens`` trick adapted to gllm's flat-page
        cache layout.
        """
        mask = is_multimodal_cpu.tolist() if isinstance(is_multimodal_cpu, torch.Tensor) else list(is_multimodal_cpu)
        out = list(token_ids)
        n = len(out)
        i = 0
        item_idx = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            # ``item_hashes`` exhaustion would mean the processor produced
            # fewer MM items than there are placeholder spans, which
            # indicates a tokenizer/processor mismatch. We leave excess
            # spans untouched (falls back to the raw token id), which is
            # the safe-but-conservative behavior — at worst it widens the
            # cache hit set, never causing a false hit.
            if item_idx < len(item_hashes):
                pad_id = _mm_pad_id_from_hash(item_hashes[item_idx])
                for k in range(i, j):
                    out[k] = pad_id
                item_idx += 1
            i = j
        return out

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
        # Per-chunk deepstack tensors aligned 1-1 with ``batch_embeddings``.
        # ``None`` means "no deepstack contribution for this chunk" (decode
        # rows, text-only prompts, non-deepstack VL models). A single
        # buffer-write at the end of this method stitches the non-``None``
        # chunks into the right rows of the model's deepstack buffer.
        batch_deepstack: List[Optional[torch.Tensor]] = []
        for work in ctx["prefill_works"]:
            seq = work["seq"]
            if work["kind"] == "uncached":
                # Encoder-disaggregation: embeddings already arrived over NIXL
                # and were cloned into this tuple, so skip the local ViT (the
                # LM node has no vision tower). ``None`` -> monolith path.
                mm_embeddings = work.get("mm_embeddings")
                mm_input = work["mm_input"]
                if mm_embeddings is None and mm_input:
                    # Skip the ViT encoder when this prompt's MM bundle is
                    # already in the cache (e.g. identical-image rerun).
                    # Cache stores the per-item embedding tuple verbatim;
                    # downstream ``embed_input_ids`` is happy with cached
                    # tensors since they live on the same device as the
                    # encoder output.
                    bundle_key = work.get("mm_bundle_key")
                    mm_embeddings = self.mm_embed_cache.get(bundle_key)
                    if mm_embeddings is None:
                        mm_embeddings = self.model.embed_multimodal(**mm_input)
                        if bundle_key is not None:
                            self.mm_embed_cache.put(bundle_key, mm_embeddings)

                # Materialize CPU tensors built in ``_mm_prepare_cpu`` onto
                # the model device for the embed kernels. Sources are small
                # (one prompt's worth of ids) so a synchronous H2D is cheap
                # and avoids the pageable-memory caveats of non_blocking.
                input_ids = work["input_ids_cpu"].to(device, non_blocking=True)
                is_multimodal = work["is_multimodal_cpu"].to(
                    device, non_blocking=True
                )
                embed_result = self.model.embed_input_ids(
                    input_ids,
                    mm_embeddings,
                    is_multimodal,
                )
                # ``embed_input_ids`` returns either a plain embedding
                # tensor (non-deepstack models / text-only prompts that
                # don't bother building the tuple) or
                # ``(embedding, deepstack)``. Unify to a 2-tuple for the
                # downstream layout code.
                if isinstance(embed_result, tuple):
                    prompt_embeddings, prompt_deepstack = embed_result
                else:
                    prompt_embeddings, prompt_deepstack = embed_result, None

                embedding_info = EmbeddingInfo(
                    prompt_embeddings,
                    work["prompt_positions"],
                    work["mrope_position_delta"],
                    deepstack_embedding=prompt_deepstack,
                    coverage_len=work.get("coverage_len"),
                )
                self.embedding_cache[seq.seq_id] = embedding_info
                embedding = prompt_embeddings[
                    seq.computed_token_num : seq.seq_len, :
                ]
                deepstack_chunk = (
                    prompt_deepstack[
                        :, seq.computed_token_num : seq.seq_len, :
                    ]
                    if prompt_deepstack is not None
                    else None
                )
            else:
                embedding_info = work["embedding_info"]
                embedding = embedding_info.embedding[
                    seq.computed_token_num : seq.seq_len, :
                ]
                deepstack_chunk = (
                    embedding_info.deepstack_embedding[
                        :, seq.computed_token_num : seq.seq_len, :
                    ]
                    if embedding_info.deepstack_embedding is not None
                    else None
                )

            if seq.seq_len == seq.prompt_len:
                # Prefill just finished; drop the cached embedding tensors
                # to free memory. We still keep mrope_position_delta around
                # for future decode-position calculations.
                embedding_info.embedding = None
                embedding_info.deepstack_embedding = None
                # Encoder-disaggregation: the per-item visual embeddings are now
                # baked into the KV cache, so the (cloned) gate-B copies can go.
                self.disagg_embeds.pop(seq.seq_id, None)

            batch_embeddings.append(embedding)
            batch_deepstack.append(deepstack_chunk)

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
            # Decode rows must contribute zero deepstack residual (they
            # represent already-prefilled tokens whose visual residuals
            # are baked into the KV cache, not the input embedding).
            batch_deepstack.insert(0, None)

        if not batch_embeddings:
            return None

        # Stitch per-chunk deepstack tensors into the model's per-batch
        # buffer at the offsets matching the final concatenated layout.
        # This makes ``model._get_deepstack_input_embeds(num_tokens)``
        # return rows aligned 1-1 with ``hidden_states`` regardless of
        # prefix-cache hits or chunked prefill -- the deepstack residual
        # for a token T at batch row R will land exactly at buffer row R.
        if any(d is not None for d in batch_deepstack) and hasattr(
            self.model, "_set_deepstack_input_embeds"
        ):
            total_tokens = sum(e.shape[0] for e in batch_embeddings)
            # Zero positions that no chunk will write to (decode rows,
            # text-only chunks). ``_clear_deepstack_input_embeds`` after
            # the previous forward only zeroed up to that batch's row
            # count, so anything beyond it could still hold stale values.
            self.model._clear_deepstack_input_embeds(total_tokens)
            offset = 0
            for chunk, emb in zip(batch_deepstack, batch_embeddings):
                n = emb.shape[0]
                if chunk is not None:
                    self.model._set_deepstack_input_embeds(
                        chunk, offset=offset
                    )
                offset += n

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
            # Kimi keeps the plain 1-D positions set by ``cal_and_set_input``
            # above; only the Qwen-VL family overrides with 3-D mrope.
            if self.uses_mrope:
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
        if self.uses_mrope:
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
                if self.uses_mrope:
                    self.input_data.set_mrope_position(torch.zeros((3, size), device="cpu"))
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
                    self.forward()
                self.size_to_graph[size] = g
        if torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    def _fixup_vl_decode_embeddings(self, num_decode_tokens: int) -> None:
        """Re-embed decode-token IDs into the front of ``input_hidden_states``.

        ``_mm_prepare_gpu`` inserts an ``torch.empty()`` placeholder for the
        decode rows of every VL batch and relies on this method to overwrite
        those rows with the real text embeddings *before* the model forward
        reads them. Both the no-overlap base path (:meth:`forward`) and the
        overlap path (:meth:`OverlapModelRunner.run_batch_async`) must call
        this; otherwise the model consumes uninitialized memory for every
        decode token and silently produces garbage from the first decode
        step onward.
        """
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
    def forward(self):
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        if is_first_pp_rank() and self.use_mm:
            # See ``_fixup_vl_decode_embeddings`` for why this is required
            # without overlap scheduling.
            num_decode_tokens = sum(
                1 for s in self.input_data.seqs if s.computed_prompt
            )
            self._fixup_vl_decode_embeddings(num_decode_tokens)
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

    # ------------------------------------------------------------------
    # Encoder-disaggregation overlap (design §6.2)
    # ------------------------------------------------------------------

    def disagg_register(self, seq_id: int, state: DisaggSeqState) -> None:
        """Register a disagg seq for overlapped, readiness-gated prefill.

        Called by the LM disagg manager once *all* per-item ``MmItemMeta`` have
        arrived (positions/hashes determined; gate A satisfied) but before the
        visual embeddings have necessarily landed. The embeddings are filled in
        progressively via :meth:`disagg_set_embedding`.
        """
        self.disagg_embeds[seq_id] = state

    def disagg_set_embedding(
        self, seq_id: int, ordered_idx: int, embed: torch.Tensor
    ) -> None:
        """Record one item's visual embedding (NIXL write completed)."""
        st = self.disagg_embeds.get(seq_id)
        if st is None:
            return
        st.item_embed[ordered_idx] = embed
        st.item_ready[ordered_idx] = True

    def disagg_prefill_limit(self, seq: Sequence) -> Optional[int]:
        """Gate-B upper bound (design §6.2): the largest token position this
        seq may prefill up to this round = the start of the first image span
        whose embedding hasn't landed yet (or ``prompt_len`` if all ready).
        ``None`` for non-disagg seqs (no cap).

        This deliberately matches :meth:`_disagg_ready_len` (the embed coverage)
        so the scheduler never advances ``computed_token_num`` past the embed
        coverage -- even when a prefix-cache hit would otherwise jump the cursor
        over an item whose embedding is still in flight. Such a (rare) seq waits
        for the embedding to land, then proceeds; the encoder's own embed cache
        keeps that wait short for repeated content.
        """
        st = self.disagg_embeds.get(seq.seq_id)
        if st is None:
            return None
        return self._disagg_ready_len(st)

    def free(self, seq: Sequence):
        self.memory_manager.free(seq)
        if self.use_mm and is_first_pp_rank():
            self.embedding_cache.pop(seq.seq_id, None)
            self.disagg_embeds.pop(seq.seq_id, None)

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
            self.disagg_embeds.pop(seq_id, None)


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
        # Route hybrid (GDN/Mamba) prefix-cache snapshot restores onto
        # ``forward_stream``. The snapshot WRITE runs inside the forward on
        # this stream; the restore is issued later from the scheduler on the
        # CPU thread (otherwise the default stream), so without sharing a
        # stream the restore could read a snapshot the in-flight forward has
        # not finished writing. Same-stream FIFO ordering closes that race.
        if getattr(self.memory_manager, "ssm_segment", None) is not None:
            self.memory_manager.ssm_segment.restore_stream = self.forward_stream
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
                # Kimi uses plain 1-D positions (already copied into the input
                # buffer above); only Qwen-VL overrides with 3-D mrope.
                if self.uses_mrope:
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
