import hashlib
import os
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
    get_dp_size,
    get_ipc_tp_group,
    get_local_rank,
    get_output_rank,
    get_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
    is_dp_attn,
    is_first_pp_rank,
    is_last_pp_rank,
    is_output_rank,
    set_dp_forward_counts,
)
from gllm.input_data import InputData
from gllm.layers.rotary_embedding import MRotaryEmbedding
from gllm.layers.sampler import Sampler
from gllm.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.model_loader import ModelLoader, propagate_serving_config
from gllm.mtp_gpu_prep import MtpGpuPrep
from gllm.sequence import Sequence
from gllm.utils import unify_decode


@dataclass
class MtpQDist:
    """The MTP draft chain's proposal distribution ``q``, in one of two forms.

    * **dense** -- ``[nd, k, vocab]`` transformed probabilities (needed when a
      request leaves ``top_k`` unrestricted, so the support isn't bounded).
    * **sparse** -- ``vals``/``idx`` ``[nd, k, k_pad]``: q's top-k support and its
      token ids, plus ``drawn`` ``[nd, k]``, the probability of the token each
      step actually sampled. Anything the rejection accept needs is recoverable
      from this (q is exactly 0 outside the support), at 1/500th the bytes.
    """

    dense: Optional[torch.Tensor] = None
    vals: Optional[torch.Tensor] = None
    idx: Optional[torch.Tensor] = None
    drawn: Optional[torch.Tensor] = None

    @property
    def is_sparse(self) -> bool:
        return self.dense is None


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
        maxp,
        maxd,
        minp,
        iterp,
        init_new_token_ratio,
        min_new_token_ratio,
        schedule_method: str,
        disable_cuda_graph: bool,
        max_cuda_graph_bs: int,
        model_max_length: int,
        mm_processor_min_pixels: int = None,
        mm_processor_max_pixels: int = None,
        skip_visual: bool = False,
        skip_language: bool = False,
        mla_decode_backend: str = "fa3",
        mla_cache_dtype: str = "bf16",
        mamba_ssm_cache_dtype: str = "auto",
        mtp_enabled: Optional[bool] = None,
        mtp_k: int = 3,
        mtp_max_batch: int = 0,
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
        self.gpu_memory_util = gpu_memory_util
        self.page_size = page_size
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
            AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        )
        # DeepSeek-V3.2 ships no usable chat_template; it bundles the official
        # message encoder at ``<model_path>/encoding/encoding_dsv32.py``. Flag it
        # so ``encode`` renders the reference DSML prompt (thinking gating, tool
        # calls) instead of a hand-written Jinja template. We only store a bool
        # (the loaded encoder is a module object and is NOT picklable -- the
        # runner is pickled to spawn TP workers); the encoder itself is
        # lazy-loaded per process inside ``encode`` via a module-level cache.
        self._use_dsv32_encoder = (
            getattr(self.model_loader, "architecture", None)
            == "DeepseekV32ForCausalLM"
        )
        self.maxp = maxp
        self.maxd = maxd
        self.minp = minp
        self.iterp = iterp
        # Adaptive KV-cache admission control (see Scheduler). ``init`` is the
        # starting/relaxed-ceiling fraction of remaining output we reserve for
        # running decodes; ``min`` is the floor the ratio decays toward when
        # the system is stable.
        self.init_new_token_ratio = init_new_token_ratio
        self.min_new_token_ratio = min_new_token_ratio
        self.schedule_method = schedule_method
        self.sampler = Sampler()
        # Per-batch-row generation logprobs from the most recent ``step_once``
        # (non-overlap path); consumed by the worker and carried alongside the
        # sampled tokens (incl. over the token socket under PP>1). ``None`` when
        # the last batch requested no logprobs.
        self._last_logprobs = None
        # Prompt logprobs that finished prefill on the most recent forward,
        # keyed by seq_id (``{seq_id: prompt_logprobs_data}``). Only used under
        # PP>1, where the output-rank follower computes them and ships them to
        # rank 0 over the token socket alongside the sampled tokens; rank 0
        # attaches them in ``process_output``. Empty on the common path. (PP=1
        # attaches directly from the local seq in the scheduler instead.)
        self._last_prompt_logprobs = {}

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
        # default is FA3 (SGLang-compatible absorbed MLA decode). FlashMLA
        # requires a KV page size of 64; bump page_size automatically when
        # that backend is selected. The attention layer performs the final
        # availability check and falls back when the kernel cannot run.
        # 64 == required FlashMLA block size (kept as a literal to avoid
        # importing the CUDA-heavy attention module in the parent process).
        _FLASHMLA_PAGE_SIZE = 64
        self.mla_decode_backend = (mla_decode_backend or "fa3").lower()
        if self.mla_decode_backend not in ("triton", "flashmla", "fa3"):
            raise ValueError(
                "mla_decode_backend must be 'fa3', 'flashmla', or 'triton', "
                f"got {self.mla_decode_backend!r}."
            )
        # MLA latent KV cache precision (DeepSeek Sparse Attention). "bf16"
        # (default) = full-precision latent cache + dense decode; "fp8" = native
        # FP8-packed cache driving FlashMLA sparse decode on SM90.
        self.mla_cache_dtype = (mla_cache_dtype or "bf16").lower()
        if self.mla_cache_dtype not in ("bf16", "fp8"):
            raise ValueError(
                f"mla_cache_dtype must be 'bf16' or 'fp8', got {self.mla_cache_dtype!r}."
            )
        # Recurrent (SSM/GDN) state cache precision for hybrid linear-attention
        # models, named + defaulted like vLLM's ``--mamba-ssm-cache-dtype``:
        # "auto" stores the state in the model's activation dtype (bf16 for these
        # checkpoints), which is what vLLM ships as its default. The Qwen3.5
        # checkpoints *hint* fp32 via ``mamba_ssm_dtype``; that hint is ignored
        # unless asked for explicitly, since the recurrence accumulates in fp32
        # inside the kernels either way and the stored state is 2x the bytes in
        # fp32 (1 MiB vs 0.5 MiB per layer per block on Qwen3.5-0.8B, and the MTP
        # verify writes ``1+k`` of them per sequence per layer per step).
        self.mamba_ssm_cache_dtype = (mamba_ssm_cache_dtype or "auto").lower()
        if self.mamba_ssm_cache_dtype not in (
            "auto", "bfloat16", "float16", "float32"
        ):
            raise ValueError(
                "mamba_ssm_cache_dtype must be 'auto', 'bfloat16', 'float16' or "
                f"'float32', got {self.mamba_ssm_cache_dtype!r}."
            )
        self.model_loader.config.mamba_ssm_cache_dtype = self.mamba_ssm_cache_dtype
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
        # MTP (multi-token prediction) config. ``mtp_enabled=None`` auto-detects:
        # enable iff the checkpoint declares nextn-predict layers. Stamped onto
        # the model config so the model builder (e.g. DeepseekV32ForCausalLM,
        # Qwen3_5ForCausalLM) constructs the MTP head from config instead of an
        # env var. ``mtp_k`` is the draft-chain length.
        #
        # Two config conventions are supported:
        #   * DeepSeek V3/V3.2, GLM-MoE-DSA: top-level ``num_nextn_predict_layers``.
        #   * Qwen3.5 (hybrid GDN): ``text_config.mtp_num_hidden_layers`` (the
        #     multimodal wrapper nests the text config; the MTP head is a single
        #     full-attention block shipped under ``mtp.*``).
        _cfg = self.model_loader.config
        _text_cfg = getattr(_cfg, "text_config", None) or _cfg
        _num_nextn = (
            getattr(_cfg, "num_nextn_predict_layers", 0)
            or getattr(_text_cfg, "mtp_num_hidden_layers", 0)
            or 0
        )
        self._mtp_enabled = (
            (_num_nextn >= 1) if mtp_enabled is None else bool(mtp_enabled)
        )
        self._mtp_k_cfg = mtp_k
        self._mtp_max_batch_cfg = mtp_max_batch
        self.model_loader.config.mtp_enabled = self._mtp_enabled
        # Nested text config (Qwen3.5-VL wrapper) reads ``mtp_enabled`` off its
        # own config object, so mirror the flag there too.
        if _text_cfg is not _cfg:
            _text_cfg.mtp_enabled = self._mtp_enabled
        propagate_serving_config(self.model_loader.config)

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
        # ``max_cuda_graph_bs`` cannot exceed *either* of two runtime bounds:
        #
        #   * ``maxd`` — the decode batch is hard-capped at ``maxd`` (scheduler)
        #     and several device buffers (``InputData.block_table``,
        #     ``slot_mapping``, the SSM working pool, ...) are sized at ``maxd``
        #     rows.
        #   * ``max_num_batched_tokens`` — a captured decode graph of ``B``
        #     sequences writes ``B`` token-rows into the shared activation
        #     buffers (``input_hidden_states`` / ``residual`` / PP recv), which
        #     are sized to ``max_num_batched_tokens``. Under ``chunked_prefill``
        #     / ``split_pd`` that equals ``maxp``, so a small ``--maxp`` (below
        #     the ``--max-cuda-graph-bs`` default of 512) would overflow those
        #     buffers *during capture* — e.g. ``--maxp 256`` tried to write 512
        #     rows into a 256-row buffer and crashed with a shape mismatch on
        #     ``ssm_state_indices`` / the output hidden states.
        #
        # A real forward never batches more than ``max_num_batched_tokens``
        # tokens (decode eats into the same per-tick budget as prefill), so
        # buckets above that bound are never replayed anyway. Clamp to the
        # tighter of the two so users can keep the ``--max-cuda-graph-bs``
        # default without manually matching ``maxd`` / ``maxp``.
        cuda_graph_cap = min(maxd, self.max_num_batched_tokens)
        if max_cuda_graph_bs > cuda_graph_cap:
            logger.warning(
                f"max_cuda_graph_bs={max_cuda_graph_bs} exceeds the runtime "
                f"decode-batch bound min(maxd={maxd}, "
                f"max_num_batched_tokens={self.max_num_batched_tokens})="
                f"{cuda_graph_cap}; clamping to {cuda_graph_cap}."
            )
            max_cuda_graph_bs = cuda_graph_cap
        self.max_cuda_graph_bs = max_cuda_graph_bs
        self.size_to_graph: Dict[int, torch.cuda.CUDAGraph] = dict()
        # Use power-of-two bucket sizes to reduce the number of captured graphs.
        # At runtime the actual batch is padded up to the nearest bucket.
        self.capture_sizes = self._build_capture_sizes(self.max_cuda_graph_bs)

        # max length
        self.model_max_length = self.resolve_model_max_length(model_max_length)

        # ``InputData``'s per-token buffers are sized ``model_max_length`` (the
        # longest single sequence), but a prefill batch may carry
        # ``max_num_batched_tokens`` (= ``maxp``) tokens. With ``maxp >
        # model_max_length`` the very first thing that happens -- the profile
        # run's full-size dummy prefill -- overflows those buffers and dies deep
        # inside ``copy_to_input_buffer`` with a bare shape mismatch
        # (``size of tensor a (4096) must match tensor b (8192)``), which says
        # nothing about the actual misconfiguration. Clamp + say so instead: a
        # prefill batch can never usefully exceed one sequence's max length,
        # since chunked prefill already splits longer prompts.
        if self.max_num_batched_tokens > self.model_max_length:
            logger.warning(
                f"maxp/max_num_batched_tokens={self.max_num_batched_tokens} "
                f"exceeds model_max_length={self.model_max_length}; clamping to "
                f"{self.model_max_length} (the input buffers are sized to one "
                f"sequence's max length). Raise --model-max-length if you want a "
                f"larger prefill batch."
            )
            self.max_num_batched_tokens = self.model_max_length
            # Keep the loader + the config it already stamped in sync: models
            # size workspaces from ``config.max_num_batched_tokens``.
            self.model_loader.max_num_batched_tokens = self.max_num_batched_tokens
            if getattr(self.model_loader, "config", None) is not None:
                self.model_loader.config.max_num_batched_tokens = (
                    self.max_num_batched_tokens
                )

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
        # MTP speculative decoding: number of draft tokens per step (k). Active
        # only when the model built an MTP head (mtp_enabled + nextn layers).
        self._mtp_k = (
            self._mtp_k_cfg
            if getattr(self.model, "mtp", None) is not None else 0
        )
        # Hybrid models (Qwen3.5 GDN) advertise a ready-to-use SSM cache
        # config via ``model.ssm_cache_config``. ``num_layers`` for the KV
        # path must then be the count of *full-attention* layers only.
        ssm_cache_config = getattr(self.model, "ssm_cache_config", None)
        memory_manager_cls = (
            PrefixMemoryManager if self.enable_prefix_caching else MemoryManager
        )
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
            # DeepSeek Sparse Attention (V3.2): non-zero => allocate a parallel
            # paged indexer key cache. 0 for every other model.
            index_head_dim=getattr(self.model, "index_head_dim", 0),
            # MLA rope head dim (needed to size the native FP8 MLA cache for DSA).
            qk_rope_head_dim=getattr(self.model, "qk_rope_head_dim", 0),
            # DSA MLA latent cache precision: FP8-packed only when explicitly
            # requested (drives SM90 sparse decode); default bf16 + dense decode.
            mla_cache_fp8=(self.mla_cache_dtype == "fp8"),
            # MTP draft-chain length for hybrid GDN models: each running seq may
            # borrow up to mtp_k transient checkpoint blocks from the shared SSM
            # block pool during a verify step. 0 for non-MTP or non-hybrid.
            mtp_k=(
                self._mtp_k
                if (ssm_cache_config is not None and self._mtp_k > 0)
                else 0
            ),
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
        # MTP draft-step CUDA-graph buffers. A draft step is a batch x 1-token
        # decode of the MTP head; we capture one graph per decode bucket and
        # replay it k times, advancing tok/hidden/positions/seq_lens/slot in
        # place on the GPU (no Python / H2D / .item() per step). Gated on MTP
        # active + graphs enabled + env opt-out. Buffers + the aliasing
        # ``_draft_input`` are lazily built in ``_init_draft_graph_state`` after
        # the model + memory manager exist.
        self._mtp_draft_graph = (
            getattr(self, "_mtp_k", 0) > 0
            and getattr(self.model, "mtp", None) is not None
            and not self.disable_cuda_graph
            # Works for both MLA (DeepSeek) and non-MLA (Qwen3.5 GDN): the draft
            # step captures ``mtp.forward`` (a single decoder layer, no dynamic
            # ops / host sync). The only MLA-specific replay op (advancing
            # ``decode_seq_lens``) is guarded in ``_draft_chain_graph``.
        )
        self._draft_size_to_graph: Dict[int, torch.cuda.CUDAGraph] = {}
        # Separate captured graphs for the sampled (rejection) draft step, which
        # runs Gumbel-max sampling + q-dist stash instead of argmax.
        self._draft_size_to_graph_sampled: Dict[int, torch.cuda.CUDAGraph] = {}
        # Sparse (top-k) sampled-draft graphs; see
        # ``_draft_step_forward_sampled_sparse``.
        self._draft_size_to_graph_sampled_sparse: Dict[int, torch.cuda.CUDAGraph] = {}
        self._draft_input = None
        # MTP verify CUDA graph: capture the full target verify forward (over the
        # uniform 1+k query per decode seq) per bucket at init and replay it. The
        # verify forward is 99% of MTP step time and is pure eager per-layer launch
        # overhead (~250ms, ~constant vs batch size), so graphing it is the main
        # speedup lever. Requires the fp8 decode-sparse verify kernel (graph-safe).
        self._mtp_verify_graph = (
            getattr(self, "_mtp_k", 0) > 0
            and getattr(self.model, "mtp", None) is not None
            and not self.disable_cuda_graph
            # Works for MLA (DeepSeek fp8 decode-sparse kernel) and non-MLA
            # (Qwen3.5 GDN): the verify forward reads only static input buffers
            # (incl. the 2D SSM block table + num_accepted static buffers filled
            # by copy_to_input_buffer), so it is graph-capturable for both.
        )
        self._verify_size_to_graph: Dict[int, torch.cuda.CUDAGraph] = {}
        self._verify_k = getattr(self, "_mtp_k", 0)
        # Fused MTP (default ON, opt out with GLLM_MTP_FUSED=0): eliminate the
        # separate x1-decode forward by relaying each step's verify bonus token +
        # its hidden as the NEXT step's draft seed. One target forward (verify)
        # per step instead of two. ``_mtp_relay`` maps seq_id -> (seed_tok:int,
        # seed_hidden:tensor[H]) carried across consecutive steps; a seq missing
        # from it (freshly admitted / batch reshuffle) forces the bootstrap path
        # (decode forward) for that step.
        self._mtp_fused = (
            getattr(self, "_mtp_k", 0) > 0
            and getattr(self.model, "mtp", None) is not None
            and os.environ.get("GLLM_MTP_FUSED", "1") == "1"
        )
        self._mtp_relay: Dict[int, tuple] = {}
        # Batch-adaptive MTP (vLLM's ``disable_by_batch_size`` analogue).
        # Speculating multiplies the per-step target work by ``1+k``; it wins only
        # while the decode batch leaves the GPU under-utilized. Past the crossover
        # a plain 1-token step is strictly faster, so skip MTP for that step (the
        # plain path is already the bootstrap path -- see ``step_once`` -- so this
        # is a scheduling decision, not a second code path). ``0`` disables the
        # gate (always speculate).
        self._mtp_max_batch = int(getattr(self, "_mtp_max_batch_cfg", 0) or 0)
        self._mtp_spec_decision = None
        if self._mtp_k > 0 and self._mtp_max_batch > 0:
            logger.info(
                f"MTP batch gate: speculating only while the decode batch is "
                f"<= {self._mtp_max_batch} seqs; larger batches take a plain "
                f"decode step."
            )
        # GPU-native MTP input prep (vLLM model-runner-V2 style; see
        # ``gllm/mtp_gpu_prep.py``). Replaces the per-step Python rebuild of the
        # draft / verify input arrays with persistent pinned staging + a few
        # vectorized CUDA ops writing straight into the static graph buffers.
        # ``GLLM_MTP_GPUPREP=0`` falls back to the CPU builders (``cal_input``).
        self._mtp_gpu_prep = None
        self._mtp_gpu_prep_on = (
            getattr(self, "_mtp_k", 0) > 0
            and getattr(self.model, "mtp", None) is not None
            and os.environ.get("GLLM_MTP_GPUPREP", "1") == "1"
        )
        # Persistent pinned/device staging for the per-seq sampling params (see
        # ``_mtp_sample_params``); lazily allocated on first use.
        self._sp_host_f = None
        self._sp_host_k = None
        self._sp_dev_f = None
        self._sp_dev_k = None
        if self._mtp_gpu_prep_on:
            self._mtp_gpu_prep = MtpGpuPrep(
                max_bs=max(self.max_running_seqs, 1),
                max_blocks=self.input_data.max_num_block,
                bt_width=(
                    1 + self._mtp_k
                    if self.memory_manager.use_ssm_cache
                    else 0
                ),
                page_size=self.page_size,
                uses_mrope=self.uses_mrope,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        # MTP rejection sampling: make MTP distribution-lossless under
        # temperature/top-p instead of greedy-only. Activated per-batch by
        # RUNTIME DETECTION of any non-greedy seq -- NOT an env flag: a greedy
        # batch takes the argmax fast path, a batch with any sampling seq takes
        # the lossless rejection path. ``_mtp_can_sample`` just means "MTP is
        # active" and gates allocating/capturing the sampled-draft buffers +
        # graphs at init (any request may sample, so they must always be ready).
        # TP consistency of the stochastic draws is handled by ``_mtp_rng``
        # (TP-synced per-step
        # seed) + broadcasting drawn tokens, since independent per-rank sampling
        # would otherwise diverge the KV caches.
        self._mtp_can_sample = (
            getattr(self, "_mtp_k", 0) > 0
            and getattr(self.model, "mtp", None) is not None
        )
        self._mtp_rng = None  # lazily built on the compute device
        self._mtp_step = 0
        # Device-side counter of sparse-top-k tie overflows (see
        # ``_mtp_sparse_probs``); read + reported at the 1 Hz metrics log so the
        # hot path never syncs on it.
        self._mtp_tie_overflow = torch.zeros((), dtype=torch.int64, device="cuda")
        # Bumped once per ``_mtp_decode`` so the GPU prep can memoize its
        # per-step staging across the draft and verify phases.
        self._mtp_prep_epoch = 0
        # Profile run
        self.profile_run()
        # Init KV cache at last; only reserve the dummy page when CUDA graphs
        # are actually enabled so we don't waste memory otherwise.
        self.memory_manager.init(reserve_dummy_page=not self.disable_cuda_graph)

        if not self.disable_cuda_graph:
            self.capture_graph()

    def encode(
        self,
        messages,
        chat: bool = False,
        has_mm: bool = False,
        chat_template_kwargs: Optional[Dict] = None,
        tools: Optional[list] = None,
    ):
        # Per-request chat-template variables (e.g. ``{"thinking": False}`` /
        # ``{"enable_thinking": False}``) forwarded straight from the request's
        # ``chat_template_kwargs``. Different chat templates gate "thinking"
        # mode via different variable names (Qwen3/3.5 read ``enable_thinking``,
        # Kimi-K2.5 reads ``thinking``); Jinja silently ignores undefined
        # template variables, so a client can send both. When omitted, the
        # model's own chat-template default applies (there is no server-wide
        # thinking flag anymore).
        #
        # ``tools`` (the request's OpenAI-style function schemas) are forwarded
        # so the chat template renders the model's tool-declaration block (e.g.
        # Kimi's ``<|im_system|>tool_declare<|im_middle|>...``). Without this the
        # model never learns the tools exist and answers as if it had none.
        template_kwargs = dict(chat_template_kwargs or {})
        if tools:
            template_kwargs["tools"] = tools
        if chat:
            # OpenAI-style requests may send ``content: null`` (e.g. an assistant
            # turn that only carries ``tool_calls``). Many chat templates assume
            # ``content`` is a str or list and iterate it in the non-string
            # branch, so a None surfaces as ``TypeError: 'NoneType' object is not
            # iterable`` mid-render. Normalize null content to "" before render.
            for message in messages:
                if isinstance(message, dict) and message.get("content") is None:
                    message["content"] = ""
            dsv32_encoder = None
            if self._use_dsv32_encoder:
                from gllm.tokenizers.deepseek_v32 import load_dsv32_encoder

                dsv32_encoder = load_dsv32_encoder(self.model_path)
            if dsv32_encoder is not None and (not self.use_mm or not has_mm):
                # DeepSeek-V3.2: render with the model's official encoder
                # (reference DSML format) instead of a Jinja chat template.
                from gllm.tokenizers.deepseek_v32 import (
                    apply_dsv32_chat_template,
                )

                out = apply_dsv32_chat_template(
                    dsv32_encoder,
                    messages,
                    self.tokenizer,
                    tokenize=True,
                    **template_kwargs,
                )
            elif not self.use_mm or not has_mm:
                out = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **template_kwargs,
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
                    tokenize=False,
                    **template_kwargs,
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
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **template_kwargs,
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

    def encode_skeleton(self, messages, chat_template_kwargs: Optional[Dict] = None):
        """Text-only tokenization with one sentinel per mm item (design §5.4).

        Used by the disaggregated LM frontend instead of the multimodal
        ``processor.apply_chat_template``: no pixels are opened or processed
        here, and each image/video collapses to a single placeholder id that
        the LM PP0 later expands to ``N_vis_i`` tokens. Returns the skeleton
        token-id list. ``chat_template_kwargs`` carries per-request chat-template
        variables (e.g. ``{"thinking": False}``) straight from the request.
        """
        from gllm.mm_common import tokenize_text_only

        cfg = self.model_loader.config
        skel = tokenize_text_only(
            self.tokenizer,
            messages,
            image_token_id=int(cfg.image_token_id),
            video_token_id=int(cfg.video_token_id),
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
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
                if self.uses_mrope:
                    # mrope (Qwen-VL): decode positions are extrapolated from
                    # the prefill-time ``mrope_position_delta`` stashed in the
                    # embedding cache, so the entry must exist here.
                    embedding_info = self.embedding_cache[seq.seq_id]
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
                    # downstream ``torch.concat`` happy. We deliberately do NOT
                    # read ``embedding_cache`` here: a Kimi decode seq does not
                    # need its prefill embedding (positions are a plain
                    # ``arange`` and the row is re-embedded by the decode fixup),
                    # and a text-only prompt may never have created an entry --
                    # touching it would raise ``KeyError`` and crash the engine.
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

    def mtp_speculate_batch(self, num_decodes: int) -> bool:
        """Should this step speculate, given the decode batch size?

        A pure function of ``num_decodes``, which every rank sees identically, so
        TP/PP ranks always agree without a collective (MTP's sync design requires
        TP-identical tokens). Callers that decline must also drop the fused
        relay -- see ``_mtp_drop_relay``.
        """
        if self._mtp_k <= 0 or getattr(self.model, "mtp", None) is None:
            return False
        # The decision is taken ONCE per iteration (``mtp_begin_iter``) and
        # cached: the four gate sites must agree within a step. If they could
        # disagree, a step whose prep was skipped as "fused MTP" could then take
        # the plain path -- running the plain forward on minimal input prep.
        cached = self._mtp_spec_decision
        if cached is not None and cached[0] == num_decodes:
            return cached[1]
        return self._mtp_decide(num_decodes)

    def _mtp_decide(self, num_decodes: int) -> bool:
        dec = self._mtp_max_batch <= 0 or num_decodes <= self._mtp_max_batch
        self._mtp_spec_decision = (num_decodes, dec)
        return dec

    def mtp_begin_iter(self, num_decodes: Optional[int]) -> bool:
        """Open an iteration: decide once whether it speculates, and cache it.

        ``num_decodes=None`` marks a prefill / mixed / idle iteration, which never
        speculates. Called by both worker loops before any prep, so every later
        gate site reads the same decision.
        """
        if not num_decodes or self._mtp_k <= 0 or (
            getattr(self.model, "mtp", None) is None
        ):
            self._mtp_spec_decision = None
            return False
        return self._mtp_decide(num_decodes)

    def _mtp_drop_relay(self) -> None:
        """Invalidate the fused relay after a step that did not speculate.

        A plain decode step advances every seq by one token without refreshing
        the relay, so the stashed ``(bonus_tok, bonus_hidden)`` no longer
        describes the seq's last position. Reusing it would seed the next draft
        from a stale hidden. Clearing costs one bootstrap (non-fused) step when
        the batch drops back below the threshold.
        """
        if self._mtp_relay:
            self._mtp_relay = {}

    def mtp_fused_prep_eligible(self, seqs: List[Sequence]) -> bool:
        """True when :meth:`step_once` will take the fused MTP fast path.

        The fused path never runs a decode forward -- it goes straight to
        ``_mtp_decode``, whose draft/verify prep overwrites every input buffer.
        So building the decode batch's per-token arrays first (``prepare_input``
        -> ``cal_input``, ~1.5 ms at nd=64, plus the VL mrope/embedding pass) is
        pure waste. Callers use this to decide between the full prep and
        :meth:`prepare_input_mtp_fused`. The predicate mirrors the gate in
        ``step_once`` exactly: same-mode batch, relay present for every seq.
        """
        if not (self._mtp_fused and not is_dp_attn() and is_last_pp_rank()):
            return False
        if not seqs or not seqs[-1].computed_prompt:
            return False   # prefill / mixed batch
        if not self.mtp_speculate_batch(len(seqs)):
            return False   # batch too large to profit from speculation
        return all(s.seq_id in self._mtp_relay for s in seqs)

    def prepare_input_mtp_fused(self, seqs: List[Sequence]) -> None:
        """Minimal prep for a fused MTP step: batch bookkeeping only.

        Sets just what ``step_once``'s fused gate and ``_mtp_decode`` read
        (``seqs`` + the decode/prefill split). No per-token arrays, no H2D, no
        multimodal pass -- ``_mtp_decode``'s GPU-native prep fills the device
        buffers for the draft and verify forwards from scratch.
        """
        idata = self.input_data
        idata.seqs = seqs
        idata.embedding_size = 0
        idata.is_mtp_verify = False
        idata.num_decodes = len(seqs)
        idata.num_decode_tokens = len(seqs)
        idata.num_prefills = 0
        idata.max_query_len = 1

    def create_dummy_seqs(self, size, runtime: bool = False):
        """Dummy 1-token decode seqs (graph capture / bucket padding).

        Pass ``runtime=True`` for any dummy batch built while real requests are
        in flight. The default ``page_table = [seq_id]`` points at pages
        ``0..size-1``, which are unowned during init-time capture but at runtime
        belong to *live* sequences -- those rows would then scribble their dummy
        KV over real sequences' cache (silent output corruption). ``runtime``
        redirects every row to the reserved ``dummy_page`` instead, which is what
        ``pad_for_cuda_graph`` already does for the decode-graph padding.
        """
        page = self.memory_manager.dummy_page if runtime else None
        seqs = [Sequence(idx, [1, 2], [], output_len=1) for idx in range(size)]
        for seq in seqs:
            seq.page_table.append(seq.seq_id if page is None else page)
            seq.prompt_len = 1
            seq.computed_token_num = 1
            seq.to_compute_token_num = 1
        return seqs

    def create_dummy_prefill_seqs(self, total_tokens):
        """Build a dummy *prefill* batch totalling ``total_tokens`` tokens.

        The largest single forward the engine can issue is a full prefill of
        ``max_num_batched_tokens`` tokens (the input buffers are sized to
        exactly this), so this is the batch shape that drives peak activation
        memory. Profiling with it lets :meth:`profile_run` size the KV cache
        from what is *actually* left after the worst-case forward.

        This matters most under DP-attention + EP: the MoE ``dp_gather`` runs
        the experts over ``dp_size x local_tokens``, so a decode-shaped dummy
        (1 token/seq, ``max_running_seqs`` seqs) under-measures the MoE
        activation by roughly ``max_num_batched_tokens / max_running_seqs``.
        The profiler then over-reserves KV and the first real prefill OOMs.

        Attention is skipped during profiling (the KV segment is only built in
        ``MemoryManager.init`` afterwards -- see ``FlashAttention.forward`` /
        ``MLAAttention.forward``), so the page-table / slot indices below are
        only used to build ``input_data`` and never dereference a real cache.
        """
        total_tokens = max(1, int(total_tokens))
        # Cap each dummy sequence at the context window so RoPE positions stay
        # valid; tile as many as needed (peak activation depends on the *total*
        # token count, not how it is split across sequences).
        per_seq = max(1, min(total_tokens, self.model_max_length))
        seqs = []
        next_page = 0
        remaining = total_tokens
        idx = 0
        while remaining > 0:
            length = min(per_seq, remaining)
            seq = Sequence(idx, [1] * length, [], output_len=1)
            seq.prompt_len = length
            seq.computed_token_num = 0
            seq.to_compute_token_num = length
            num_pages = (length + self.page_size - 1) // self.page_size
            seq.page_table.extend(range(next_page, next_page + num_pages))
            next_page += num_pages
            seqs.append(seq)
            remaining -= length
            idx += 1
        return seqs

    @torch.inference_mode()
    def profile_run(self, stream: Optional[torch.cuda.Stream] = None):
        """Run one dummy forward at the peak batch shape.

        The dummy is a full prefill of ``max_num_batched_tokens`` tokens (see
        :meth:`create_dummy_prefill_seqs`) -- the largest single forward the
        engine can issue -- so the memory profile captures true peak activation
        (including the DP+EP ``dp_gather`` amplification) before the KV cache is
        sized from the remainder.

        Used both for startup memory profiling (``stream=None``, runs on the
        current stream) and as the pre-capture warmup in :meth:`capture_graph`
        (``stream`` set to the capture stream so cuBLAS allocates its per-stream
        workspace there *before* graph capture begins; the run is synchronized
        on return so all that lazy init has completed).
        """
        seqs = self.create_dummy_prefill_seqs(self.max_num_batched_tokens)
        self.input_data.cal_and_set_input(seqs)
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        if self.uses_mrope:
            self.input_data.set_mrope_position(
                torch.zeros((3, num_cal_tokens), device="cpu")
            )
        stream_ctx = (
            torch.cuda.stream(stream) if stream is not None else _nullcontext()
        )
        with stream_ctx:
            if is_first_pp_rank():
                self.model(self.input_data)
            else:
                self.model(
                    self.input_data,
                    self.input_hidden_states[:num_cal_tokens],
                    self.input_residual[:num_cal_tokens],
                )
        # Wait for the dummy forward to finish before returning so all lazy
        # init has completed (required by the capture-stream warmup) and the
        # startup memory profile reflects the real peak.
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()

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
            logger.info(f"Capturing decode full CUDA graphs for bucket sizes: {list(reversed(self.capture_sizes))}")
            iterator = tqdm(self.capture_sizes, desc="Capturing Decode Full Graphs", ncols=100)
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

        # Warm up lazy cuBLAS/Triton init on the capture stream. cuBLAS creates
        # its handle and per-stream workspace on first use; if that happens
        # mid-capture the implicit cudaMalloc is illegal and aborts the capture
        # (cudaErrorStreamCaptureInvalidated). The startup profile_run doesn't
        # survive the intervening memory_manager.init (~30 GB KV/SSM alloc), so
        # re-run it on the capture stream to force + sync that init first.
        self.profile_run(stream=stream)

        # Some FP8 backends (DeepGEMM, and FlashInfer's swapAB for M<32) JIT-
        # compile a distinct kernel per decode M-bucket on first use. That
        # compilation issues an implicit cudaMalloc, which is illegal mid-capture
        # and aborts the graph with cudaErrorStreamCaptureInvalidated. Run one
        # eager forward per bucket (outside the capture context) so every such
        # kernel is compiled *before* we capture it.
        try:
            from gllm.layers.quantization.fp8 import (
                deepgemm_available,
                flashinfer_swapab_available,
            )

            warmup_per_bucket = deepgemm_available() or flashinfer_swapab_available()
        except Exception:  # noqa: BLE001
            warmup_per_bucket = False
        # In DP+EP every group captures each bucket with a uniform global batch
        # (``dp_size * size``): publish ``[size] * dp_size`` so the MoE layer's
        # gather/all-reduce is baked at the right static shape (SGLang MAX_LEN).
        dp_size = get_dp_size() if is_dp_attn() else 1

        def _set_dp_counts(size: int) -> None:
            if is_dp_attn():
                set_dp_forward_counts([size] * dp_size)

        try:
            if warmup_per_bucket:
                for size in self.capture_sizes:
                    seqs = self.create_dummy_seqs(size)
                    self.input_data.cal_and_set_input(seqs=seqs)
                    if self.uses_mrope:
                        self.input_data.set_mrope_position(torch.zeros((3, size), device="cpu"))
                    _set_dp_counts(size)
                    self.forward()
                torch.cuda.synchronize()

            capture_ctx = car.capture() if car is not None else _nullcontext()
            with capture_ctx:
                for size in iterator:
                    seqs = self.create_dummy_seqs(size)
                    self.input_data.cal_and_set_input(seqs=seqs)
                    if self.uses_mrope:
                        self.input_data.set_mrope_position(torch.zeros((3, size), device="cpu"))
                    _set_dp_counts(size)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
                        self.forward()
                    self.size_to_graph[size] = g
                # MTP: capture the draft-step graph per bucket on the same
                # stream/pool (inside the custom-AR capture context) so replay in
                # ``_draft_chain_graph`` agrees on stream + IPC handles.
                if self._mtp_draft_graph:
                    self._capture_draft_graphs(memory_pool, stream)
                # MTP: capture the full verify forward per bucket (same stream/
                # pool/AR-context). This is the dominant MTP cost; replay in
                # ``_mtp_decode`` collapses the ~250ms eager forward to a graph.
                if self._mtp_verify_graph:
                    self._capture_verify_graphs(memory_pool, stream)
        finally:
            if is_dp_attn():
                set_dp_forward_counts(None)
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

    def dp_select_bucket(self, max_tokens: int) -> Optional[int]:
        """Pick the CUDA-graph bucket for a DP decode step, or ``None``.

        In DP+EP the graph bucket must be the *same* on every DP group (the
        global MoE batch is a static ``dp_size * bucket``), so the driver feeds
        the group-wide ``max_tokens`` here. Returns the smallest captured bucket
        ``>= max_tokens``, or ``None`` when graphs are disabled / the batch is
        larger than any captured bucket (caller then runs eager).
        """
        if self.disable_cuda_graph:
            return None
        padded_size = None
        for bucket in self.capture_sizes:
            if bucket >= max_tokens:
                padded_size = bucket
        if padded_size is not None and padded_size in self.size_to_graph:
            return padded_size
        return None

    @staticmethod
    def _build_logprob_rows(seqs, logprobs):
        """Materialize per-batch-row generation logprobs as a Python list.

        ``logprobs`` is the GPU tuple from ``Sampler.compute_logprobs``
        (``sampled`` ``[B]``, ``top_vals`` ``[B, k]``, ``top_ids`` ``[B, k]``).
        Returns a list aligned with ``seqs`` (batch order): ``None`` for seqs
        that did not request logprobs, else ``(sampled, top_ids, top_vals)``
        sliced to that seq's own ``num_top_logprobs``. The scheduler keys into
        this by batch index.
        """
        sampled, top_vals, top_ids = logprobs
        sampled = sampled.cpu().tolist()
        top_vals = top_vals.cpu().tolist()
        top_ids = top_ids.cpu().tolist()
        rows = []
        for i, seq in enumerate(seqs):
            if not seq.logprobs_enabled:
                rows.append(None)
                continue
            k = seq.num_top_logprobs
            rows.append((sampled[i], top_ids[i][:k], top_vals[i][:k]))
        return rows

    def _compute_prompt_logprobs(self, seqs, hidden_states):
        """Accumulate prompt-token logprobs for prefilling seqs (pp_size==1).

        For each seq requesting ``prompt_logprobs`` that is still in prefill,
        run the LM head over this chunk's positions, then record the logprob of
        the *actual* next prompt token at each position (plus top-k). Handles
        chunked prefill by filling only the positions this chunk covers;
        prefix-cache-skipped positions stay ``None``. Needs the full prompt
        ``token_ids`` + ``raw_prompt_len``: works on the real ``Sequence``
        (PP=1) and on the ``FollowerSeq`` mirror (PP>1), which carries those
        fields when ``prompt_logprobs_enabled``.

        Runs on both PP=1 and the PP>1 output-rank follower. Under PP>1 the
        completed lists are stashed in ``self._last_prompt_logprobs`` (keyed on
        the prefill-completing step) for the worker to ship to rank 0 over the
        token socket; under PP=1 the scheduler attaches directly from the seq.

        TP>1: ``logits_from_hidden`` -> ``ParallelLMHead`` issues a
        ``tensor_model_parallel_all_gather``, so this MUST be invoked on *every*
        TP rank of the (last-PP) stage, not just the output rank, or it
        deadlocks. That is safe because every TP rank of the stage holds
        identical seqs (real ``Sequence`` for PP=1, identical ``FollowerSeq``
        mirrors for PP>1), identical ``hidden_states`` and ``query_start_loc``:
        the per-seq ``project`` calls (count + shapes) match bit-for-bit across
        ranks, so the collective is balanced. Each rank computes the same
        result; only the output rank's copy is actually shipped (others drop).
        """
        # Reset each call so stale completions don't re-ship under PP>1.
        self._last_prompt_logprobs = {}
        if not any(getattr(s, "prompt_logprobs_enabled", False) for s in seqs):
            return
        # Models expose ``logits_from_hidden`` to project arbitrary positions to
        # full-vocab logits (LM-head placement stays a model-internal detail).
        # A model lacking it simply doesn't support prompt logprobs (no-op).
        project = getattr(self.model, "logits_from_hidden", None)
        if project is None:
            return
        qsl = self.input_data.query_start_loc_cpu
        for i, seq in enumerate(seqs):
            if not getattr(seq, "prompt_logprobs_enabled", False):
                continue
            if seq.computed_prompt:
                continue
            c0 = seq.computed_token_num
            prompt_len = seq.raw_prompt_len
            start = int(qsl[i])
            n = int(qsl[i + 1]) - start
            # positions p=c0+j predict prompt token p+1; only p+1 <= prompt_len-1
            # is a prompt token (the last position predicts the first generated
            # token, handled by the generation-logprobs path).
            jmax = min(n, prompt_len - 1 - c0)
            if jmax <= 0:
                continue
            logits = project(hidden_states[start : start + jmax])
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            target_ids = seq.token_ids[c0 + 1 : c0 + 1 + jmax]
            target = torch.tensor(
                target_ids, device=logprobs.device, dtype=torch.long
            ).view(-1, 1)
            sampled = logprobs.gather(1, target).squeeze(1).cpu().tolist()
            k = min(seq.num_prompt_logprobs, logprobs.shape[-1])
            if k > 0:
                top_vals, top_ids = torch.topk(logprobs, k, dim=-1)
                top_vals = top_vals.cpu().tolist()
                top_ids = top_ids.cpu().tolist()
            else:
                top_vals = [[] for _ in range(jmax)]
                top_ids = [[] for _ in range(jmax)]
            if seq.prompt_logprobs_data is None:
                seq.prompt_logprobs_data = [None] * prompt_len
            for j in range(jmax):
                pos = c0 + 1 + j
                seq.prompt_logprobs_data[pos] = (
                    target_ids[j],
                    sampled[j],
                    top_ids[j],
                    top_vals[j],
                )
            # The prompt finishes prefill this step once the chunk reaches the
            # end of the prompt (c0 + n >= prompt_len). At that point every
            # position 1..prompt_len-1 is filled, so the list is complete --
            # record it for the PP>1 socket path (harmless/ignored under PP=1).
            if c0 + n >= prompt_len:
                self._last_prompt_logprobs[seq.seq_id] = seq.prompt_logprobs_data

    def _record_mtp_metrics(self, nd: int, k: int, n_accepted: list) -> None:
        """Accumulate MTP acceptance stats and periodically log them (TP0 only).

        Metric definitions (so the numbers are comparable to common
        spec-decode reporting):

        * **draft acceptance rate** = accepted_draft_tokens / drafted_tokens.
          ``drafted_tokens = num_drafts * k`` (k proposals per draft/step).
        * **mean acceptance length** = 1 + accepted/num_drafts (the ``1`` is the
          always-committed target token x1, i.e. the "bonus"; so a value of
          ``k+1`` means every draft accepted).
        * **per-position acceptance rate**[i] = fraction of drafts whose accepted
          length reached position ``i`` (i in ``0..k-1``), i.e.
          ``count(n_accepted > i) / num_drafts`` -- the standard per-position
          histogram.

        ``num_drafts`` counts one draft *per sequence per step* (nd per call).
        Logging is time-based on a 1s window to match gLLM's scheduler status
        log period (``scheduler.py`` ``time.time() - log_time > 1``), and only on
        TP rank 0 to avoid N duplicate lines. Reset the window after each log.
        """
        import time as _time

        if not hasattr(self, "_mtp_m_drafts"):
            self._mtp_m_drafts = 0            # number of (seq,step) drafts
            self._mtp_m_accepted = 0          # total accepted draft tokens
            self._mtp_m_pos = [0] * k         # per-position accept counts
            self._mtp_m_t0 = _time.time()
        self._mtp_m_drafts += nd
        for na in n_accepted:
            self._mtp_m_accepted += na
            for i in range(na):
                self._mtp_m_pos[i] += 1

        if get_tp_rank() != 0:
            return
        now = _time.time()
        if now - self._mtp_m_t0 < 1.0:
            return
        drafts = self._mtp_m_drafts
        acc = self._mtp_m_accepted
        drafted_tokens = drafts * k
        rate = 100.0 * acc / drafted_tokens if drafted_tokens else 0.0
        mean_len = 1.0 + acc / drafts if drafts else 1.0
        per_pos = ", ".join(
            f"{(self._mtp_m_pos[i] / drafts if drafts else 0.0):.3f}" for i in range(k)
        )
        logger.info(
            "MTP metrics: Mean acceptance length: %.2f, Accepted: %d tokens, "
            "Drafted: %d tokens, Per-position acceptance rate: %s, "
            "Draft acceptance rate: %.1f%%",
            mean_len, acc, drafted_tokens, per_pos, rate,
        )
        # Sparse top-k window overflow (see ``_mtp_sparse_probs``). Read at most
        # once per log window so the hot path never syncs; a nonzero count means
        # probability ties spilled past ``top_k + _SPARSE_TIE_MARGIN`` and a few
        # tied tokens were dropped from p's support (tiny distribution skew, not a
        # correctness break) -- raise the margin if it ever shows up.
        n_of = int(self._mtp_tie_overflow)
        if n_of:
            logger.warning(
                "MTP sparse top-k tie overflow on %d rows (margin=%d); "
                "consider raising ModelRunner._SPARSE_TIE_MARGIN",
                n_of, self._SPARSE_TIE_MARGIN,
            )
            self._mtp_tie_overflow.zero_()
        # reset window
        self._mtp_m_drafts = 0
        self._mtp_m_accepted = 0
        self._mtp_m_pos = [0] * k
        self._mtp_m_t0 = now

    # ------------------------------------------------------------------
    # MTP rejection sampling (lossless speculative decoding under sampling)
    # ------------------------------------------------------------------
    def _mtp_rng_step(self, device):
        """Return a TP-synchronized ``torch.Generator`` seeded for this step.

        Every column driver runs the same deterministic schedule, so seeding
        from a per-runner step counter makes the seed identical on every TP
        rank -> identical draws -> TP-consistent committed tokens (the sampling
        analog of the greedy-argmax determinism the MTP sync path relies on).
        """
        if self._mtp_rng is None:
            self._mtp_rng = torch.Generator(device=device)
        # A fixed base keeps runs reproducible; the counter advances in lockstep
        # across ranks. 0x9E3779B9 (golden-ratio) spreads consecutive seeds.
        self._mtp_rng.manual_seed(0x9E3779B9 * (self._mtp_step + 1) & 0x7FFFFFFFFFFFFFFF)
        self._mtp_step += 1
        return self._mtp_rng

    def _mtp_bcast_tp(self, tok: torch.Tensor) -> torch.Tensor:
        """Broadcast an int64 token tensor from TP-rank-0 across the TP group.

        Rejection sampling draws random tokens, so unlike greedy argmax it is NOT
        deterministic across TP ranks (per-rank logits differ by fp all-reduce
        epsilon, and multinomial amplifies that into different token picks). To
        keep every column driver's committed tokens + draft-forward inputs
        identical (the invariant the MTP sync path relies on), TP-rank-0's draws
        win: sample only there, broadcast to peers. No-op for tp_size==1.

        Uses the SAME TP communicator (``get_tp_group``) as the model's
        all-reduces and ``run_batch_async``'s token broadcast, NOT the IPC group:
        NCCL's per-communicator FIFO ordering then implicitly serializes this
        broadcast against the surrounding forward all-reduces on every rank,
        which is exactly what prevents the cross-communicator ordering hazard
        that otherwise deadlocks (broadcast on one communicator racing the
        forward all-reduce on another when ranks reach them in different orders).
        """
        if get_tp_size() <= 1:
            return tok
        dist.broadcast(tok, src=get_rank() - get_tp_rank(), group=get_tp_group())
        return tok

    def _mtp_probs_from_logits(self, logits, seqs):
        """Apply the per-seq sampling transform (temp -> softmax -> top-k/top-p
        renorm) to ``logits`` [n, vocab], returning a proper prob distribution
        [n, vocab]. Mirrors ``Sampler.forward_gpu`` so the MTP draft dist ``q``
        and target dist ``p`` live on the SAME transformed space, which is what
        rejection sampling requires. ``seqs`` is aligned row-for-row with logits.
        """
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        dev = logits.device
        temps = torch.tensor(
            [s.temperature if s.temperature > 1e-5 else 1.0 for s in seqs],
            device=dev, dtype=torch.float32,
        ).unsqueeze(1)
        probs = torch.softmax(logits.float() / temps, dim=-1)
        top_ks = torch.tensor(
            [s.top_k if s.top_k != -1 else self.memory_manager.vocab_size for s in seqs],
            device=dev, dtype=torch.int32,
        )
        top_ps = torch.tensor(
            [s.top_p for s in seqs], device=dev, dtype=torch.float32
        )
        # Only renorm when some seq actually restricts (cheap guard).
        if int(top_ks.min().item()) < self.memory_manager.vocab_size:
            probs = top_k_renorm_prob(probs, top_ks)
        if float(top_ps.min().item()) < 1.0:
            probs = top_p_renorm_prob(probs, top_ps)
        return probs

    def _mtp_sample_params(self, seqs, dev):
        """Per-seq ``(temps[n,1], top_ks[n], top_ps[n])`` on the device.

        Staged through **persistent pinned** buffers with ``non_blocking`` copies.
        The obvious ``torch.tensor(list, device="cuda")`` form copies from
        *pageable* memory, which torch has to serialize with a
        ``cudaStreamSynchronize`` -- and since this runs right after the verify
        graph was enqueued, that sync blocks the host on the whole outstanding
        GPU queue. The torch profiler put 558 ms of a 2 s sampling window in
        exactly these three lines (3 syncs per MTP step, ~28% of the window),
        purely as a serialization bubble in the middle of the step.
        """
        V = self.memory_manager.vocab_size
        n = len(seqs)
        if self._sp_host_f is None:
            b = max(self.max_running_seqs, 1)
            self._sp_host_f = torch.empty(
                (2, b), dtype=torch.float32, device="cpu", pin_memory=True
            )
            self._sp_host_k = torch.empty(
                b, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self._sp_dev_f = torch.empty((2, b), dtype=torch.float32, device=dev)
            self._sp_dev_k = torch.empty(b, dtype=torch.int32, device=dev)
        hf, hk = self._sp_host_f.numpy(), self._sp_host_k.numpy()
        hf[0, :n] = [s.temperature if s.temperature > 1e-5 else 1.0 for s in seqs]
        hf[1, :n] = [s.top_p for s in seqs]
        hk[:n] = [s.top_k if s.top_k != -1 else V for s in seqs]
        self._sp_dev_f[:, :n].copy_(self._sp_host_f[:, :n], non_blocking=True)
        self._sp_dev_k[:n].copy_(self._sp_host_k[:n], non_blocking=True)
        return (
            self._sp_dev_f[0, :n].unsqueeze(1),
            self._sp_dev_k[:n],
            self._sp_dev_f[1, :n],
        )

    def _mtp_probs_static(self, logits, temps, top_ks, top_ps):
        """Graph-safe variant of ``_mtp_probs_from_logits``: all inputs are
        static GPU tensors (no python seq list, no ``.item()`` guards) so the
        whole thing is CUDA-graph capturable. Always applies top-k/top-p renorm
        kernels unconditionally (a seq that doesn't restrict passes top_k=vocab /
        top_p=1, which the kernels treat as no-ops). ``temps`` [n,1], ``top_ks``
        [n] int32, ``top_ps`` [n] float32.
        """
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        probs = torch.softmax(logits.float() / temps, dim=-1)
        probs = top_k_renorm_prob(probs, top_ks)
        probs = top_p_renorm_prob(probs, top_ps)
        return probs

    # Headroom over the largest per-request ``top_k`` when building the sparse
    # (top-k) distribution: the reference kernel is TIE-INCLUSIVE (it keeps every
    # token whose prob equals the k-th largest), and bf16 logits over a 248k
    # vocab tie often enough to matter -- measured support for ``top_k=20`` was
    # 20..24. ``torch.topk``'s cost is dominated by the vocab scan, so a fat
    # margin is nearly free (k=20 and k=64 both measured 0.16 ms at 64 rows).
    _SPARSE_TIE_MARGIN = 64
    # ``k_pad`` baked into the captured sparse sampled-draft graph. A batch whose
    # largest ``top_k`` needs a wider window falls back to the dense captured
    # graph (still correct, just slower), so this only has to cover the common
    # serving range (``top_k`` up to 64 with the tie margin on top).
    _sparse_kpad_capture = 128

    def _mtp_kpad(self, seqs) -> int:
        """Sparse top-k window width for this batch, computed on the HOST.

        The per-request ``top_k`` values are plain python ints, so taking the max
        here costs nothing -- doing it as ``int(top_ks.max().item())`` on the
        staged device tensor (as the first version did) inserted a
        ``cudaStreamSynchronize`` into every MTP step, which the torch profiler
        duly showed sitting in the critical path.
        """
        mx = 1
        for s in seqs:
            tk = s.top_k
            if tk is not None and tk > mx:
                mx = tk
        return min(mx + self._SPARSE_TIE_MARGIN, self.memory_manager.vocab_size)

    def _mtp_sparse_eligible(self, seqs) -> bool:
        """True when every seq's ``top_k`` fits the captured sparse window.

        The sparse path represents ``q``/``p`` by their top-k support, which is
        only exact when ``top_k`` is restricted -- an unrestricted request
        (``top_k == -1`` / vocab, i.e. top-p only) keeps the dense path.
        """
        cap = self._sparse_kpad_capture - self._SPARSE_TIE_MARGIN
        for s in seqs:
            tk = s.top_k
            if tk is None or tk <= 0 or tk > cap:
                return False
        return True

    def _mtp_sparse_probs(self, logits, temps, top_ks, top_ps, k_pad):
        """Top-k-sparse form of :meth:`_mtp_probs_static`.

        Returns ``(vals, idx)`` -- ``[n, k_pad]`` probabilities (descending, zero
        outside the kept support) and their token ids. Mathematically identical to
        the dense ``softmax -> top_k_renorm -> top_p_renorm`` chain (verified to
        1e-7 on non-tied logits), because
        ``softmax`` restricted to the kept set == the dense renormalization of
        that set, and ``keep`` is a prefix of the descending order.

        Dense costs one full-vocab softmax plus two renorm passes over
        ``[n, vocab]`` (1.5 ms at n=64, 3.1 ms at n=256 for this vocab); this is a
        single ``topk`` plus ``[n, k_pad]`` arithmetic (0.2 / 0.6 ms).

        Ties beyond ``k_pad`` would silently drop tokens the dense kernel keeps,
        so the (sync-free) overflow counter is accumulated on-device and reported
        by ``_record_mtp_metrics``.

        ``topk`` runs on the raw logits and only the selected ``[n, k_pad]`` slice
        is widened to fp32: ``topk`` costs the same for any ``k_pad`` in this range
        but scales with the bytes it scans, so selecting on bf16 halves it (0.31 ->
        0.16 ms at n=64, 1.01 -> 0.55 ms at n=256). Bit-identical to selecting on
        the widened logits -- the cast is lossless and order preserving.
        """
        vals, idx = torch.topk(logits, k_pad, dim=-1)            # descending
        vals = vals.float()
        # Tie-inclusive top-k: keep everything >= the top_k-th largest value.
        kth = vals.gather(
            1, (top_ks.long() - 1).clamp_(0, k_pad - 1).unsqueeze(1)
        )
        keep = vals >= kth
        probs = torch.softmax(
            vals.masked_fill(~keep, float("-inf")) / temps, dim=-1
        )
        # top-p over the descending probs: keep the shortest prefix that reaches
        # ``top_p`` (exclusive cumsum < top_p), then renormalize.
        csum = probs.cumsum(dim=-1)
        probs = torch.where(
            (csum - probs) < top_ps.unsqueeze(1), probs, torch.zeros_like(probs)
        )
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # Ties spilling past the window: ``keep`` reaching the last column means
        # more equal-valued tokens may exist beyond it.
        self._mtp_tie_overflow += keep[:, -1].sum()
        return probs, idx

    @staticmethod
    def _q_dense(dense):
        """Draft-distribution handle, dense form: ``dense`` is ``[nd, k, vocab]``."""
        return MtpQDist(dense=dense, vals=None, idx=None, drawn=None)

    @staticmethod
    def _q_sparse(vals, idx, drawn):
        """Draft-distribution handle, sparse form.

        ``vals``/``idx``: ``[nd, k, k_pad]`` top-k support of each step's ``q``;
        ``drawn``: ``[nd, k]`` probability of the token that step actually drew.
        """
        return MtpQDist(dense=None, vals=vals, idx=idx, drawn=drawn)

    def _gumbel_argmax(self, q):
        """Draw from ``q`` [n, vocab] via the Gumbel-max trick:
        ``argmax(q / Exp(1))``. Distributionally equal to ``torch.multinomial``
        but graph-safe -- the only randomness is ``exponential_`` (a capturable
        RNG kernel whose Philox offset advances correctly across graph replays),
        and ``multinomial``'s device-side distribution-validity assert (which a
        graph would capture + replay every step) is avoided. Uses the DEFAULT
        CUDA generator (the only one capturable inside ``torch.cuda.graph``).
        """
        noise = torch.empty_like(q, dtype=torch.float32).exponential_(1.0)
        noise.clamp_min_(torch.finfo(torch.float32).tiny)
        return (q.float() / noise).argmax(dim=-1).to(torch.int64)

    @torch.inference_mode()
    def _draft_chain_eager_sampled(
        self, decode_seqs, orig_tokens, x1, hidden, k, nd, gen, sparse=False
    ):
        """Eager draft chain that SAMPLES each draft token (for rejection mode).

        Same forward structure as ``_draft_chain_eager`` but instead of argmax it
        draws each draft token from the per-seq transformed distribution ``q`` and
        records that ``q`` so the accept step can compute ``min(1, p/q)`` and the
        residual ``(p-q)+``. Returns ``(drafts, q)`` where ``drafts`` is per-seq
        ``[d1..dk]`` (CPU ints) and ``q`` is an :class:`MtpQDist` -- sparse
        (top-k support) when ``sparse``, dense ``[nd, k, vocab]`` otherwise.
        """
        mtp = self.model.mtp
        dev = hidden.device
        drafts_cols = [[] for _ in range(nd)]
        tok = torch.tensor(x1, device=dev, dtype=torch.int64)
        cur_hidden = hidden
        q_steps, qv_steps, qi_steps, qd_steps = [], [], [], []
        if sparse:
            temps, top_ks, top_ps = self._mtp_sample_params(decode_seqs, dev)
            k_pad = self._mtp_kpad(decode_seqs)
        for j in range(k):
            for i, s in enumerate(decode_seqs):
                s.token_ids = orig_tokens[i] + [x1[i]] + drafts_cols[i]
                s.computed_token_num = len(s.token_ids) - 1
                s.to_compute_token_num = 1
            self.memory_manager.pre_allocate_page(decode_seqs, cacheable=False)
            self.prepare_input(decode_seqs)
            out_hidden = mtp.forward(self.input_data, cur_hidden, tok)
            logits = mtp.logits_from_hidden(out_hidden)
            # Sample one draft token per seq from q (TP-synced generator), then
            # broadcast TP-rank-0's picks so every rank feeds the SAME token into
            # the next draft forward (multinomial isn't TP-deterministic).
            if sparse:
                qv, qi = self._mtp_sparse_probs(logits, temps, top_ks, top_ps, k_pad)
                col = torch.multinomial(qv, num_samples=1, generator=gen)
                tok = qi.gather(1, col).squeeze(1).to(torch.int64)
                tok = self._mtp_bcast_tp(tok)
                # After the TP broadcast the drawn token may come from rank 0, so
                # look its probability up by id rather than by column.
                qd = (qv * (qi == tok.unsqueeze(1)).to(qv.dtype)).sum(dim=1)
                qv_steps.append(qv)
                qi_steps.append(qi)
                qd_steps.append(qd)
            else:
                q = self._mtp_probs_from_logits(logits, decode_seqs)  # [nd, vocab]
                tok = (
                    torch.multinomial(q, num_samples=1, generator=gen)
                    .squeeze(1).to(torch.int64)
                )
                tok = self._mtp_bcast_tp(tok)
                q_steps.append(q)
            tok_cpu = tok.tolist()
            for i in range(nd):
                drafts_cols[i].append(tok_cpu[i])
            cur_hidden = out_hidden
        if sparse:
            return drafts_cols, self._q_sparse(
                torch.stack(qv_steps, dim=1),
                torch.stack(qi_steps, dim=1),
                torch.stack(qd_steps, dim=1),
            )
        return drafts_cols, self._q_dense(torch.stack(q_steps, dim=1))

    @torch.inference_mode()
    def _draft_chain_eager(self, decode_seqs, orig_tokens, x1, hidden, k, nd):
        """Eager k-step MTP draft chain (fallback / graph-disabled path).

        One D2H at the end (a ``[nd, k]`` tensor -> list) instead of a ``.item()``
        per token per step. Returns ``drafts`` = per-seq ``[d1..dk]`` (CPU ints).
        """
        mtp = self.model.mtp
        dev = hidden.device
        drafts_cols = [[] for _ in range(nd)]   # per-seq python (for token_ids mutation)
        tok = torch.tensor(x1, device=dev, dtype=torch.int64)
        cur_hidden = hidden
        step_toks = []  # list of [nd] GPU tensors
        for j in range(k):
            cols = [drafts_cols[i] for i in range(nd)]
            for i, s in enumerate(decode_seqs):
                s.token_ids = orig_tokens[i] + [x1[i]] + cols[i]
                s.computed_token_num = len(s.token_ids) - 1
                s.to_compute_token_num = 1
            self.memory_manager.pre_allocate_page(decode_seqs, cacheable=False)
            self.prepare_input(decode_seqs)
            out_hidden = mtp.forward(self.input_data, cur_hidden, tok)
            tok = mtp.logits_from_hidden(out_hidden).argmax(dim=-1).to(torch.int64)
            step_toks.append(tok)
            # Need python token_ids for the next step's slot bookkeeping, so this
            # step's tokens must be materialized before building step j+1. Keep a
            # single per-step D2H (unavoidable in the eager path since positions/
            # slots are rebuilt from python token_ids each step).
            tok_cpu = tok.tolist()
            for i in range(nd):
                drafts_cols[i].append(tok_cpu[i])
            cur_hidden = out_hidden
        return drafts_cols

    @torch.inference_mode()
    def _draft_chain_graph(self, decode_seqs, orig_tokens, x1, hidden, k, nd):
        """CUDA-graph k-step MTP draft chain.

        Captures ONE draft-step graph per exact batch size ``nd`` (lazily) and
        replays it k times. Between replays, tok/hidden/positions/slot_mapping/
        seq_lens are advanced **in place on the GPU** (no Python/H2D/.item()),
        so the whole chain has zero per-step host overhead and a single D2H at
        the end. The captured graph runs ``mtp.forward`` over ``self.input_data``
        with the MTP head's ``prev_hidden``/``input_ids`` aliased to static
        buffers ``self._d_hidden``/``self._d_tok``; its argmax is written to
        ``self._d_next_tok`` and post-block hidden to ``self._d_out_hidden``.

        Requires all seqs share one page-table width and per-step positions stay
        within the pre-allocated draft slots. Falls back is handled by the caller
        (this is only entered when ``nd <= max bucket``).
        """
        dev = hidden.device
        page_sz = self.memory_manager.page_size

        # Smallest captured bucket >= nd (sorted() ascending, take the first
        # match). Fall back to eager if this batch size wasn't captured at init.
        bucket = None
        for b in sorted(self._draft_size_to_graph.keys()):
            if b >= nd:
                bucket = b
                break
        if bucket is None:
            return self._draft_chain_eager(decode_seqs, orig_tokens, x1, hidden, k, nd)
        g = self._draft_size_to_graph[bucket]

        # KV pages for the whole speculative window were pre-allocated once by
        # ``_mtp_decode`` (draft writes ctx..ctx+k-1, verify ctx..ctx+k), so the
        # page tables are already frozen for this step -- nothing to do here.

        # Fill the static draft-input buffers IN PLACE for this step (the captured
        # graph reads these exact buffers). Padded rows [nd:bucket] are written
        # as dummy rows by the GPU prep (no throwaway ``Sequence`` objects); the
        # CPU fallback still builds dummy decode seqs.
        gp = self._mtp_gpu_prep_batch(decode_seqs, orig_tokens, x1, bucket)
        if gp is not None:
            gp.fill_draft(self._draft_input)
        else:
            # CPU fallback: the builders read the seq state, so put it in the
            # draft shape first (one new token at ``ctx``). ``_mtp_decode`` left
            # the seqs in the *verify* shape (1+k speculative tokens) for the
            # one-shot page allocation.
            for i, s in enumerate(decode_seqs):
                s.token_ids = orig_tokens[i] + [x1[i]]
                s.computed_token_num = len(s.token_ids) - 1
                s.to_compute_token_num = 1
            pad_seqs = (
                self.create_dummy_seqs(bucket - nd, runtime=True) if bucket > nd else []
            )
            graph_seqs = list(decode_seqs) + pad_seqs
            self._draft_input.cal_and_set_input(graph_seqs)
        self._d_nd = bucket
        if gp is not None:
            # x1 already sits in the staged metadata on the device -> D2D copy
            # instead of another pageable H2D of the host ``x1`` list.
            self._d_tok[:nd].copy_(gp.x1_gpu(nd))
        else:
            self._d_tok[:nd].copy_(torch.tensor(x1, device=dev, dtype=torch.int64))
        if bucket > nd:
            self._d_tok[nd:bucket].zero_()
        self._d_hidden[:nd].copy_(hidden)
        if bucket > nd:
            self._d_hidden[nd:bucket].zero_()

        base_pos = self._draft_input.positions[:bucket].clone()
        block_table = self._draft_input.block_table[:bucket]

        def _slot_for(pos):
            blk = block_table[torch.arange(bucket, device=dev), (pos // page_sz)]
            return (blk.to(torch.int64) * page_sz + (pos % page_sz))

        step_next = []
        for j in range(k):
            if j > 0:
                self._d_tok[:bucket].copy_(self._d_next_tok[:bucket])
                self._d_hidden[:bucket].copy_(self._d_out_hidden[:bucket])
                new_pos = base_pos + j
                self._draft_input.positions[:bucket].copy_(new_pos)
                self._draft_input.slot_mapping[:bucket].copy_(_slot_for(new_pos))
                # ``decode_seq_lens`` is MLA-only metadata (set in
                # ``_cal_mla_metadata``); non-MLA models advance only
                # ``seq_lens``, which the GDN/full-attn decode kernels read.
                if self.use_mla:
                    self._draft_input.decode_seq_lens[:bucket].add_(1)
                self._draft_input.seq_lens[:bucket].add_(1)
            g.replay()
            step_next.append(self._d_next_tok[:nd].clone())

        drafts_gpu = torch.stack(step_next, dim=1)  # [nd, k] on GPU
        # Stash the GPU draft tensor and return ``None`` for the host copy:
        # ``.tolist()`` here is a blocking D2H that stalls the host on the whole
        # draft chain (~0.55 ms at nd=64) only to hand the accept loop token ids
        # it now gets from the single end-of-step packed D2H. Paths that truly
        # need host-side drafts go through :meth:`_drafts_host`.
        self._drafts_gpu = drafts_gpu
        return None

    @torch.inference_mode()
    def _draft_chain_graph_sampled(
        self, decode_seqs, orig_tokens, x1, hidden, k, nd, sparse=False
    ):
        """CUDA-graph k-step SAMPLED (rejection) MTP draft chain.

        Mirrors :meth:`_draft_chain_graph` but replays the sampled draft step
        (Gumbel-max draw + q-dist stash). Between replays the drawn token is
        broadcast across TP (host side, OUTSIDE the graph) so every rank feeds
        the same token into the next step -- the Gumbel-max RNG (default CUDA
        generator, captured) is not guaranteed identical across ranks, so we
        sync the token explicitly rather than rely on RNG lockstep. Returns
        ``(drafts, q)`` -- ``drafts`` per-seq ``[d1..dk]`` (CPU ints) and ``q`` an
        :class:`MtpQDist` (sparse top-k support when ``sparse``, else dense
        ``[nd, k, vocab]``). Falls back to eager if the bucket wasn't captured.
        """
        dev = hidden.device
        page_sz = self.memory_manager.page_size
        # ``sparse``: every request restricts top_k, so the batch can use the
        # captured top-k-sparse draft step (one topk instead of a full-vocab
        # softmax + two renorm passes per step).
        graphs = (
            self._draft_size_to_graph_sampled_sparse
            if sparse
            else self._draft_size_to_graph_sampled
        )
        bucket = None
        for b in sorted(graphs.keys()):
            if b >= nd:
                bucket = b
                break
        if bucket is None:
            gen = self._mtp_rng_step(dev)
            return self._draft_chain_eager_sampled(
                decode_seqs, orig_tokens, x1, hidden, k, nd, gen, sparse=sparse
            )
        g = graphs[bucket]

        # KV pages for the whole speculative window were pre-allocated once by
        # ``_mtp_decode``; fill the static draft buffers the same way the greedy
        # graph chain does (GPU-native prep, CPU builders as the fallback).
        gp = self._mtp_gpu_prep_batch(decode_seqs, orig_tokens, x1, bucket)
        if gp is not None:
            gp.fill_draft(self._draft_input)
        else:
            for i, s in enumerate(decode_seqs):
                s.token_ids = orig_tokens[i] + [x1[i]]
                s.computed_token_num = len(s.token_ids) - 1
                s.to_compute_token_num = 1
            pad_seqs = (
                self.create_dummy_seqs(bucket - nd, runtime=True) if bucket > nd else []
            )
            graph_seqs = list(decode_seqs) + pad_seqs
            self._draft_input.cal_and_set_input(graph_seqs)
        self._d_nd = bucket
        if gp is not None:
            self._d_tok[:nd].copy_(gp.x1_gpu(nd))
        else:
            self._d_tok[:nd].copy_(torch.tensor(x1, device=dev, dtype=torch.int64))
        if bucket > nd:
            self._d_tok[nd:bucket].zero_()
        self._d_hidden[:nd].copy_(hidden)
        if bucket > nd:
            self._d_hidden[nd:bucket].zero_()
        # Fill per-seq sampling params into the static buffers the graph reads.
        # Via the pinned staging (D2D copies here) -- the previous
        # ``torch.tensor(list, device=cuda)`` form was three pageable H2Ds, i.e.
        # three implicit stream syncs per draft chain.
        _temps, _top_ks, _top_ps = self._mtp_sample_params(decode_seqs, dev)
        self._d_temp[:nd, 0].copy_(_temps.squeeze(1))
        self._d_topk[:nd].copy_(_top_ks)
        self._d_topp[:nd].copy_(_top_ps)
        V = self.memory_manager.vocab_size
        if bucket > nd:  # padded rows: harmless greedy-ish params
            self._d_temp[nd:bucket].fill_(1.0)
            # ``top_k = 1`` (not ``vocab``): on the sparse path an unrestricted
            # ``top_k`` clamps the tie threshold to the last column of the window,
            # which always trips the tie-overflow counter. Padded rows' output is
            # discarded, so pick the value that keeps the diagnostic meaningful.
            self._d_topk[nd:bucket].fill_(1)
            self._d_topp[nd:bucket].fill_(1.0)

        base_pos = self._draft_input.positions[:bucket].clone()
        block_table = self._draft_input.block_table[:bucket]

        def _slot_for(pos):
            blk = block_table[torch.arange(bucket, device=dev), (pos // page_sz)]
            return (blk.to(torch.int64) * page_sz + (pos % page_sz))

        step_next = []
        step_q, step_qv, step_qi, step_qd = [], [], [], []
        for j in range(k):
            if j > 0:
                self._d_tok[:bucket].copy_(self._d_next_tok[:bucket])
                self._d_hidden[:bucket].copy_(self._d_out_hidden[:bucket])
                new_pos = base_pos + j
                self._draft_input.positions[:bucket].copy_(new_pos)
                self._draft_input.slot_mapping[:bucket].copy_(_slot_for(new_pos))
                # ``decode_seq_lens`` is MLA-only metadata (set in
                # ``_cal_mla_metadata``); non-MLA models advance only
                # ``seq_lens``, which the GDN/full-attn decode kernels read.
                if self.use_mla:
                    self._draft_input.decode_seq_lens[:bucket].add_(1)
                self._draft_input.seq_lens[:bucket].add_(1)
            g.replay()
            # TP-sync the drawn token (Gumbel RNG isn't guaranteed identical
            # across ranks); broadcast BEFORE it seeds the next step's forward.
            self._mtp_bcast_tp(self._d_next_tok[:bucket])
            step_next.append(self._d_next_tok[:nd].clone())
            if sparse:
                step_qv.append(self._d_qv[:nd].clone())
                step_qi.append(self._d_qi[:nd].clone())
                # The broadcast above can replace this rank's drawn token with
                # rank 0's, so re-derive the drawn probability by token id rather
                # than trusting the in-graph ``_d_qd`` column lookup.
                step_qd.append(
                    (
                        self._d_qv[:nd]
                        * (self._d_qi[:nd] == self._d_next_tok[:nd].unsqueeze(1))
                    ).sum(dim=1)
                )
            else:
                step_q.append(self._d_q[:nd].clone())

        drafts_gpu = torch.stack(step_next, dim=1)     # [nd, k] on GPU
        if sparse:
            q = self._q_sparse(
                torch.stack(step_qv, dim=1),           # [nd, k, k_pad]
                torch.stack(step_qi, dim=1),
                torch.stack(step_qd, dim=1),           # [nd, k]
            )
        else:
            q = self._q_dense(torch.stack(step_q, dim=1))   # [nd, k, vocab]
        # Stash the GPU copy so the verify prep can take the tokens straight from
        # the device. The rejection accept still walks the drafts host-side (it
        # slices the committed prefix per seq), so materialize them here -- one
        # D2H, same as before.
        self._drafts_gpu = drafts_gpu
        mat = drafts_gpu.tolist()
        return [mat[i] for i in range(nd)], q

    @torch.inference_mode()
    def _ensure_draft_buffers(self):
        """Allocate the static MTP-draft-step buffers + aliasing InputData once."""
        if self._draft_input is not None:
            return
        dev = torch.cuda.current_device()
        B = max(self.capture_sizes)
        H = self.hidden_size
        dt = self.output_hidden_states.dtype
        self._d_tok = torch.zeros(B, dtype=torch.int64, device=dev)
        self._d_hidden = torch.zeros((B, H), dtype=dt, device=dev)
        self._d_next_tok = torch.zeros(B, dtype=torch.int64, device=dev)
        self._d_out_hidden = torch.zeros((B, H), dtype=dt, device=dev)
        # Sampled-draft (rejection sampling) static buffers: per-seq sampling
        # params + the drawn q distribution the accept step reads. Vocab-wide q
        # is the only large one (B*vocab); allocated once, reused across replays.
        if self._mtp_can_sample:
            V = self.memory_manager.vocab_size
            self._d_temp = torch.ones((B, 1), dtype=torch.float32, device=dev)
            self._d_topk = torch.full((B,), V, dtype=torch.int32, device=dev)
            self._d_topp = torch.ones((B,), dtype=torch.float32, device=dev)
            self._d_q = torch.zeros((B, V), dtype=torch.float32, device=dev)
            # Sparse (top-k) draft distribution: ``[B, k_pad]`` values + token ids
            # + the probability of the token actually drawn. Replaces the
            # ``[B, vocab]`` dense ``q`` for top_k-restricted batches; see
            # ``_draft_step_forward_sampled_sparse``. 1.5 MB vs 63 MB.
            kp = self._sparse_kpad_capture
            self._d_qv = torch.zeros((B, kp), dtype=torch.float32, device=dev)
            self._d_qi = torch.zeros((B, kp), dtype=torch.int64, device=dev)
            self._d_qd = torch.zeros(B, dtype=torch.float32, device=dev)
        self._draft_input = InputData(
            max_running_seqs=self.max_running_seqs,
            max_seq_length=self.model_max_length,
            memory_manager=self.memory_manager,
            use_buffer=True,
        )

    @torch.inference_mode()
    def _capture_draft_graphs(self, memory_pool, stream):
        """Capture one MTP draft-step graph per decode bucket (init-time).

        Mirrors the decode-graph capture: per bucket, set up the draft-input with
        dummy decode seqs (KV -> dummy page), seed the static ``_d_*`` buffers,
        warm up eager once (DeepGEMM/FlashInfer JIT), then capture
        ``_draft_step_forward`` on the shared capture ``stream``/``pool``. Replay
        (``_draft_chain_graph``) later updates the same buffers in place.
        """
        self._ensure_draft_buffers()
        iterator = self.capture_sizes
        if get_local_rank() == 0:
            logger.info(
                f"Capturing MTP draft CUDA graphs for bucket sizes: "
                f"{list(reversed(self.capture_sizes))}"
            )
            iterator = tqdm(
                self.capture_sizes, desc="Capturing MTP Draft Graphs", ncols=100
            )
        for bucket in iterator:
            seqs = self.create_dummy_seqs(bucket)
            self._draft_input.cal_and_set_input(seqs)
            self._d_nd = bucket
            # seed dummy head inputs
            self._d_tok[:bucket].zero_()
            self._d_hidden[:bucket].zero_()
            # warm up JIT outside capture
            self._draft_step_forward()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
                self._draft_step_forward()
            self._draft_size_to_graph[bucket] = g
            # Also capture the sampled (rejection) draft step when enabled, so
            # sampling requests get a graphed draft too (Gumbel-max, graph-safe).
            if self._mtp_can_sample:
                self._d_temp[:bucket].fill_(1.0)
                self._d_topk[:bucket].fill_(self.memory_manager.vocab_size)
                self._d_topp[:bucket].fill_(1.0)
                self._draft_step_forward_sampled()
                torch.cuda.synchronize()
                gs = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph=gs, pool=memory_pool, stream=stream):
                    self._draft_step_forward_sampled()
                self._draft_size_to_graph_sampled[bucket] = gs
                # Sparse (top-k) sampled variant, used whenever every request in
                # the batch restricts top_k (the common serving case). Captured
                # separately because ``k_pad`` is baked in; the dense graph above
                # stays as the fallback for unrestricted-top_k batches.
                self._d_topk[:bucket].fill_(self._sparse_kpad_capture
                                            - self._SPARSE_TIE_MARGIN)
                self._draft_step_forward_sampled_sparse()
                torch.cuda.synchronize()
                gsp = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph=gsp, pool=memory_pool, stream=stream):
                    self._draft_step_forward_sampled_sparse()
                self._draft_size_to_graph_sampled_sparse[bucket] = gsp
                self._d_topk[:bucket].fill_(self.memory_manager.vocab_size)

    @torch.inference_mode()
    def _draft_step_forward(self):
        """One MTP draft step over the static draft buffers (captured/replayed).

        Reads ``self._draft_input`` (positions/slot/seq_lens/block already set),
        ``self._d_hidden`` (prev hidden), ``self._d_tok`` (input token); writes
        argmax to ``self._d_next_tok`` and post-norm hidden to ``self._d_out_hidden``.
        """
        nd = self._d_nd
        mtp = self.model.mtp
        out_hidden = mtp.forward(self._draft_input, self._d_hidden[:nd], self._d_tok[:nd])
        self._d_out_hidden[:nd].copy_(out_hidden)
        tok = mtp.logits_from_hidden(out_hidden).argmax(dim=-1).to(torch.int64)
        self._d_next_tok[:nd].copy_(tok)

    @torch.inference_mode()
    def _draft_step_forward_sampled(self):
        """Sampled (rejection) MTP draft step over the static buffers.

        Like :meth:`_draft_step_forward` but draws the draft token from the
        temperature/top-k/top-p transformed distribution via Gumbel-max (graph-
        safe) instead of argmax, and stashes that distribution ``q`` into
        ``_d_q`` (the accept step needs it for ``min(1,p/q)`` + residual). The
        per-seq sampling params (``_d_temp``/``_d_topk``/``_d_topp``) are static
        buffers filled before each replay. RNG uses the default CUDA generator
        (advances across replays); TP consistency is enforced by broadcasting the
        drawn token between replays in :meth:`_draft_chain_graph_sampled` (host
        side, outside the graph).
        """
        nd = self._d_nd
        mtp = self.model.mtp
        out_hidden = mtp.forward(self._draft_input, self._d_hidden[:nd], self._d_tok[:nd])
        self._d_out_hidden[:nd].copy_(out_hidden)
        logits = mtp.logits_from_hidden(out_hidden)
        q = self._mtp_probs_static(
            logits, self._d_temp[:nd], self._d_topk[:nd], self._d_topp[:nd]
        )
        self._d_q[:nd].copy_(q)
        self._d_next_tok[:nd].copy_(self._gumbel_argmax(q))

    @torch.inference_mode()
    def _draft_step_forward_sampled_sparse(self):
        """Sparse (top-k) variant of :meth:`_draft_step_forward_sampled`.

        Same draw, but ``q`` is kept as its top-k support (``_d_qv`` values /
        ``_d_qi`` token ids, ``k_pad`` wide) instead of a dense ``[B, vocab]``
        row. Everything a rejection accept needs is preserved: the drawn token's
        own probability (``_d_qd``) and, for the residual ``(p-q)+``, q's values
        at any token id -- which can only be nonzero inside this support.

        Cost at 64 rows / 248k vocab: one ``topk`` + ``[B, k_pad]`` math (~0.2 ms)
        versus a full-vocab softmax + two renorm passes (~1.5 ms), and the Gumbel
        draw shrinks from ``[B, vocab]`` to ``[B, k_pad]``.
        """
        nd = self._d_nd
        mtp = self.model.mtp
        out_hidden = mtp.forward(self._draft_input, self._d_hidden[:nd], self._d_tok[:nd])
        self._d_out_hidden[:nd].copy_(out_hidden)
        logits = mtp.logits_from_hidden(out_hidden)
        qv, qi = self._mtp_sparse_probs(
            logits,
            self._d_temp[:nd],
            self._d_topk[:nd],
            self._d_topp[:nd],
            self._sparse_kpad_capture,
        )
        self._d_qv[:nd].copy_(qv)
        self._d_qi[:nd].copy_(qi)
        # Gumbel-max over the support, then map the column back to a token id.
        col = self._gumbel_argmax(qv).unsqueeze(1)
        self._d_qd[:nd].copy_(qv.gather(1, col).squeeze(1))
        self._d_next_tok[:nd].copy_(qi.gather(1, col).squeeze(1))

    def _create_dummy_verify_seqs(self, nd: int, qlen: int, ctx: int = 1):
        """Build ``nd`` dummy MTP-verify seqs (each: ``ctx`` cached + ``qlen`` new).

        Each seq references the memory manager's dummy page (all KV reads/writes
        land there harmlessly), is flagged ``_mtp_verify`` so ``cal_input``
        classifies the batch as verify (prefill-with-context, fp8 decode-sparse
        selector + kernel), and has ``computed_token_num=ctx`` /
        ``to_compute_token_num=qlen`` so ``prepare_input`` builds the exact
        uniform 1+k shape the captured graph expects.
        """
        dummy_page = self.memory_manager.dummy_page
        # Dummy SSM block table (all point at the dummy block 0) so the verify
        # metadata (2D block table) is built during graph capture with the same
        # shape as a real verify batch. num_accepted stays 1 (resume col 0).
        k1 = 1 + self._mtp_k
        # Dummy seq ids are offset high to avoid the common case of colliding
        # with a live request id, but correctness does NOT rely on that: we only
        # seed an embedding-cache stub when the id is currently ABSENT, remember
        # exactly which ids we inserted (``self._dummy_verify_cache_ids``), and
        # ``_drop_dummy_verify_cache`` removes only those after the forward. So a
        # dummy id can safely overlap a real one -- we never clobber or delete a
        # genuine entry.
        base_id = 1_000_000
        seeded_ids = []
        seqs = []
        for i in range(nd):
            sid = base_id + i
            s = Sequence(sid, [1] * (ctx + qlen), [], output_len=1)
            # ``prompt_len = ctx`` => computed_prompt True => the seq is
            # decode-classified in the VL mm-prep path (reads a position-delta
            # stub from embedding_cache instead of trying to build image
            # embeddings for a text-only dummy). ``_mtp_verify`` still forces the
            # verify (prefill-with-context) ATTENTION shape independently.
            s.prompt_len = ctx
            npages = (ctx + qlen + self.page_size - 1) // self.page_size
            s.page_table = [dummy_page] * npages
            s.computed_token_num = ctx
            s.to_compute_token_num = qlen
            s._mtp_verify = True
            if self.memory_manager.ssm_segment is not None and self._mtp_k > 0:
                s.ssm_block_table = [0] * k1
                s.ssm_state_slot = 0
                s.ssm_num_accepted = 1
            # Seed a minimal embedding-cache stub so the VL mrope decode branch
            # (``mm_prepare_inputs``) finds a position delta for this dummy --
            # but ONLY if this id isn't already a live entry (never clobber).
            if self.use_mm and self.uses_mrope and sid not in self.embedding_cache:
                self.embedding_cache[sid] = EmbeddingInfo(
                    None, None, torch.zeros(1, dtype=torch.long)
                )
                seeded_ids.append(sid)
            seqs.append(s)
        self._dummy_verify_cache_ids = seeded_ids
        return seqs

    def _drop_dummy_verify_cache(self) -> None:
        """Remove the embedding-cache stubs seeded by the last
        ``_create_dummy_verify_seqs`` so they never leak / shadow a real seq id
        allocated later."""
        for sid in getattr(self, "_dummy_verify_cache_ids", ()):
            self.embedding_cache.pop(sid, None)
        self._dummy_verify_cache_ids = []

    @torch.inference_mode()
    def _capture_verify_graphs(self, memory_pool, stream):
        """Capture the full MTP-verify forward per decode bucket (init-time).

        The verify forward (target model over the uniform 1+k query per decode
        seq) is 99% of MTP step time and pure eager per-layer launch overhead.
        Capture ``self.forward()`` at each bucket into a graph keyed by bucket
        size; ``_mtp_decode`` pads the real verify batch up to a bucket and
        replays. Uses the fp8 decode-sparse verify kernel + batched selector,
        both graph-safe (the selector reads only static-buffer views; the kernel
        + ``get_mla_metadata`` already run inside the captured decode graph).
        """
        qlen = 1 + self._verify_k
        iterator = self.capture_sizes
        if get_local_rank() == 0:
            logger.info(
                f"Capturing MTP verify CUDA graphs (qlen={qlen}) for bucket sizes: "
                f"{list(reversed(self.capture_sizes))}"
            )
            iterator = tqdm(
                self.capture_sizes, desc="Capturing MTP Verify Graphs", ncols=100
            )
        for bucket in iterator:
            seqs = self._create_dummy_verify_seqs(bucket, qlen)
            self.input_data.cal_and_set_input(seqs=seqs)
            # warm up JIT (DeepGEMM/FlashInfer per-M-bucket) outside capture
            self.forward()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph=g, pool=memory_pool, stream=stream):
                self.forward()
            self._verify_size_to_graph[bucket] = g
            self._drop_dummy_verify_cache()

    @torch.inference_mode()
    def _verify_forward_graph(self, decode_seqs, orig_tokens, x1, drafts, nd, kk):
        """Run the verify forward via a captured graph if the batch fits a bucket.

        Sets up ``decode_seqs`` as the uniform ``1+kk`` verify batch, selects the
        smallest captured bucket >= nd, fills the static ``input_data`` buffers,
        replays, and returns ``(v_logits, query_start_loc, v_hidden)`` for the
        REAL rows -- matching the eager path's contract. Returns ``None`` when no
        bucket fits (caller falls back to eager).
        """
        qlen = 1 + kk
        if qlen != 1 + self._verify_k:
            return None  # only the captured qlen is supported
        bucket = None
        for b in sorted(self._verify_size_to_graph.keys()):
            if b >= nd:
                bucket = b
                break
        if bucket is None:
            return None

        # GPU-native prep (default): write the verify batch's per-token arrays
        # straight into the static graph buffers from the staged per-seq facts +
        # the GPU draft tensor -- no Python array rebuild, no dummy pad seqs, no
        # H2D of the token ids. ``drafts_gpu`` is threaded from the draft chain;
        # the eager draft path leaves it unset, so fall back to a one-shot H2D.
        gp = self._mtp_gpu_prep_batch(decode_seqs, orig_tokens, x1, bucket)
        if gp is not None:
            dg = getattr(self, "_drafts_gpu", None)
            if dg is None or tuple(dg.shape) != (nd, kk):
                dg = (
                    torch.tensor(
                        drafts, device=self.input_data.tokens.device, dtype=torch.int64
                    )
                    if kk
                    else self.input_data.tokens.new_zeros((nd, 0))
                )
            gp.fill_verify(self.input_data, qlen, dg)
            self.input_data.mark_gpu_prepared(
                seqs=decode_seqs, num_rows=bucket, qlen=qlen, is_mtp_verify=True
            )
        else:
            drafts = self._drafts_host(drafts, nd, kk)
            for i, s in enumerate(decode_seqs):
                s.token_ids = orig_tokens[i] + [x1[i]] + drafts[i]
                s.computed_token_num = len(orig_tokens[i])
                s.to_compute_token_num = qlen
                s._mtp_verify = True
            pad_seqs = (
                self._create_dummy_verify_seqs(bucket - nd, qlen) if bucket > nd else []
            )
            graph_seqs = list(decode_seqs) + pad_seqs
            self.prepare_input(graph_seqs)
        self._verify_size_to_graph[bucket].replay()
        # Drop the pad dummies' embedding-cache stubs immediately so they can
        # never shadow a real seq id allocated on a later step.
        self._drop_dummy_verify_cache()
        num_real = nd * qlen
        v_hidden = self.output_hidden_states[:num_real]
        # Return the raw verify LOGITS, not the argmax: the greedy accept wants
        # ``argmax`` while the rejection accept wants the transformed prob dist,
        # and computing them from one lm-head pass avoids a second full
        # ``[nd*qlen, vocab]`` GEMM + write (the rejection path used to call
        # ``logits_from_hidden`` again on the same hidden). Everything downstream
        # stays on the GPU; only a tiny ``[nd]``-shaped result is ever D2H'd.
        v_logits = self.model.logits_from_hidden(v_hidden)
        # query_start_loc over the REAL seqs only (uniform qlen).
        qsl = [i * qlen for i in range(nd + 1)]
        return v_logits, qsl, v_hidden

    def _mtp_mrope_deltas(self, seqs) -> Optional[List[int]]:
        """Per-seq mrope position delta (Qwen-VL family), or ``None``.

        Mirrors the decode branch of ``_mm_prepare_cpu``: a decode token's mrope
        position is ``computed_token_num + delta`` on all three rows, with
        ``delta`` stashed in the embedding cache at prefill time. A text-only
        prompt has ``delta == 0``, so a missing entry is not fatal here.
        """
        if not self.uses_mrope:
            return None
        out = []
        for s in seqs:
            info = self.embedding_cache.get(s.seq_id)
            delta = getattr(info, "mrope_position_delta", None) if info else None
            out.append(int(delta) if delta is not None else 0)
        return out

    def _mtp_gpu_prep_batch(self, decode_seqs, orig_tokens, x1, bucket):
        """Stage this MTP step's per-seq facts, returning the prep helper.

        Returns ``None`` when GPU-native prep is disabled (``GLLM_MTP_GPUPREP=0``)
        so callers fall back to the CPU ``cal_input`` builders. Staging is
        memoized per (step, bucket), so calling this from both the draft and the
        verify phase costs one pass.
        """
        gp = self._mtp_gpu_prep
        if gp is None or not self._mtp_gpu_prep_on:
            return None
        gp.push_meta(
            decode_seqs,
            bucket,
            epoch=self._mtp_prep_epoch,
            ctx_lens=[len(t) for t in orig_tokens],
            x1=x1,
            dummy_page=self.memory_manager.dummy_page,
            mrope_deltas=self._mtp_mrope_deltas(decode_seqs),
        )
        return gp

    def _drafts_host(self, drafts, nd: int, kk: int):
        """Host-side ``[nd][kk]`` draft token ids, materializing on demand.

        The graph draft chain returns ``None`` and leaves its output on the GPU
        (``self._drafts_gpu``) so the hot path never syncs mid-step. Only the
        CPU-side fallbacks (eager verify) need the host copy; they pay the D2H
        here.
        """
        if drafts is not None:
            return drafts
        dg = getattr(self, "_drafts_gpu", None)
        if dg is None or nd == 0 or kk == 0:
            return [[] for _ in range(nd)]
        return dg[:nd, :kk].tolist()

    @torch.inference_mode()
    def _mtp_decode(self, hidden: torch.Tensor, x1_tokens: list):
        """MTP speculative decode for a pure-decode batch (greedy, correct-first).

        Returns a list (len == batch size) of per-seq committed-token lists. For
        decode seqs the list is the accepted prefix + bonus (1..k+1 tokens); any
        non-decode seq keeps its single ``x1`` token.

        Correctness model (greedy): with a greedy target and greedy draft, the
        accepted tokens are exactly the tokens the target would have produced
        one-at-a-time, so committing them is identical to non-speculative greedy
        decoding. KV for committed tokens is written by the verify forward into
        the seqs' real slots; rejected-tail slots are simply overwritten next
        step (seq length only advances by the accepted count).
        """
        nd = self.input_data.num_decodes
        k = self._mtp_k
        mtp = self.model.mtp
        decode_seqs = self.input_data.seqs[:nd]
        dev = hidden.device
        # ``hidden`` is a view into the persistent output_hidden_states buffer;
        # clone so subsequent forwards (draft/verify) can't mutate it underfoot.
        hidden = hidden[:nd].clone()

        x1 = [int(x1_tokens[i]) for i in range(nd)]
        # x1 is already TP-synchronized by ``step_once`` (it broadcasts the
        # sampled x1 from TP-rank-0 before calling this method whenever sampling
        # is active), so no broadcast here. Runtime dispatch: if ANY seq in the
        # batch samples (temperature != 1 or top_k != 1) take the lossless
        # rejection path; an all-greedy batch takes the argmax fast path. No env
        # flag -- purely data-driven.
        _rej_active = self._mtp_can_sample and any(
            (s.temperature > 1e-5 and abs(s.temperature - 1.0) > 1e-5) or s.top_k != 1
            for s in decode_seqs
        )
        # Original committed token_ids / lengths (pre spec-mutation). These are
        # REFERENCES, not copies: every spec mutation below rebinds
        # ``seq.token_ids`` to a freshly concatenated list and never mutates the
        # committed one in place, so ``restore()`` can just rebind the original
        # object back. Copying them cost O(context) per seq per step -- ~1-2 ms at
        # 64x2k context, i.e. the step got more expensive the longer the requests
        # ran, for nothing.
        orig_tokens = [s.token_ids for s in decode_seqs]
        orig_ctn = [s.computed_token_num for s in decode_seqs]
        orig_tctn = [s.to_compute_token_num for s in decode_seqs]
        self._mtp_prep_epoch += 1
        # Pre-allocate the KV pages for the WHOLE speculative window once: the
        # draft chain writes tokens at ctx .. ctx+k-1 and the verify forward at
        # ctx .. ctx+k, so the verify shape's allocation covers both phases.
        # Doing it here (instead of once per phase) also freezes every seq's page
        # table for the rest of the step, which is what lets the GPU-native prep
        # stage the page tables a single time (see ``_mtp_gpu_prep_batch``).
        for i, s in enumerate(decode_seqs):
            s.token_ids = orig_tokens[i] + [x1[i]] + [0] * k
            s.computed_token_num = len(orig_tokens[i])
            s.to_compute_token_num = 1 + k
        self.memory_manager.pre_allocate_page(decode_seqs, cacheable=False)

        # --- Hybrid GDN recurrent-state: block-table column commit ---
        # The verify forward (GDN ``_forward_mtp_verify``) wrote each of the
        # ``1+k`` verify tokens' post-state into that column of every seq's SSM
        # block table (column 0 held the committed pre-x1 state on entry). After
        # the accept step decides how many drafts each seq kept (``na``), the
        # exact post-acceptance state (after ``1+na`` committed tokens) sits at
        # column ``na``. We copy it back into column 0 so the plain decode /
        # snapshot paths keep reading column 0. Pure-attention models (DeepSeek
        # MTP) have no SSM segment and skip this.
        _ssm_seg = getattr(self.memory_manager, "ssm_segment", None)
        _has_gdn = (
            _ssm_seg is not None
            and all(s.ssm_block_table is not None for s in decode_seqs)
        )

        def restore():
            # NOTE: deliberately do NOT restore page_table. The verify forward
            # wrote each committed token's KV into the pages allocated during
            # verify; the scheduler will commit ``m`` tokens and must keep those
            # exact pages so the next decode step reads valid KV. Restoring the
            # page_table here would orphan the verify pages and let the scheduler
            # hand out different physical slots (garbage KV) -> divergence.
            for i, s in enumerate(decode_seqs):
                # Rebind the original list object (see the ``orig_tokens``
                # comment above -- it was never mutated in place).
                s.token_ids = orig_tokens[i]
                s.computed_token_num = orig_ctn[i]
                s.to_compute_token_num = orig_tctn[i]
                s._mtp_verify = False

        # --- 1. Draft k tokens per seq by looping the MTP head. ---
        # Two paths: a CUDA-graph replay path (``_mtp_draft_graph``) that captures
        # one draft-step graph per decode bucket and replays it k times with
        # in-place GPU buffer advance (no per-step Python / H2D / .item()); and an
        # eager fallback. Both produce ``drafts`` = per-seq [d1..dk] on CPU.
        # ``_rej_active`` (computed above for the x1 broadcast) also selects the
        # sampling draft chain, which keeps the per-step draft dist ``q``.
        _use_rej = _rej_active
        q_dists = None
        # Sparse (top-k) rejection path: decided ONCE per step so the draft's ``q``
        # and the verify's ``p`` are always built by the same code (they must live
        # on the same transformed space for the accept test to be exact).
        _sparse = _rej_active and self._mtp_sparse_eligible(decode_seqs)
        # One TP-synced generator per MTP step, used by BOTH the eager sampled
        # draft chain (if taken) AND the accept-step residual/bonus draws below.
        # (The graph draft path uses the default CUDA generator internally and
        # broadcasts its tokens, so it doesn't consume ``gen``.)
        gen = self._mtp_rng_step(dev) if _use_rej else None
        # Reset the GPU-draft stash; only the greedy graph draft chain fills it
        # (the accept step reads it to skip an H2D of the host drafts list).
        self._drafts_gpu = None
        if _use_rej:
            # Sampling draft chain: prefer the captured Gumbel-max graph (draft
            # forward + sampling in-graph, token broadcast between replays); fall
            # back to eager when graphs are off or the batch exceeds the max
            # bucket. Both draw from q and keep the per-step q dist for accept.
            _sg = (
                self._draft_size_to_graph_sampled_sparse
                if _sparse
                else self._draft_size_to_graph_sampled
            )
            if self._mtp_draft_graph and _sg and nd <= max(_sg.keys()):
                drafts, q_dists = self._draft_chain_graph_sampled(
                    decode_seqs, orig_tokens, x1, hidden, k, nd, sparse=_sparse
                )
            else:
                drafts, q_dists = self._draft_chain_eager_sampled(
                    decode_seqs, orig_tokens, x1, hidden, k, nd, gen, sparse=_sparse
                )
        elif self._mtp_draft_graph and self.capture_sizes and nd <= max(self.capture_sizes):
            drafts = self._draft_chain_graph(decode_seqs, orig_tokens, x1, hidden, k, nd)
        else:
            drafts = self._draft_chain_eager(decode_seqs, orig_tokens, x1, hidden, k, nd)
        restore()

        # --- 2. Verify: one base forward over [x1, d1..dk] per seq. ---
        # ``drafts is None`` == the graph draft chain kept its output on the GPU
        # (``self._drafts_gpu``); it always drafts the full ``k``.
        kk = (k if drafts is None else len(drafts[0])) if nd else 0
        # Fast path: replay the captured verify graph (collapses the eager
        # per-layer launch overhead). Only when MTP verify graphs are captured
        # and the batch fits a bucket; falls back to the eager forward otherwise.
        _v_done = False
        if (
            self._mtp_verify_graph
            and self._verify_size_to_graph
            and nd <= max(self._verify_size_to_graph.keys())
        ):
            _res = self._verify_forward_graph(decode_seqs, orig_tokens, x1, drafts, nd, kk)
            if _res is not None:
                v_logits, qsl, v_hidden = _res
                _v_done = True
        if not _v_done:
            drafts = self._drafts_host(drafts, nd, kk)
            for i, s in enumerate(decode_seqs):
                s.token_ids = orig_tokens[i] + [x1[i]] + drafts[i]  # +1+kk new
                s.computed_token_num = len(orig_tokens[i])          # context cached
                s.to_compute_token_num = 1 + kk
                s._mtp_verify = True   # force the prefill-with-context attn path
            self.memory_manager.pre_allocate_page(decode_seqs, cacheable=False)
            self.prepare_input(decode_seqs)
            v_out = self.model(self.input_data)
            v_hidden = v_out[0] if isinstance(v_out, tuple) else v_out
            # Verify seqs are uniform ``1+kk`` query length, so ``qsl`` is the
            # uniform stride. Logits stay raw (see ``_verify_forward_graph``).
            v_logits = self.model.logits_from_hidden(v_hidden)
            qsl = [i * (1 + kk) for i in range(nd + 1)]

        # Rejection mode: the target dist ``p`` at each of the 1+kk verify
        # positions, in the SAME transformed space as the draft dist ``q``
        # (temperature + top-k/top-p renorm) -- that equality is what makes the
        # accept test distribution-lossless. Rows are laid out
        # [seq0: x1,d1..dk | seq1: ...]; the per-seq sampling params are expanded
        # on-device with ``repeat_interleave`` instead of building a 256-entry
        # python seq list per step.
        p_dists = None          # dense [num_v, vocab] (unrestricted-top_k batches)
        p_sparse = None         # (vals, idx) [nd, 1+kk, k_pad] (top_k-restricted)
        if _use_rej:
            num_v = nd * (1 + kk)
            temps, top_ks, top_ps = self._mtp_sample_params(decode_seqs, dev)
            rep = 1 + kk
            t_r = temps.repeat_interleave(rep, dim=0)
            k_r = top_ks.repeat_interleave(rep)
            p_r = top_ps.repeat_interleave(rep)
            if _sparse:
                # Only the top-k support can carry probability, so keep p in the
                # sparse form: one ``topk`` instead of a full-vocab softmax + two
                # renorm passes over ``[nd*(1+k), vocab]`` (3.1 ms -> 0.6 ms at
                # nd=64, and 254 MB of dense probs never materialized).
                k_pad = self._mtp_kpad(decode_seqs)
                pv, pi = self._mtp_sparse_probs(
                    v_logits[:num_v], t_r, k_r, p_r, k_pad
                )
                p_sparse = (pv.view(nd, rep, -1), pi.view(nd, rep, -1))
            else:
                p_dists = self._mtp_probs_static(
                    v_logits[:num_v], t_r, k_r, p_r
                )  # [num_v, vocab]

        # --- 3. Greedy accept per seq. ---
        # Verify inputs per seq are [x1, d1..dk] at positions start..start+k.
        # v_pred[start+p] = target's greedy token AFTER consuming input p. So the
        # target's token following x1 is v_pred[start+0]; accept d_{p+1} iff it
        # equals v_pred[start+p]. Commit x1 always; append accepted drafts; the
        # bonus token is the target prediction at the first rejection (or after
        # the last accepted draft).
        # Commit x1 + the longest prefix of drafts the target agrees with. We do
        # NOT commit a trailing "bonus"/corrected token: the next normal decode
        # step produces it through the exact decode path, avoiding any reliance
        # on the verify forward's prediction at the last (post-drafts) position.
        # Every committed token here was a verify INPUT, so its KV is valid.
        # Size to ``nd`` (the real decode seqs captured at entry). The graph path
        # leaves ``self.input_data.seqs`` holding bucket-padded seqs, so it is no
        # longer a reliable count here -- use ``nd`` directly. This batch is pure
        # decode/verify (no real non-decode seqs), so there is no tail to fill.
        results = [None] * nd
        n_accepted = [0] * nd   # accepted DRAFT tokens per seq (excludes x1/bonus)
        _fused_stash = self._mtp_fused  # relay bonus+hidden for next step's draft
        new_relay = {} if _fused_stash else None

        if _use_rej:
            # --- 3b. Rejection-sampling accept (distribution-lossless). ---
            # x1 is already a proper target sample (step_once's sampler), always
            # committed. For each draft d_p ~ q, accept with prob min(1, p/q);
            # on reject, resample the position from the residual (p-q)+ and stop;
            # if all accepted, sample a bonus from p at the last position.
            #
            # Bonus handling matches the fused/non-fused split of the GREEDY path:
            #   * non-fused: append the bonus to ``committed`` (it rides the
            #     scheduler's uncached-tail path -- reprocessed next decode step).
            #   * fused: DON'T commit the bonus this step; relay (bonus_tok,
            #     bonus_hidden) as the NEXT step's x1 (committed there). Committing
            #     it both here AND as next-step x1 would double-emit it (observed
            #     as token repetition). ``bonus_hidden`` is the verify hidden at
            #     the last-accepted position (start+na) -- exactly the state the
            #     bonus was sampled from, so it correctly seeds the next draft.
            # Fully vectorized on the GPU. The previous per-seq python loop read
            # ``float(q_dists[i,p,d])`` / ``float(px_all[start+p,d])`` -- each of
            # those is a device-scalar sync, up to ``2*nd*kk`` (384 at nd=64) per
            # step -- and then ran one ``torch.multinomial`` + ``.item()`` per
            # sequence. That accept phase measured 20.5 ms of a 34.6 ms sampling
            # step. Here every decision is a batched tensor op and the host learns
            # the outcome through ONE packed D2H at the end.
            qlen = 1 + kk
            seq_ar = torch.arange(nd, device=dev)
            sparse = p_sparse is not None
            if sparse:
                p3_vals, p3_idx = p_sparse          # [nd, qlen, k_pad] each
            else:
                p3 = p_dists.view(nd, qlen, p_dists.shape[-1])
            if kk > 0:
                dg = getattr(self, "_drafts_gpu", None)
                if dg is not None and tuple(dg.shape) == (nd, kk):
                    d_gpu = dg.to(torch.int64)
                else:
                    d_gpu = torch.tensor(drafts, device=dev, dtype=torch.int64)
                idx = d_gpu.unsqueeze(-1)
                if sparse:
                    # p(d_p): the drafted token either sits in p's kept support
                    # (pick its value) or outside it, where p is exactly 0 -- the
                    # same value the dense gather would return.
                    hit = (p3_idx[:, :kk] == idx).to(p3_vals.dtype)
                    p_d = (p3_vals[:, :kk] * hit).sum(dim=-1)      # [nd,kk] p(d_p)
                    # q(d_p) was recorded by the draft step that drew it.
                    q_d = q_dists.drawn
                else:
                    p_d = p3[:, :kk].gather(2, idx).squeeze(-1)    # [nd,kk] p(d_p)
                    q_d = q_dists.dense.gather(2, idx).squeeze(-1)  # [nd,kk] q(d_p)
                # accept d_p with prob min(1, p/q); q<=0 can only happen for a
                # token q never proposes, so treat it as certain acceptance
                # (matches the old scalar branch).
                ratio = torch.where(
                    q_d > 0, (p_d / q_d.clamp_min(1e-30)).clamp_max(1.0),
                    torch.ones_like(p_d),
                )
                u = torch.rand((nd, kk), generator=gen, device=dev)
                accept = u < ratio                                 # [nd,kk] bool
                # Leading all-accept prefix length (same cumprod trick as greedy).
                na_gpu = torch.cumprod(accept.to(torch.int32), dim=1).sum(dim=1)
            else:
                d_gpu = None
                na_gpu = torch.zeros(nd, dtype=torch.int64, device=dev)
            na_l = na_gpu.to(torch.long)
            # Bonus draw. Rejected at position ``na`` (``na < kk``): draw from the
            # residual ``(p-q)+`` there. All accepted (``na == kk``): draw from p
            # at the tail position. Both are row ``na``, so one gather serves both
            # cases and a single batched multinomial does the draw.
            if sparse:
                # ``(p-q)+`` is supported inside p's kept set (outside it p == 0,
                # so the clamp is 0 there), which is why the residual only needs
                # p's ``k_pad`` columns plus q's values at those same token ids.
                p_row = p3_vals[seq_ar, na_l]                      # [nd, k_pad]
                p_row_idx = p3_idx[seq_ar, na_l]                   # [nd, k_pad]
            else:
                p_row = p3[seq_ar, na_l]                           # [nd, V]
            if kk > 0:
                sel = na_l.clamp(max=kk - 1)
                if sparse:
                    # q's values at p's kept token ids. Everything outside q's own
                    # support is q == 0, so matching the two id lists (k_pad x
                    # k_pad per row, ~16k comparisons) is all that's needed.
                    q_row_v = q_dists.vals[seq_ar, sel]            # [nd, k_pad]
                    q_row_i = q_dists.idx[seq_ar, sel]             # [nd, k_pad]
                    match = (q_row_i.unsqueeze(1) == p_row_idx.unsqueeze(2))
                    q_at_p = (match.to(q_row_v.dtype) * q_row_v.unsqueeze(1)).sum(-1)
                else:
                    q_at_p = q_dists.dense[seq_ar, sel]            # [nd, V]
                resid = torch.where(
                    (na_gpu < kk).unsqueeze(1),
                    (p_row - q_at_p).clamp_min(0),
                    p_row,
                )
                # Degenerate rows (p == q over the whole support) carry no mass;
                # fall back to sampling p, as the scalar path did.
                resid = torch.where(
                    resid.sum(dim=1, keepdim=True) > 1e-12, resid, p_row
                )
            else:
                resid = p_row
            bonus_gpu = torch.multinomial(resid, 1, generator=gen).squeeze(1)
            if sparse:
                # Map the drawn column back to a token id.
                bonus_gpu = p_row_idx.gather(1, bonus_gpu.unsqueeze(1)).squeeze(1)
            # ONE D2H: [n_accepted | bonus | drafts...] -- same packing as the
            # greedy accept, so the draft chain's tokens also arrive here.
            rows = [na_gpu.to(torch.int64), bonus_gpu.to(torch.int64)]
            if d_gpu is not None:
                rows.extend(d_gpu.t())
            packed_cpu = torch.stack(rows).cpu()                   # [2+kk, nd]
            na_cpu = packed_cpu[0].tolist()
            bonus_cpu2 = packed_cpu[1].tolist()
            drafts_cpu = packed_cpu[2:].t().tolist() if d_gpu is not None else None
            if _fused_stash:
                # Batch-gather the per-seq draft-seed hidden (verify row
                # ``i*qlen + na``) in one op instead of nd tiny clones.
                bonus_hidden_all = v_hidden.index_select(0, seq_ar * qlen + na_l)
            for i, s in enumerate(decode_seqs):
                na = na_cpu[i]
                committed = [x1[i]] + (drafts_cpu[i][:na] if na else [])
                if _fused_stash:
                    # Relay the bonus as next step's x1; do NOT commit it now.
                    new_relay[s.seq_id] = (bonus_cpu2[i], bonus_hidden_all[i])
                else:
                    committed.append(bonus_cpu2[i])
                n_accepted[i] = na
                results[i] = committed
            # The accept decisions used per-rank distributions (p/q differ by fp
            # all-reduce epsilon across TP ranks) + per-rank RNG draws, so
            # ``results`` / ``n_accepted`` / the relayed bonus can diverge. Make
            # TP-rank-0's decisions authoritative: broadcast a padded token grid
            # (committed lists) + the relayed bonus token, then every rank rebuilds
            # identical state. (Draft tokens were already broadcast; the accept +
            # bonus draws happen here.)
            if get_tp_size() > 1:
                # ``results`` holds [x1 + accepted_drafts] (+ bonus only when NOT
                # fused). Max len = x1 + kk drafts + (bonus if not fused).
                maxlen = kk + 1 + (0 if _fused_stash else 1)
                grid = torch.full((nd, maxlen), -1, dtype=torch.int64, device=dev)
                lens = torch.zeros(nd, dtype=torch.int64, device=dev)
                bonus_t = torch.zeros(nd, dtype=torch.int64, device=dev)
                if get_tp_rank() == 0:
                    for i in range(nd):
                        c = results[i]
                        lens[i] = len(c)
                        grid[i, : len(c)] = torch.tensor(c, dtype=torch.int64, device=dev)
                        if _fused_stash:
                            bonus_t[i] = new_relay[decode_seqs[i].seq_id][0]
                src = get_rank() - get_tp_rank()
                dist.broadcast(lens, src=src, group=get_ipc_tp_group())
                dist.broadcast(grid, src=src, group=get_ipc_tp_group())
                if _fused_stash:
                    dist.broadcast(bonus_t, src=src, group=get_ipc_tp_group())
                lens_cpu = lens.cpu().tolist()
                grid_cpu = grid.cpu().tolist()
                bonus_cpu = bonus_t.cpu().tolist()
                for i in range(nd):
                    n = lens_cpu[i]
                    results[i] = grid_cpu[i][:n]
                    # committed = [x1] + accepted_drafts (+ bonus if not fused).
                    # n_accepted = accepted drafts = n - x1(1) - (bonus if in results).
                    n_accepted[i] = max(0, n - (1 if _fused_stash else 2))
                    if _fused_stash:
                        # Adopt rank-0's bonus token; keep this rank's own hidden
                        # (only seeds the next draft, whose token is broadcast).
                        _, h = new_relay[decode_seqs[i].seq_id]
                        new_relay[decode_seqs[i].seq_id] = (bonus_cpu[i], h)
        else:
            # --- 3. Greedy accept per seq (vectorized on GPU). ---
            # Verify inputs per seq are [x1, d1..dk] at positions start..start+k.
            # v_pred[start+p] = target's greedy token AFTER consuming input p, so
            # accept d_{p+1} iff it equals v_pred[start+p]; the accepted count is
            # the length of the leading all-match prefix. ``v_pred`` is a GPU
            # tensor ``[nd*qlen]`` (qlen == 1+kk, uniform); compute ``n_accepted``
            # on-device and D2H only the tiny ``[nd]`` results (+ ``[nd]`` fused
            # bonus tokens), instead of the old blocking ``[nd*qlen]`` v_pred D2H.
            # ``results`` is rebuilt on the host from the already-known ``x1`` /
            # ``drafts`` lists sliced to ``n_accepted`` -- no full v_pred needed.
            qlen = 1 + kk
            # Greedy target: one argmax over the verify logits (the rejection
            # branch instead transforms the same logits into ``p``).
            vp = v_logits[: nd * qlen].argmax(dim=-1).view(nd, qlen)
            seq_ar = torch.arange(nd, device=dev)
            if kk > 0:
                # Prefer the GPU draft tensor threaded from the draft chain (the
                # graph chain never materializes the drafts on the host at all).
                # Fall back to a one-shot H2D only when a host-side chain ran.
                dg = getattr(self, "_drafts_gpu", None)
                if dg is not None and tuple(dg.shape) == (nd, kk):
                    drafts_gpu = dg.to(vp.dtype)
                else:
                    drafts_gpu = torch.tensor(drafts, device=dev, dtype=vp.dtype)
                match = (vp[:, :kk] == drafts_gpu)                    # [nd,kk] bool
                # cumprod over the bool prefix: 1 until the first mismatch, 0 after
                # -> sum = length of the leading all-accept prefix.
                na_gpu = torch.cumprod(match.to(torch.int32), dim=1).sum(dim=1)  # [nd]
            else:
                drafts_gpu = None
                na_gpu = torch.zeros(nd, device=dev, dtype=torch.int32)
            # ONE D2H per step carries everything the host still needs:
            #   row 0        : n_accepted
            #   row 1        : the fused bonus token (ignored when not fused)
            #   rows 2..2+kk : the draft token grid (transposed), so the draft
            #                  chain never has to sync mid-step just to give the
            #                  commit loop its token ids.
            rows = [na_gpu.to(torch.int64)]
            rows.append(
                vp[seq_ar, na_gpu.to(torch.long)].to(torch.int64)
                if _fused_stash
                else na_gpu.to(torch.int64)
            )
            if drafts_gpu is not None:
                rows.extend(drafts_gpu.to(torch.int64).t())
            packed_cpu = torch.stack(rows).cpu()   # [2+kk, nd]
            na_cpu = packed_cpu[0].tolist()
            bonus_cpu2 = packed_cpu[1].tolist() if _fused_stash else None
            drafts_cpu = packed_cpu[2:].t().tolist() if drafts_gpu is not None else None
            if _fused_stash:
                # Batch-gather every seq's bonus hidden in ONE op (verify row
                # ``i*qlen + na``) instead of nd separate per-seq ``.clone()``s
                # (those were 64 tiny D2D copies per step). Row i of this tensor
                # is seq i's draft-seed hidden for the next step.
                bonus_rows = seq_ar * qlen + na_gpu.to(torch.long)
                bonus_hidden_all = v_hidden.index_select(0, bonus_rows)  # [nd, H]
            for i, s in enumerate(decode_seqs):
                na = na_cpu[i]
                n_accepted[i] = na
                # committed = x1 + the accepted draft prefix.
                results[i] = [x1[i]] + (drafts_cpu[i][:na] if na else [])
                if _fused_stash:
                    # bonus token from the on-device gather; bonus hidden is row i
                    # of the batched gather (a GPU view; only seeds the next draft).
                    new_relay[s.seq_id] = (bonus_cpu2[i], bonus_hidden_all[i])

        self._record_mtp_metrics(nd, kk, n_accepted)

        if _fused_stash:
            # Replace the relay map wholesale so seqs absent from this batch are
            # evicted (their stale hidden must never seed a later draft).
            self._mtp_relay = new_relay

        # --- Hybrid GDN recurrent-state: commit the accepted column to col 0 ---
        # The verify forward wrote token t's post-state into block-table column
        # t. Each seq committed ``1+na`` tokens, so the post-acceptance state is
        # already sitting in the block at column ``na``. Rather than COPY that
        # block's 18.6MB (all 18 layers) into column 0 -- pure memory-bandwidth
        # cost, ~2.2ms at nd=64 -- we just SWAP the two block-table entries: the
        # physical block holding the committed state becomes column 0, and the
        # old column-0 block moves to column ``na`` where it's overwritten as
        # scratch by the next verify. O(1) per seq, zero data movement. The next
        # step rebuilds ``ssm_block_table_2d`` / ``ssm_state_slot`` from the
        # (now-permuted) list, so decode/snapshot read the committed state from
        # column 0 as before. na==0 -> committed state already at column 0.
        if _has_gdn:
            for i in range(nd):
                na = n_accepted[i]
                if na > 0:
                    bt = decode_seqs[i].ssm_block_table
                    bt[0], bt[na] = bt[na], bt[0]
                    # Keep the scalar slot mirror consistent with column 0 (read
                    # by the plain decode path + prefix-cache snapshot capture).
                    decode_seqs[i].ssm_state_slot = bt[0]
                # Reset the persisted resume column: committed state is now at
                # column 0, so next step's num_accepted is neutral (1).
                decode_seqs[i].ssm_num_accepted = 1

        restore()
        return results

    @torch.inference_mode()
    def step_once(self, dp_padded_size: Optional[int] = None):
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        # Fused MTP fast path: when every seq in a pure-decode batch carries relay
        # state from the immediately-preceding step, skip the leading x1-decode
        # forward entirely -- the previous verify already produced each seq's
        # bonus token + hidden, which seed this step's draft. One target forward
        # (verify) per step instead of two. Falls back to the normal path (which
        # runs the decode forward and stashes relay) on any relay miss.
        if (
            self._mtp_fused
            and dp_padded_size is None
            and not is_dp_attn()
            and is_last_pp_rank()
            and self.input_data.num_prefills == 0
            and self.input_data.num_decodes > 0
            and self.mtp_speculate_batch(self.input_data.num_decodes)
            and self.check_decode_batch()
        ):
            seqs = self.input_data.seqs[: self.input_data.num_decodes]
            if seqs and all(s.seq_id in self._mtp_relay for s in seqs):
                relay = [self._mtp_relay[s.seq_id] for s in seqs]
                x1 = [r[0] for r in relay]
                hidden = torch.stack([r[1] for r in relay], dim=0)
                # Fused bypasses the sampler/logprobs block below; the relayed
                # x1 (last step's bonus) is committed by ``_mtp_decode``. Works
                # for both greedy (bonus = argmax) and sampling (bonus = the
                # rejection resample), since the sampling accept relays-not-commits
                # the bonus when fused (no double-commit).
                self._last_logprobs = None
                return self._mtp_decode(hidden, x1)
        if dp_padded_size is not None:
            # DP+EP CUDA-graph decode: every group pads to the *same*
            # group-wide bucket (chosen by the driver via ``dp_select_bucket``)
            # so the captured global ``dp_size * bucket`` MoE batch matches.
            num_real_tokens = self.input_data.pad_for_cuda_graph(dp_padded_size)
            self.size_to_graph[dp_padded_size].replay()
            num_cal_tokens = num_real_tokens
        elif is_dp_attn():
            # DP+EP eager path (prefill / mixed, or bucket miss): plain forward.
            # The per-group bucket decision was already made by the driver, so
            # never fall into the local-only bucket selection below.
            self.forward()
        # Only pure decode batches use CUDA graph.
        elif self.check_decode_batch():
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
            hidden = self.output_hidden_states[:num_cal_tokens]
            logits = self.model.compute_logits(self.input_data, hidden)
            self.input_data.prepare_sample()
            # Logprobs are only computable on the output rank (only it holds the
            # gathered full-vocab logits) and only worth the extra full-vocab
            # log_softmax + top-k when some seq in the batch asked for them.
            # ``_last_logprobs`` (per-batch-row list) is picked up by the worker
            # and travels with ``next_tokens`` -- including back to rank 0 over
            # the token socket under PP>1, where the sampling rank is a follower.
            seqs = self.input_data.seqs
            self._last_logprobs = None
            if is_output_rank() and any(s.logprobs_enabled for s in seqs):
                num_logprobs = max(
                    (s.num_top_logprobs for s in seqs if s.logprobs_enabled),
                    default=0,
                )
                next_tokens_gpu, logprobs = self.sampler.forward_gpu(
                    logits, self.input_data, True, num_logprobs
                )
                self._last_logprobs = self._build_logprob_rows(seqs, logprobs)
                next_tokens = next_tokens_gpu.cpu().tolist()
            else:
                next_tokens = self.sampler.forward(logits, self.input_data)
            # Prompt logprobs re-enter the LM head (a TP all-gather), so this
            # runs on ALL last-PP TP ranks (not just the output rank) to keep
            # the collective balanced; every rank has the same real seqs, so
            # they compute identical data and only tp0's copy is shipped.
            self._compute_prompt_logprobs(seqs, hidden)
            # MTP speculative decoding: on a pure-decode batch, draft k tokens
            # per seq with the MTP head and verify them with one base forward,
            # committing the accepted prefix. Returns per-seq token LISTS.
            if (getattr(self.model, "mtp", None) is not None
                    and self._mtp_k > 0
                    and self.input_data.num_prefills == 0
                    and self.input_data.num_decodes > 0
                    and not self.mtp_speculate_batch(self.input_data.num_decodes)):
                # Batch too large to profit from speculation: this step already
                # sampled one token per seq the plain way, which is the answer.
                # The relay it leaves behind is stale (see ``_mtp_drop_relay``).
                self._mtp_drop_relay()
            elif (getattr(self.model, "mtp", None) is not None
                    and self._mtp_k > 0
                    and self.input_data.num_prefills == 0
                    and self.input_data.num_decodes > 0):
                # x1 (this decode batch's first target token) was just sampled by
                # the per-rank sampler above. Under GREEDY (top_k==1 -> argmax) it
                # is TP-deterministic, but under SAMPLING each TP rank draws
                # independently -> x1 diverges -> the draft/verify forwards in
                # ``_mtp_decode`` get different inputs -> the seq token_ids diverge
                # -> next-iter scheduling / overlap ``_gpu_pending`` depth diverges
                # across ranks -> NCCL deadlock. MTP's whole sync design assumes
                # TP-identical tokens. So when any seq samples, make TP-rank-0's x1
                # authoritative before ``_mtp_decode`` touches the model. (This is
                # independent of rejection sampling -- plain sampling + MTP needs
                # it too.) Greedy skips the broadcast (argmax already matches).
                nd_dec = self.input_data.num_decodes
                dec_seqs = self.input_data.seqs[:nd_dec]
                _sampling = any(
                    (s.temperature > 1e-5 and abs(s.temperature - 1.0) > 1e-5)
                    or s.top_k != 1
                    for s in dec_seqs
                )
                if _sampling and get_tp_size() > 1:
                    x1_t = torch.tensor(
                        next_tokens[:nd_dec], dtype=torch.int64, device=hidden.device
                    )
                    self._mtp_bcast_tp(x1_t)
                    x1_list = x1_t.tolist()
                    for _i in range(nd_dec):
                        next_tokens[_i] = x1_list[_i]
                return self._mtp_decode(hidden, next_tokens)
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

    def register_decode_page_hash(self, seq: Sequence, pos: int) -> None:
        """Register the prefix-cache page hash for a decode boundary the seq
        just completed with a *real* (finalized) token at ``seq.token_ids[pos]``.

        Called from the scheduler's output-finalization hooks
        (``Scheduler.process_output`` after appending the real token, and
        ``OverlapScheduler.process_output_finalize`` after overwriting the
        placeholder). Keeping the trigger here -- rather than inside
        ``MemoryManager.pre_allocate_page`` -- guarantees the hash is only ever
        computed over real tokens, never an unfinalized overlap placeholder
        (see ``docs/prefix_cache_overlap_poisoning.md``). No-op for caches
        without prefix support.
        """
        self.memory_manager.register_decode_boundary(seq, pos)

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
        # Double-buffered pinned staging for per-token logprobs, mirroring
        # ``_next_tokens_bufs`` (keyed by the same ``buf_idx``). Only written on
        # the output rank and only when a batch requested logprobs; sized to the
        # OpenAI ``top_logprobs`` ceiling so the top-k columns never overflow.
        self._max_top_logprobs = 20
        self._lp_sampled_bufs = [
            torch.zeros(
                self.max_running_seqs,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self._lp_topval_bufs = [
            torch.zeros(
                (self.max_running_seqs, self._max_top_logprobs),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self._lp_topid_bufs = [
            torch.zeros(
                (self.max_running_seqs, self._max_top_logprobs),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        ]
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

    def _run_forward_on_stream(
        self, num_cal_tokens: int, dp_padded_size: Optional[int] = None
    ) -> int:
        if dp_padded_size is not None:
            # DP+EP graph decode: replay the driver-agreed group-wide bucket so
            # the captured global ``dp_size * bucket`` MoE batch matches.
            num_cal_tokens = self.input_data.pad_for_cuda_graph(dp_padded_size)
            self.size_to_graph[dp_padded_size].replay()
        elif is_dp_attn():
            # DP+EP eager path (prefill / mixed / bucket miss): the driver
            # already made the per-group decision, so never fall into the
            # local-only bucket selection below.
            self.forward()
        elif self.check_decode_batch():
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
    def run_batch_async(
        self, dp_padded_size: Optional[int] = None
    ) -> Tuple[torch.cuda.Event, int, List[int], int]:
        """Launch forward + sample on forward_stream (pp_size=1 only).

        ``dp_padded_size`` (DP+EP only) forces the graph bucket agreed on by the
        driver across all DP groups, keeping the captured MoE collectives'
        shapes identical world-wide; ``None`` runs eager (prefill/mixed) or the
        normal local bucket selection (non-DP).
        """
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

            num_cal_tokens = self._run_forward_on_stream(
                num_cal_tokens, dp_padded_size=dp_padded_size
            )

            hidden = self.output_hidden_states[:num_cal_tokens]
            logits = self.model.compute_logits(self.input_data, hidden)
            self.input_data.prepare_sample()
            next_tokens_gpu = None
            # ``lp_k`` doubles as the "logprobs requested this batch" flag for
            # the collect side: ``None`` => none requested (skip staging),
            # >= 0 => number of top alternatives staged. Only the output rank
            # holds full logits, so only it computes/stages logprobs.
            lp_k = None
            lp_gpu = None
            # Determinism/deadlock note: ``compute_logits`` all-gathers, so EVERY
            # TP rank holds full logits and CAN sample. Historically only the
            # output rank sampled (others received the broadcast), but under
            # non-greedy sampling that made the output rank do a heavier kernel
            # (multi-round rejection sampling) than its peers every step. In the
            # overlap pipeline that per-rank GPU-time asymmetry lets ranks drift
            # out of lockstep, and the get_tp_group collective sequence
            # (graph all-reduce -> LM-head all-gather -> token broadcast) then
            # interleaves across iterations -> NCCL deadlock (greedy's argmax is
            # cheap enough to hide it, which is why it only surfaced with
            # sampling). Fix: run the SAMPLER on every rank so the per-iteration
            # GPU work + collective cadence is identical across ranks. The result
            # still diverges by fp all-reduce epsilon (sampling amplifies it), so
            # the broadcast below keeps the output rank's draw authoritative for
            # correctness -- but the timing is now symmetric, which is what makes
            # the pipeline deadlock-free by construction.
            _all_greedy = all(s.top_k == 1 for s in self.input_data.seqs)
            _sample_here = is_output_rank() or not _all_greedy
            if _sample_here:
                seqs = self.input_data.seqs
                if is_output_rank() and any(s.logprobs_enabled for s in seqs):
                    lp_k = min(
                        self._max_top_logprobs,
                        max(
                            (s.num_top_logprobs for s in seqs if s.logprobs_enabled),
                            default=0,
                        ),
                    )
                    next_tokens_gpu, lp_gpu = self.sampler.forward_gpu(
                        logits, self.input_data, True, lp_k
                    )
                else:
                    _nt = self.sampler.forward_gpu(logits, self.input_data)
                    # Non-output ranks discard their own draw (only run it for
                    # timing symmetry); the broadcast overwrites it anyway.
                    next_tokens_gpu = _nt
            # Prompt logprobs accumulate directly onto the (real) seqs; the
            # scheduler ships the completed list once the prompt finishes
            # prefill. Gated inside the helper on ``prompt_logprobs_enabled``.
            # Run on ALL last-PP TP ranks (outside the output-rank block): it
            # re-enters the LM head (a TP all-gather) and must stay balanced
            # across the TP group. Every rank has identical seqs, so the result
            # is identical; only the output rank's IPC package is forwarded.
            self._compute_prompt_logprobs(self.input_data.seqs, hidden)
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
                #
                # ``src`` is a *global* rank. In DP+EP each DP group is its own
                # TP subgroup, so the group's output rank is its local tp_rank-0
                # (``get_rank() - get_tp_rank()``), not the world's output rank
                # 0 (which isn't even a member of group>0's TP subgroup).
                tp_src = (
                    get_rank() - get_tp_rank()
                    if is_dp_attn()
                    else get_output_rank()
                )
                dist.broadcast(
                    next_tokens_gpu,
                    src=tp_src,
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
                    # Stage this batch's logprobs into the same buf_idx slot so
                    # ``_collect_batch`` can read them once ``copy_done`` fires.
                    if lp_gpu is not None:
                        sampled, top_vals, top_ids = lp_gpu
                        self._lp_sampled_bufs[buf_idx][:batch_size].copy_(
                            sampled, non_blocking=True
                        )
                        if lp_k > 0:
                            self._lp_topval_bufs[buf_idx][
                                :batch_size, :lp_k
                            ].copy_(top_vals, non_blocking=True)
                            self._lp_topid_bufs[buf_idx][
                                :batch_size, :lp_k
                            ].copy_(top_ids, non_blocking=True)

            self.overlap_runtime.input_consumed_event.record(self.forward_stream)

        copy_done = torch.cuda.Event()
        if get_tp_size() > 1 or is_output_rank():
            copy_done.record(self.copy_stream)
        else:
            copy_done.record(self.forward_stream)
        return copy_done, batch_size, future_slot_ids, buf_idx, lp_k

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
