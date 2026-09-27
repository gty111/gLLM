import gc
import os
import time
from contextlib import nullcontext as _nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from attr import dataclass
from logger import logger
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

from gllm.distributed.parallel_state import (
    get_last_pp_rank,
    get_local_rank,
    get_next_pp_rank,
    get_output_rank,
    get_pp_size,
    get_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
    is_dp_attn,
    is_first_pp_rank,
    is_last_pp_rank,
    is_output_rank,
    recv_pp_data_async,
    recv_pp_tokens_from_last_stage,
    send_pp_data_async,
    send_pp_tokens_to_previous_stages,
)
from gllm.layers.attention.qkv_backends import (
    MLA_DECODE_BACKENDS,
    QKV_ATTENTION_BACKENDS,
    bind_qkv_attention_backend,
    create_qkv_attention_backend,
    find_qkv_attention_layers,
)
from gllm.layers.sampler import Sampler
from gllm.multimodal.mixin import (
    EmbeddingInfo,
    MmMixin,
    MultiModalEmbeddingCache,
)
# Re-exported for ``gllm.runtime.vision_encoder_runner`` and tests that import
# these helpers from their historical ``gllm.runtime.model_runner`` location.
from gllm.multimodal.mixin import (  # noqa: F401
    _build_item_content_hash as _build_item_content_hash,
    _concat_mrope_positions_pinned as _concat_mrope_positions_pinned,
)
from gllm.runtime.async_runtime import FutureIndices, FutureMap, OverlapRuntime
from gllm.runtime.config import EngineConfig
from gllm.runtime.cuda_graph import CudaGraphMixin
from gllm.runtime.forward_metadata import ForwardMetadataPlan
from gllm.runtime.input_data import InputData
from gllm.runtime.memory_manager import MemoryManager, PrefixMemoryManager
from gllm.runtime.model_loader import ModelLoader, propagate_serving_config
from gllm.runtime.piecewise_cuda_graph import PiecewiseGraphRunner
from gllm.runtime.sequence import GenerationSequence
from gllm.speculative.async_state import MtpAsyncBatchState
from gllm.speculative.gpu_prep import MtpGpuPrep
from gllm.speculative.mtp import MtpMixin, MtpQDist, MtpVerifyResult
from gllm.speculative.staging import MtpStagingBuffers
from gllm.tokenizers.mixin import TokenizerMixin


def apply_mm_processor_pixels(
    image_processor,
    video_processor,
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> None:
    """Apply mm processor min/max pixel bounds to the image/video processors.

    Shared by :class:`ModelRunner` and
    :class:`gllm.runtime.vision_encoder_runner.VisionEncoderRunner` so the
    monolith and the encoder-disaggregation paths resize identically. Sets
    both the ``*_pixels`` attributes and the ``size`` edge entries the HF
    processors actually consult.
    """
    if min_pixels is not None:
        image_processor.min_pixels = min_pixels
        video_processor.min_pixels = min_pixels
        image_processor.size["shortest_edge"] = min_pixels
        video_processor.size["shortest_edge"] = min_pixels
        logger.info(f"Min pixels: {min_pixels}")
    if max_pixels is not None:
        image_processor.max_pixels = max_pixels
        video_processor.max_pixels = max_pixels
        image_processor.size["longest_edge"] = max_pixels
        video_processor.size["longest_edge"] = max_pixels
        logger.info(f"Max pixels: {max_pixels}")


@dataclass
class DisaggSeqState:
    """Per-seq encoder-disaggregation overlap state.

    Owned by the :class:`ModelRunner` (keyed by ``seq_id``) so it is immune to
    the scheduler's chunked-prefill ``deepcopy`` of the :class:`GenerationSequence`. The
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
    item_span: List[Tuple[int, int]]  # ordered: (start, end) in tokens
    item_modality: List[str]
    item_ready: List[bool]
    item_embed: List[Optional[torch.Tensor]]  # ordered, filled on NIXL notif
    image_grid_thw: Optional[torch.Tensor]
    video_grid_thw: Optional[torch.Tensor]
    input_ids_cpu: torch.Tensor  # full expanded prompt ids (cpu)
    is_multimodal_cpu: torch.Tensor  # full mask (cpu)
    prompt_positions: torch.Tensor  # full-prompt mrope positions
    mrope_position_delta: torch.Tensor
    prompt_len: int


class ModelRunner(MtpMixin, MmMixin, TokenizerMixin, CudaGraphMixin):
    def __init__(self, config: EngineConfig):
        self.config = config

        self.max_num_batched_tokens = (
            config.maxp
            if config.schedule_method in ["chunked_prefill", "split_pd"]
            else config.maxp + config.maxd
        )

        # Concurrent decode slots (SSM arena entries, input buffers, CUDA
        # graph capture). Bounded by ``maxd`` for all schedule methods.
        self.max_running_seqs = config.maxd

        self.model_path = config.model_path
        # Encoder-disaggregation role flags ride in on the config's
        # ``DisaggConfig`` (None for the monolith).
        disagg_config = config.disagg_config
        skip_visual = disagg_config.skip_visual if disagg_config is not None else False
        skip_language = (
            disagg_config.skip_language if disagg_config is not None else False
        )
        self.model_loader = ModelLoader(
            config.load_format,
            config.model_path,
            self.max_num_batched_tokens,
            skip_visual=skip_visual,
            skip_language=skip_language,
        )
        self.enable_prefix_caching = config.enable_prefix_caching
        self.gpu_memory_util = config.gpu_memory_util
        self.page_size = config.page_size
        self._piecewise_cuda_graph_cfg = config.piecewise_cuda_graph
        self._max_piecewise_cuda_graph_tokens_cfg = config.max_piecewise_cuda_graph_tokens
        # Recurrent-state (GDN/Mamba) prefix-cache granularity, in tokens.
        # Only meaningful for hybrid models with prefix caching on.
        self.ssm_snapshot_stride_tokens = config.ssm_snapshot_stride_tokens
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
        architecture = getattr(self.model_loader, "architecture", None)
        self._deepseek_encoder_variant = {
            "DeepseekV32ForCausalLM": "dsv32",
            "DeepseekV4ForCausalLM": "dsv4",
        }.get(architecture)
        self._use_dsv32_encoder = self._deepseek_encoder_variant == "dsv32"
        self.maxp = config.maxp
        self.maxd = config.maxd
        self.minp = config.minp
        self.iterp = config.iterp
        # Adaptive KV-cache admission control (see Scheduler). ``init`` is the
        # starting/relaxed-ceiling fraction of remaining output we reserve for
        # running decodes; ``min`` is the floor the ratio decays toward when
        # the system is stable.
        self.init_new_token_ratio = config.init_new_token_ratio
        self.min_new_token_ratio = config.min_new_token_ratio
        self.schedule_method = config.schedule_method
        self.sampler = Sampler(self.tokenizer)
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
        self.attention_backend = (config.attention_backend or "flashinfer").lower()
        if self.attention_backend not in ("auto",) + QKV_ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be 'auto', 'fa4', 'flashinfer', or 'fa3', "
                f"got {self.attention_backend!r}."
            )
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

        # Backend names are syntax-checked here in the parent process. Their
        # hardware/import compatibility is resolved once per worker by
        # ``verify_config`` after that worker selects its CUDA device.
        self.mla_decode_backend = (config.mla_decode_backend or "fa4").lower()
        if self.mla_decode_backend not in MLA_DECODE_BACKENDS:
            raise ValueError(
                "mla_decode_backend must be 'fa4', 'flashmla', or 'triton', "
                f"got {self.mla_decode_backend!r}."
            )
        # MLA latent KV cache precision (DeepSeek Sparse Attention). "bf16"
        # (default) = full-precision latent cache + dense decode; "fp8" = native
        # FP8-packed cache driving FlashMLA sparse decode on SM90.
        self.mla_cache_dtype = (config.mla_cache_dtype or "bf16").lower()
        if self.mla_cache_dtype not in ("bf16", "fp8"):
            raise ValueError(
                f"mla_cache_dtype must be 'bf16' or 'fp8', got {self.mla_cache_dtype!r}."
            )
        # Recurrent-state cache precision for hybrid linear-attention models.
        # "auto" honours the checkpoint's ``mamba_ssm_dtype`` recommendation
        # when present (Qwen3.5 requires float32), otherwise it falls back to
        # the activation dtype. Explicit CLI values override that recommendation.
        self.mamba_ssm_cache_dtype = (config.mamba_ssm_cache_dtype or "auto").lower()
        if self.mamba_ssm_cache_dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(
                "mamba_ssm_cache_dtype must be 'auto', 'bfloat16', 'float16' or "
                f"'float32', got {self.mamba_ssm_cache_dtype!r}."
            )
        self.model_loader.config.mamba_ssm_cache_dtype = self.mamba_ssm_cache_dtype
        # Stamp the resolved preference + final page size onto the model config
        # so ``MLAAttention`` can pick them up at construction time.
        self.model_loader.config.mla_decode_backend = (
            self.mla_decode_backend if self.use_mla else None
        )
        self.model_loader.config.attention_backend = self.attention_backend
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
        if (
            config.mtp_enabled is None
            and self.model_loader.architecture == "DeepseekV4ForCausalLM"
        ):
            # V4's ``mtp.0/1/2`` are a joint noisy-block DSpark model with a
            # Markov correction and confidence head. They are not compatible
            # with gLLM's sequential one-token NextN protocol, so auto mode
            # keeps the verified base model lean. Explicit ``--mtp-enabled on``
            # still constructs the DSpark numerical/reference module.
            resolved_mtp_enabled = False
        else:
            resolved_mtp_enabled = (
                (_num_nextn >= 1) if config.mtp_enabled is None else bool(config.mtp_enabled)
            )
        self._mtp_k_cfg = config.mtp_k
        self._mtp_max_batch_cfg = config.mtp_max_batch
        self.model_loader.config.mtp_enabled = resolved_mtp_enabled
        # Nested text config (Qwen3.5-VL wrapper) reads ``mtp_enabled`` off its
        # own config object, so mirror the flag there too.
        if _text_cfg is not _cfg:
            _text_cfg.mtp_enabled = resolved_mtp_enabled
        propagate_serving_config(self.model_loader.config)

        # Kimi-K2.5 ships a bespoke processor (``KimiK25Processor``) whose API
        # and outputs diverge from the Qwen-VL ``AutoProcessor`` contract:
        # output keys are ``pixel_values``/``grid_thws`` (not
        # ``image_grid_thw``), no separate ``image_processor``/
        # ``video_processor`` split, and the chat template emits a single
        # ``<|media_pad|>`` per image that must be expanded downstream.
        if self.use_mm and self.is_kimi_mm:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=True
            )
            self.image_processor = None
            self.video_processor = None
        elif self.use_mm:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, use_fast=True
            )
            self.image_processor = self.processor.image_processor
            self.video_processor = self.processor.video_processor
            apply_mm_processor_pixels(
                self.image_processor,
                self.video_processor,
                min_pixels=config.mm_processor_min_pixels,
                max_pixels=config.mm_processor_max_pixels,
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

        # Encoder-disaggregation overlap: seq_id => per-item
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
        self.mm_embed_cache = MultiModalEmbeddingCache(max_entries=64, max_mb=256.0)

        # cuda graph
        self.disable_cuda_graph = config.disable_cuda_graph
        # ``max_cuda_graph_bs`` cannot exceed *either* of two runtime bounds:
        #
        #   * ``maxd`` — the decode batch is hard-capped at ``maxd`` (scheduler)
        #     and several device buffers (``InputData.block_table``,
        #     ``slot_mapping``, SSM metadata, ...) are sized at ``maxd``
        #     rows.
        #   * ``max_num_batched_tokens`` — a captured decode graph of ``B``
        #     sequences writes ``B`` token-rows into the shared activation
        #     buffers (``input_hidden_states`` / ``residual`` / PP recv), which
        #     are sized to ``max_num_batched_tokens``. Under ``chunked_prefill``
        #     / ``split_pd`` that equals ``maxp``, so a small ``--maxp`` would
        #     overflow those buffers *during capture*.
        #
        # A real forward never batches more than ``max_num_batched_tokens``
        # tokens (decode eats into the same per-tick budget as prefill), so
        # buckets above that bound are never replayed anyway. Clamp to the
        # tighter of the two so users can keep the ``--max-cuda-graph-bs``
        # default without manually matching ``maxd`` / ``maxp``.
        cuda_graph_cap = min(config.maxd, self.max_num_batched_tokens)
        max_cuda_graph_bs = config.max_cuda_graph_bs
        if not self.disable_cuda_graph and max_cuda_graph_bs > cuda_graph_cap:
            logger.warning(
                f"max_cuda_graph_bs={max_cuda_graph_bs} exceeds the runtime "
                f"decode-batch bound min(maxd={config.maxd}, "
                f"max_num_batched_tokens={self.max_num_batched_tokens})="
                f"{cuda_graph_cap}; clamping to {cuda_graph_cap}."
            )
            max_cuda_graph_bs = cuda_graph_cap
        self.max_cuda_graph_bs = max_cuda_graph_bs
        self.size_to_graph: Dict[int, torch.cuda.CUDAGraph] = dict()
        # Use power-of-two bucket sizes to reduce the number of captured graphs.
        # At runtime the actual batch is padded up to the nearest bucket.
        self.capture_sizes = self._build_capture_sizes(self.max_cuda_graph_bs)

        self.model_max_length = self.resolve_model_max_length(config.model_max_length)
        # Models size static tables and CUDA-graph-safe static bounds from the
        # config (RoPE tables, worst-case candidate counts). The serving length
        # is the *resolved* runtime limit, which is routinely orders of
        # magnitude below the checkpoint's advertised
        # ``max_position_embeddings`` -- DeepSeek-V4 advertises 1M. Publish it
        # on the config so a model never has to guess.
        self.model_loader.config.model_max_length = self.model_max_length
        _text_config = getattr(self.model_loader.config, "text_config", None)
        if _text_config is not None:
            _text_config.model_max_length = self.model_max_length

        # ``InputData``'s per-token buffers are sized ``model_max_length`` (the
        # longest single sequence), but a prefill batch may carry
        # ``max_num_batched_tokens`` (= ``maxp``) tokens. With ``maxp >
        # model_max_length`` the profile run's full-size dummy prefill would
        # overflow those buffers and die deep inside ``copy_to_input_buffer``
        # with a bare shape mismatch. Clamp + say so instead: a prefill batch
        # can never usefully exceed one sequence's max length, since chunked
        # prefill already splits longer prompts.
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

    def verify_config(self) -> None:
        """Resolve hardware-dependent backend preferences once per worker."""
        requested = self.attention_backend
        capability = torch.cuda.get_device_capability()

        # Instantiate candidates and launch the FlashInfer/FA3 kernels.
        # Some wheels import successfully but contain no cubin for the current
        # GPU; only a real launch detects that condition.
        preference = {
            "auto": (
                ("fa3", "fa4", "flashinfer")
                if capability[0] == 8
                else ("fa4", "flashinfer", "fa3")
            ),
            "fa4": ("fa4", "flashinfer", "fa3"),
            "flashinfer": ("flashinfer", "fa4", "fa3"),
            "fa3": ("fa3", "fa4", "flashinfer"),
        }[requested]
        backend_errors = {}
        validated_backend = None
        resolved = None
        for candidate in preference:
            if candidate == "flashinfer" and capability[0] == 8:
                backend_errors[candidate] = "TRT-LLM/XQA paged KV is unsupported on SM8x"
                continue
            if candidate == "fa3" and capability[0] not in (8, 9):
                backend_errors[candidate] = "SGL kernel FlashAttention-3 requires SM8x or SM90"
                continue
            if candidate == "fa4" and capability[0] not in (9, 10, 11):
                backend_errors[candidate] = (
                    f"paged KV is unsupported on SM{capability[0]}{capability[1]}"
                )
                continue
            candidate_backend = None
            try:
                candidate_backend = create_qkv_attention_backend(
                    candidate,
                    self.model_max_length,
                    self.max_running_seqs,
                )
                if candidate in ("flashinfer", "fa3"):
                    candidate_backend.smoke_test(self.page_size)
            except Exception as exc:  # noqa: BLE001 - backend probe boundary
                backend_errors[candidate] = str(exc)
                del candidate_backend
                gc.collect()
                torch.cuda.empty_cache()
                continue
            validated_backend = candidate_backend
            resolved = candidate
            break

        if validated_backend is None or resolved is None:
            details = "; ".join(
                f"{name}: {backend_errors.get(name, 'not attempted')}"
                for name in preference
            )
            raise RuntimeError(
                "No runnable paged-QKV attention backend was found "
                f"(requested {requested!r}; {details})"
            )

        self._validated_qkv_attention_backend = validated_backend
        if requested == "auto":
            logger.info(
                "Attention backend 'auto' resolved to %r during startup "
                "kernel validation.",
                resolved,
            )
        elif resolved != requested:
            logger.warning(
                "Attention backend %r failed startup kernel validation (%s); "
                "falling back to %r.",
                requested,
                backend_errors.get(requested),
                resolved,
            )

        self.attention_backend = resolved
        self.model_loader.config.attention_backend = resolved

        if self.use_mla:
            requested_mla = self.mla_decode_backend
            resolved_mla = requested_mla
            mla_error = None
            if requested_mla == "fa4":
                if capability[0] not in (10, 11):
                    mla_error = (
                        "absorbed MLA is unsupported on "
                        f"SM{capability[0]}{capability[1]}"
                    )
                else:
                    try:
                        from flash_attn.cute import flash_attn_varlen_func  # noqa: F401
                    except Exception as exc:
                        mla_error = str(exc)
                if mla_error is not None:
                    resolved_mla = "triton"
            elif requested_mla == "flashmla":
                try:
                    from sgl_kernel.flash_mla import (  # noqa: F401
                        flash_mla_with_kvcache,
                        get_mla_metadata,
                    )
                except Exception as exc:
                    mla_error = str(exc)
                    resolved_mla = "triton"
                else:
                    flashmla_page_size = 64
                    if self.page_size != flashmla_page_size:
                        logger.info(
                            "MLA FlashMLA decode backend requires page_size=%d; "
                            "overriding page_size %d -> %d.",
                            flashmla_page_size,
                            self.page_size,
                            flashmla_page_size,
                        )
                        self.page_size = flashmla_page_size

            if resolved_mla != requested_mla:
                logger.warning(
                    "MLA decode backend %r is unavailable (%s); resolved "
                    "startup configuration to %r.",
                    requested_mla,
                    mla_error,
                    resolved_mla,
                )
            self.mla_decode_backend = resolved_mla
            self.model_loader.config.mla_decode_backend = resolved_mla

        self.model_loader.config.page_size = self.page_size
        propagate_serving_config(self.model_loader.config)
        logger.info(
            "Verified attention backend: %s (requested %s, compute capability "
            "SM%d%d)",
            resolved,
            requested,
            capability[0],
            capability[1],
        )

    # Read-only facades over private runner state. Workers / the scheduler
    # consume these; the underscored attributes remain the writer-side
    # representation. All are populated in ``init`` (or ``__init__`` where
    # noted) and read only afterwards.

    @property
    def last_logprobs(self):
        """Per-batch-row generation logprobs from the latest ``step_once``."""
        return self._last_logprobs

    @property
    def last_prompt_logprobs(self):
        """Prompt logprobs that finished prefill on the latest forward."""
        return self._last_prompt_logprobs

    @property
    def mtp_k(self) -> int:
        """Draft-chain length of the MTP head (0 when MTP is unavailable)."""
        return self._mtp_k

    @property
    def mtp_max_batch(self) -> int:
        """Batch-size performance gate for MTP; 0 disables the gate."""
        return self._mtp_max_batch

    def init(self, mp_load_progress=None):
        self.verify_config()
        self.model = self.model_loader.load_model(mp_load_progress)
        # Models may opt out of monolithic decode graphs while individual
        # components are still being made graph-safe.  Keep this capability on
        # the model rather than architecture-switching in the runner: new model
        # adapters get the common graph path by default, and an adapter can
        # remove the opt-out once capture *and real-data replay* are verified.
        # ``--disable-cuda-graph`` remains the user-facing global override.
        self._full_cuda_graph_on = bool(
            not self.disable_cuda_graph
            and getattr(self.model, "supports_full_cuda_graph", True)
        )
        if not self.disable_cuda_graph and not self._full_cuda_graph_on:
            logger.warning(
                "%s does not yet support decode FULL CUDA graph replay; "
                "FULL graphs are disabled (piecewise graphs remain available).",
                type(self.model).__name__,
            )
        # MTP speculative decoding: number of draft tokens per step (k). Active
        # only when the model built an MTP head (mtp_enabled + nextn layers).
        self._mtp_k = (
            self._mtp_k_cfg if getattr(self.model, "mtp", None) is not None else 0
        )
        # The one authoritative runtime capability flag. The earlier resolved
        # preference only controls whether the loader constructs the MTP head;
        # after loading, availability additionally requires a positive k.
        self.mtp_enabled = bool(
            self._mtp_k > 0 and getattr(self.model, "mtp", None) is not None
        )
        # Hybrid models (Qwen3.5 GDN) advertise a ready-to-use SSM cache
        # config via ``model.ssm_cache_config``. ``num_layers`` for the KV
        # path must then be the count of *full-attention* layers only.
        ssm_cache_config = getattr(self.model, "ssm_cache_config", None)
        dsv4_state_cache_config = getattr(
            self.model, "dsv4_state_cache_config", None
        )
        dsv4_kv_cache_config = getattr(
            self.model, "dsv4_kv_cache_config", None
        )
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
            dsv4_state_cache_config=dsv4_state_cache_config,
            dsv4_kv_cache_config=dsv4_kv_cache_config,
            max_running_seqs=self.max_running_seqs,
            # DeepSeek Sparse Attention (V3.2): non-zero => allocate a parallel
            # paged indexer key cache. 0 for every other model.
            index_head_dim=getattr(self.model, "index_head_dim", 0),
            # MLA rope head dim (needed to size the native FP8 MLA cache for DSA).
            qk_rope_head_dim=getattr(self.model, "qk_rope_head_dim", 0),
            # DSA MLA latent cache precision: FP8-packed only when explicitly
            # requested (drives SM90 sparse decode); default bf16 + dense decode.
            mla_cache_fp8=(self.mla_cache_dtype == "fp8"),
            # Recurrent-state prefix-cache granularity (tokens). Rounded to
            # whole pages by ``PrefixMemoryManager.init``.
            ssm_snapshot_stride_tokens=self.ssm_snapshot_stride_tokens,
            # MTP draft-chain length for hybrid GDN models: each running seq may
            # claim 1+mtp_k working/checkpoint entries from the cache arena.
            # 0 for non-MTP or non-hybrid.
            mtp_k=(
                self._mtp_k
                if (ssm_cache_config is not None and self.mtp_enabled)
                else 0
            ),
        )
        self.input_data = InputData(
            max_running_seqs=self.max_running_seqs,
            max_seq_length=self.model_max_length,
            memory_manager=self.memory_manager,
            use_buffer=True,
        )
        # Detect actual QKV MHA/GQA layers instead of treating MLA and QKV
        # attention as model-wide mutually-exclusive modes. The runner owns the
        # shared backend; each QKV attention layer gets an
        # explicit reference, while InputData carries only per-forward metadata.
        self.qkv_attention_backend = None
        qkv_attention_layers = find_qkv_attention_layers(self.model)
        if qkv_attention_layers:
            self.qkv_attention_backend = getattr(
                self, "_validated_qkv_attention_backend", None
            )
            if self.qkv_attention_backend is None:
                self.qkv_attention_backend = create_qkv_attention_backend(
                    self.attention_backend,
                    self.model_max_length,
                    self.max_running_seqs,
                )
            num_bound_attention_layers = bind_qkv_attention_backend(
                qkv_attention_layers, self.qkv_attention_backend
            )
            logger.info(
                "Bound QKV attention backend %s to %d MHA/GQA/MQA layers",
                self.qkv_attention_backend.name,
                num_bound_attention_layers,
            )
        device = torch.device("cuda", torch.cuda.current_device())
        self.input_hidden_states = torch.zeros(
            (self.max_num_batched_tokens, self.hidden_size), device=device
        )
        self.input_residual = torch.zeros(
            (self.max_num_batched_tokens, self.hidden_size), device=device
        )
        self.output_hidden_states = torch.zeros(
            (self.max_num_batched_tokens, self.hidden_size), device=device
        )
        self.output_residual = torch.zeros(
            (self.max_num_batched_tokens, self.hidden_size), device=device
        )
        # MTP draft-step CUDA-graph buffers. A draft step is a batch x 1-token
        # decode of the MTP head; we capture one graph per decode bucket and
        # replay it k times, advancing tok/hidden/positions/seq_lens/slot in
        # place on the GPU (no Python / H2D / .item() per step). Gated on MTP
        # active + graphs enabled; DP-attention is excluded until its phase
        # counts are coordinated. Buffers + the aliasing
        # ``_draft_input`` are lazily built in ``_init_draft_graph_state`` after
        # the model + memory manager exist.
        self._mtp_draft_graph = (
            self.mtp_enabled
            and not is_dp_attn()
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
        self._draft_penalty_graphs = {mode: {} for mode in ("greedy", "dense", "sparse")}
        self._capture_draft_penalty = False
        self._mtp_penalties = None
        self._draft_input = None
        # MTP verify CUDA graph: capture the full target verify forward (over the
        # uniform 1+k query per decode seq) per bucket at init and replay it. The
        # verify forward is 99% of MTP step time and is pure eager per-layer launch
        # overhead (~250ms, ~constant vs batch size), so graphing it is the main
        # speedup lever. Requires the fp8 decode-sparse verify kernel (graph-safe).
        self._mtp_verify_graph = (
            self.mtp_enabled
            and not is_dp_attn()
            and not self.disable_cuda_graph
            # Works for MLA (DeepSeek fp8 decode-sparse kernel) and non-MLA
            # (Qwen3.5 GDN): the verify forward reads only static input buffers
            # (incl. the 2D SSM block table + num_accepted static buffers filled
            # by copy_to_input_buffer), so it is graph-capturable for both.
        )
        self._verify_size_to_graph: Dict[int, torch.cuda.CUDAGraph] = {}
        self._verify_k = getattr(self, "_mtp_k", 0)
        # Fused MTP is the only MTP execution mode: eliminate the separate
        # x1-decode forward by relaying each step's verify bonus token + its
        # hidden as the NEXT step's draft seed. One target forward (verify) per
        # steady-state step instead of two. ``_mtp_relay`` maps seq_id ->
        # (seed_tok:int, seed_hidden:tensor[H]) across consecutive steps; a seq
        # missing from it (fresh admission / batch reshuffle) is seeded by one
        # padded verify-shaped bootstrap before joining the fused steady state.
        # Piecewise graphs split a dynamic forward at model-declared
        # Attention/SSM boundaries. ``auto`` (None) preserves the historical
        # behavior by enabling them only for MTP mixed forwards; explicit True
        # promotes the same runner to ordinary prefill and mixed batches.
        # PP and DP need a coordinated cross-rank segment protocol, so they
        # remain eager until that protocol exists.
        piecewise_requested = (
            self.mtp_enabled
            if self._piecewise_cuda_graph_cfg is None
            else bool(self._piecewise_cuda_graph_cfg)
        )
        piecewise_model_supported = any(
            getattr(module, "supports_piecewise_cuda_graph", False)
            for module in self.model.modules()
        )
        self._piecewise_cuda_graph_on = (
            piecewise_requested
            and piecewise_model_supported
            and not self.disable_cuda_graph
            and not is_dp_attn()
            and is_first_pp_rank()
            and is_last_pp_rank()
        )
        self._piecewise_generic_on = (
            self._piecewise_cuda_graph_on and self._piecewise_cuda_graph_cfg is True
        )
        if (
            piecewise_requested
            and not self.disable_cuda_graph
            and not piecewise_model_supported
        ):
            logger.warning(
                "Piecewise CUDA graph requested, but model %s declares no "
                "dynamic Attention/SSM boundaries; using eager prefill/mixed "
                "forwards.",
                type(self.model).__name__,
            )
        from gllm.layers.ops.fla._sgl_compat import set_piecewise_cuda_graph_enabled

        set_piecewise_cuda_graph_enabled(self._piecewise_cuda_graph_on)
        self._piecewise_runner = None
        if self._piecewise_cuda_graph_on:
            piecewise_capture_limit = self.max_num_batched_tokens
            if self._max_piecewise_cuda_graph_tokens_cfg is not None:
                configured_limit = int(self._max_piecewise_cuda_graph_tokens_cfg)
                if configured_limit <= 0:
                    raise ValueError(
                        "max_piecewise_cuda_graph_tokens must be positive, got "
                        f"{configured_limit}"
                    )
                piecewise_capture_limit = min(piecewise_capture_limit, configured_limit)
            piecewise_capture_sizes = PiecewiseGraphRunner.build_capture_sizes(
                piecewise_capture_limit
            )
            self._piecewise_runner = PiecewiseGraphRunner(
                self.model,
                capture_sizes=piecewise_capture_sizes,
            )
            logger.info(
                "Piecewise CUDA graph enabled (mode=%s, max_tokens=%d; "
                "attention/GDN eager breaks, graph-resident MoE, "
                "bucket_sizes=%s)",
                ("auto-mtp" if self._piecewise_cuda_graph_cfg is None else "generic"),
                piecewise_capture_limit,
                piecewise_capture_sizes,
            )
        self._mtp_relay: Dict[int, tuple] = {}
        # Set only while OverlapWorker launches a greedy MTP step.  The accept
        # path publishes its fixed-width result into this GPU/pinned-host ring
        # instead of synchronizing on ``Tensor.cpu()`` in the launch call.
        self._mtp_async_state: Optional[MtpAsyncBatchState] = None
        self._mtp_async_publish = False
        # One-shot warning when a multimodal prompt has to skip the head KV
        # pass (image placeholder ids do not embed to the prompt's real
        # features, so replaying the head over them would poison its cache).
        self._mtp_kv_sync_mm_warned = False
        # Batch-adaptive MTP gate. Speculating multiplies the per-step target
        # work by ``1+k``; it wins only
        # while the decode batch leaves the GPU under-utilized. Past the crossover
        # a plain 1-token step is strictly faster, so skip MTP for that step.
        # Returning to MTP invalidates the relay and takes one padded bootstrap.
        # ``0`` disables the performance gate (always speculate when capacity and
        # execution mode permit it).
        self._mtp_max_batch = int(getattr(self, "_mtp_max_batch_cfg", 0) or 0)
        self._mtp_spec_decision = None
        if self.mtp_enabled and is_dp_attn() and get_local_rank() == 0:
            logger.warning(
                "MTP speculative decoding is disabled under DP-attention: "
                "bootstrap/draft/verify collective shapes are not yet "
                "coordinated across DP replicas; using plain decode."
            )
        if self.mtp_enabled and self._mtp_max_batch > 0:
            logger.info(
                f"MTP batch gate: speculating only while the decode batch is "
                f"<= {self._mtp_max_batch} seqs; larger batches take a plain "
                f"decode step."
            )
        # GPU-native MTP input prep (see ``gllm/speculative/gpu_prep.py``). It replaces
        # the per-step Python rebuild of the
        # draft / verify input arrays with persistent pinned staging + a few
        # vectorized CUDA ops writing straight into the static graph buffers.
        # ``GLLM_MTP_GPUPREP=0`` falls back to the CPU builders (``cal_input``).
        self._mtp_gpu_prep = None
        self._mtp_gpu_prep_on = (
            self.mtp_enabled and os.environ.get("GLLM_MTP_GPUPREP", "1") == "1"
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
                bt_width=(1 + self._mtp_k if self.memory_manager.use_ssm_cache else 0),
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
        self._mtp_can_sample = self.mtp_enabled
        self._mtp_rng = None  # lazily built on the compute device
        self._mtp_step = 0
        # Device-side counter of sparse-top-k tie overflows (see
        # ``_mtp_sparse_probs``); read + reported at the 1 Hz metrics log so the
        # hot path never syncs on it.
        self._mtp_tie_overflow = torch.zeros((), dtype=torch.int64, device="cuda")
        # The overflow counter is relevant only to rejection sampling. Greedy
        # MTP must never read this CUDA scalar from the periodic metrics path:
        # ``int(cuda_tensor)`` synchronizes the whole producer stream.
        self._mtp_sampling_seen = False
        # Persistent staging for the ragged prefill suffix of a mixed MTP
        # target. These replace per-forward ``torch.as_tensor(list,
        # device="cuda")`` calls, whose pageable H2D copies introduce a hidden
        # cudaStreamSynchronize even for one-element lists.
        mixed_cap = max(self.max_running_seqs, 1)
        mixed_device = torch.device("cuda", torch.cuda.current_device())
        # Persistent pinned/device staging avoids per-step allocation and
        # pageable transfers across draft, verify, and head-KV maintenance.
        self._mtp_staging = MtpStagingBuffers(
            capacity=mixed_cap,
            max_tokens=self.max_num_batched_tokens,
            hidden_size=self.hidden_size,
            hidden_dtype=self.output_hidden_states.dtype,
            device=mixed_device,
        )
        # Bumped once per ``_mtp_decode`` so the GPU prep can memoize its
        # per-step staging across the draft and verify phases.
        self._mtp_prep_epoch = 0
        self.profile_run()
        # Init KV cache at last; only reserve the dummy page when CUDA graphs
        # are actually enabled so we don't waste memory otherwise.
        self.memory_manager.init(reserve_dummy_page=self._full_cuda_graph_on)
        self.memory_manager.validate_model_max_length(
            self.model_max_length,
            mtp_lookahead=self._mtp_k if self.mtp_enabled else 0,
        )

        if self._full_cuda_graph_on or self._piecewise_runner is not None:
            self.capture_graph()

    def prepare_input_embeddings(self, hidden_states=None):
        if hidden_states is not None:
            assert is_first_pp_rank()
            self.input_hidden_states[: hidden_states.shape[0]] = hidden_states
            self.input_data.embedding_size = hidden_states.shape[0]

    def prepare_input(
        self, seqs: List[GenerationSequence] = None, input_data: InputData = None
    ):
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
        seqs = [
            GenerationSequence(idx, [1, 2], [], output_len=1) for idx in range(size)
        ]
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
        ``MemoryManager.init`` afterwards -- see ``QKVAttention.forward`` /
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
            seq = GenerationSequence(idx, [1] * length, [], output_len=1)
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
        stream_ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with stream_ctx:
            self._prepare_attention_metadata(self.input_data)
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
    def forward(self):
        self._prepare_attention_metadata(self.input_data)
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        if is_first_pp_rank() and self.use_mm:
            # See ``_fixup_vl_decode_embeddings`` for why this is required
            # without overlap scheduling.
            num_decode_tokens = sum(
                s.to_compute_token_num
                for s in self.input_data.seqs
                if s.computed_prompt
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

    def _prepare_attention_metadata(self, input_data: InputData) -> None:
        """Prepare backend metadata through the current forward-level plan."""
        backend = self.qkv_attention_backend
        if (
            backend is not None
            and getattr(self.memory_manager, "segment", None) is not None
        ):
            plan = input_data.forward_metadata_plan
            if plan is None:
                raise RuntimeError("model forward has no ForwardMetadataPlan")
            plan.prepare_attention(backend, input_data)

    def check_decode_batch(self):
        # Since the scheduler put prefill seqs at the end
        # we only check the last seq
        return self.input_data.seqs[-1].computed_prompt

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
        ``token_ids`` + ``raw_prompt_len``: works on the real ``GenerationSequence``
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
        identical seqs (real ``GenerationSequence`` for PP=1, identical ``FollowerSeq``
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

    @torch.inference_mode()
    def step_once(self, dp_padded_size: Optional[int] = None):
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        # A mixed prefill+decode batch deliberately uses the ordinary forward
        # path below: speculative draft/verify is only legal for pure decode.
        # Any decode row in this mixed batch nevertheless advances by one token,
        # invalidating the bonus token + hidden relayed by its previous MTP
        # verify. Drop exactly those rows before the plain forward; otherwise a
        # following pure-decode iteration can mistake the stale entry for a full
        # relay hit and seed its draft from the wrong sequence position.
        if self.input_data.num_prefills > 0 and self.input_data.num_decodes > 0:
            self._mtp_drop_relay(self.input_data.seqs[: self.input_data.num_decodes])
        # Fused MTP path. Full-relay batches go straight to draft/verify. Fresh
        # or relay-miss seqs are seeded by a verify-shaped padded bootstrap so
        # every MTP target forward keeps the same fixed qlen=1+k; the ordinary
        # batch-x-1 decode path is never used for an enabled MTP step.
        if (
            self.mtp_enabled
            and dp_padded_size is None
            and not is_dp_attn()
            and is_last_pp_rank()
            and self.input_data.num_prefills == 0
            and self.input_data.num_decodes > 0
            and self.mtp_speculate_batch(self.input_data.num_decodes)
            and self.check_decode_batch()
            and self.mtp_sampling_compatible(self.input_data.seqs)
        ):
            seqs = self.input_data.seqs[: self.input_data.num_decodes]
            if seqs:
                async_state = self._mtp_async_state
                if (
                    self._mtp_async_publish
                    and async_state is not None
                    and async_state.matches([s.seq_id for s in seqs])
                ):
                    # True chained MTP: predecessor acceptance, relay token,
                    # hidden state and real context length stay on the GPU.
                    x1 = async_state.relay_tokens[: len(seqs)]
                    hidden = async_state.relay_hidden[: len(seqs)]
                elif all(s.seq_id in self._mtp_relay for s in seqs):
                    relay = [self._mtp_relay[s.seq_id] for s in seqs]
                    x1 = [r[0] for r in relay]
                    hidden = torch.stack([r[1] for r in relay], dim=0)
                else:
                    hidden, x1 = self._mtp_bootstrap_padded_verify(seqs)
                # Fused bypasses the sampler/logprobs block below; the relayed
                # or bootstrapped x1 is committed by ``_mtp_decode``. The next
                # bonus is relayed, not committed, so it cannot be double-emitted.
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
                if not self._run_generic_piecewise_forward(num_cal_tokens):
                    self.forward()
        else:
            if not self._run_generic_piecewise_forward(num_cal_tokens):
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
            elif any(getattr(s, "structured_output", None) is not None for s in seqs):
                next_tokens_gpu = self.sampler.forward_gpu(logits, self.input_data)
                next_tokens = None
            else:
                next_tokens = self.sampler.forward(logits, self.input_data)
            if any(getattr(s, "structured_output", None) is not None for s in seqs):
                # Matchers on all sampling ranks must consume the same token.
                # Preserve the tensor identity retained by StructuredSampler.
                if get_tp_size() > 1:
                    src = get_rank() - get_tp_rank() if is_dp_attn() else get_output_rank()
                    dist.broadcast(next_tokens_gpu, src=src, group=get_tp_group())
                self.sampler.stage_structured_feedback(next_tokens_gpu, seqs)
                next_tokens = next_tokens_gpu.cpu().tolist()
            # Prompt logprobs re-enter the LM head (a TP all-gather), so this
            # runs on ALL last-PP TP ranks (not just the output rank) to keep
            # the collective balanced; every rank has the same real seqs, so
            # they compute identical data and only tp0's copy is shipped.
            self._compute_prompt_logprobs(seqs, hidden)
            # Keep the MTP head's KV layer in lockstep with the target for
            # every batch that does NOT go through the speculative path
            # (prefill, chunked prefill, mixed, and plain decode).  Without
            # this the head attends over recycled pages it never wrote.
            self._mtp_sync_kv(
                self.input_data,
                hidden,
                tail_seqs=self.input_data.seqs,
                tail_next_tokens=next_tokens,
            )
            # MTP speculative decoding: on a pure-decode batch, draft k tokens
            # per seq with the MTP head and verify them with one base forward,
            # committing the accepted prefix. Returns per-seq token LISTS.
            if (
                self.mtp_enabled
                and self.input_data.num_prefills == 0
                and self.input_data.num_decodes > 0
                and (not self.mtp_speculate_batch(self.input_data.num_decodes)
                     or not self.mtp_sampling_compatible(seqs))
            ):
                # Constraints require plain sampling, or this batch is too large
                # to profit from speculation. Its one sampled token is final.
                # The relay it leaves behind is stale (see ``_mtp_drop_relay``).
                self._mtp_drop_relay()
            elif (
                self.mtp_enabled
                and self.input_data.num_prefills == 0
                and self.input_data.num_decodes > 0
            ):
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

    def disagg_prefill_limit(self, seq: GenerationSequence) -> Optional[int]:
        """Gate-B upper bound: the largest token position this
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

    def register_decode_page_hash(self, seq: GenerationSequence, pos: int) -> None:
        """Register the prefix-cache page hash for a decode boundary the seq
        just completed with a *real* (finalized) token at ``seq.token_ids[pos]``.

        Called from the scheduler's output-finalization hooks
        (``Scheduler.process_output`` after appending the real token, and
        ``OverlapScheduler.process_output_finalize`` after overwriting the
        placeholder). Keeping the trigger here -- rather than inside
        ``MemoryManager.pre_allocate_page`` -- guarantees the hash is only ever
        computed over real tokens, never an unfinalized overlap placeholder.
        No-op for caches without prefix support.
        """
        self.memory_manager.register_decode_boundary(seq, pos)

    def free(self, seq: GenerationSequence):
        # A relay hidden owns a full hidden-size GPU tensor.  Drop it immediately
        # when the request finishes instead of waiting for another MTP iteration
        # to replace the relay map (which may never happen when the batch ends).
        self._mtp_relay.pop(seq.seq_id, None)
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

        Followers *do* need to release the ``embedding_cache`` row on free
        (VL only, first PP rank only) -- otherwise each finished VL request
        leaks a multimodal-embedding tensor.
        """
        self._mtp_relay.pop(seq_id, None)
        if self.use_mm and is_first_pp_rank():
            self.embedding_cache.pop(seq_id, None)
            self.disagg_embeds.pop(seq_id, None)


class OverlapModelRunner(ModelRunner):
    """ModelRunner with FutureMap-based overlap scheduling across TP and PP."""

    def init(self, mp_load_progress=None):
        # Create the overlap CUDA streams BEFORE ``super().init()`` so that
        # ``capture_graph`` (invoked from inside ``super().init()``) can use
        # ``forward_stream`` as the capture stream. Capture stream must equal
        # replay stream: a mismatch makes the NCCL kernels baked into the
        # graph drift TP ranks out of lockstep over many decode steps.
        device = torch.device(f"cuda:{get_local_rank()}")
        self.overlap_runtime = OverlapRuntime(device)
        self.forward_stream = self.overlap_runtime.forward_stream
        self.copy_stream = self.overlap_runtime.copy_stream
        self.feedback_stream = self.overlap_runtime.feedback_stream
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
        # Capture stream must equal replay stream (see ``init``), or NCCL
        # kernels baked into the graph drift TP ranks out of lockstep.
        super().capture_graph(stream=self.forward_stream)

    # Read-only facades over the overlap-only state built by
    # ``_init_overlap_buffers`` during ``init``; consumed by OverlapWorker.

    @property
    def overlap_depth(self) -> int:
        """How many launched batches may stay un-retired."""
        return self._overlap_depth

    @property
    def num_output_bufs(self) -> int:
        """Pinned staging slots for sampled tokens / logprobs."""
        return self._num_output_bufs

    @property
    def next_tokens_bufs(self):
        """Pinned CPU staging slots for sampled tokens, keyed by buf idx."""
        return self._next_tokens_bufs

    @property
    def lp_sampled_bufs(self):
        return self._lp_sampled_bufs

    @property
    def lp_topval_bufs(self):
        return self._lp_topval_bufs

    @property
    def lp_topid_bufs(self):
        return self._lp_topid_bufs

    @property
    def pending_mm_ctx(self):
        """mm prep context carried between the CPU and GPU input-prep phases."""
        return self._pending_mm_ctx

    def _init_overlap_buffers(self, num_prefill_chunks: int = 256) -> None:
        device = self.forward_stream.device
        self.future_map = FutureMap(
            max_running_requests=self.max_running_seqs,
            context_len=self.model_max_length,
            chunked_prefill_size=num_prefill_chunks,
            device=device,
        )
        # How many launched batches the overlap worker leaves un-retired.
        #
        # Under PP the pipeline stalls once per turn-over of the in-flight set
        # -- measured as a bubble of ~1.7 ms recurring with a period of exactly
        # ``depth + 1`` launches, against ~0.44 ms for every other launch --
        # so a deeper queue amortizes that fixed stall over more batches. The
        # cost is latency, not throughput: a batch's tokens reach the frontend
        # ``depth`` launches after it runs, and a finished sequence keeps its
        # row (and pages) for that long too.
        # ``pp_size + 2`` for PP: the stall recurs once per ``depth + 1``
        # launches, and a sweep at PP=2 measured 1983 / 2020 / 2023 / 2019
        # tok/s at depths 2 / 4 / 6 / 8 -- flat from 4 on, because the stall
        # itself grows as it is spread thinner. 4 and 6 tie within noise, so
        # take the shallower one: depth also delays output publication and
        # EOS detection by that many launches. PP=1 has no cross-stage token
        # round-trip and keeps the historical lag of one.
        self._overlap_depth = (
            get_pp_size() + 2 if get_pp_size() > 1 else 1
        )
        # One pinned staging slot per batch that can be live at once: ``depth``
        # un-retired, the one just launched, and the one being written.
        self._num_output_bufs = max(2, self._overlap_depth + 2)
        self._next_tokens_bufs = [
            torch.zeros(
                self.max_running_seqs,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self._num_output_bufs)
        ]
        self._next_tokens_buf_idx = 0
        # Pinned staging for per-token logprobs, mirroring ``_next_tokens_bufs``
        # (keyed by the same ``buf_idx``). Only written on the output rank and
        # only when a batch requested logprobs; sized to the OpenAI
        # ``top_logprobs`` ceiling so the top-k columns never overflow.
        self._max_top_logprobs = 20
        self._lp_sampled_bufs = [
            torch.zeros(
                self.max_running_seqs,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self._num_output_bufs)
        ]
        self._lp_topval_bufs = [
            torch.zeros(
                (self.max_running_seqs, self._max_top_logprobs),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self._num_output_bufs)
        ]
        self._lp_topid_bufs = [
            torch.zeros(
                (self.max_running_seqs, self._max_top_logprobs),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self._num_output_bufs)
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
        actual H2D and embed work on ``forward_stream`` behind the previous
        batch's GPU work.
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

        All work (H2D copies into the shared input buffers, deferred multimodal
        embed for prefill seqs, and decode-embedding scatter) is enqueued on
        ``forward_stream``. Because the buffers are shared, prep must follow
        the previous forward anyway; stream FIFO provides that ordering without
        consumed/ready events. Enqueueing remains host-asynchronous.
        """
        with torch.cuda.stream(self.forward_stream):
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
                if not self._run_generic_piecewise_forward(num_cal_tokens):
                    self.forward()
        else:
            if not self._run_generic_piecewise_forward(num_cal_tokens):
                self.forward()
        return num_cal_tokens

    @torch.inference_mode()
    def run_batch_async(
        self, dp_padded_size: Optional[int] = None
    ) -> Tuple[Optional[torch.cuda.Event], int, List[int], int, Optional[int]]:
        """Launch one pipeline stage on ``forward_stream`` without host waits.

        ``dp_padded_size`` (DP+EP only) forces the graph bucket agreed on by the
        driver across all DP groups, keeping the captured MoE collectives'
        shapes identical world-wide; ``None`` runs eager (prefill/mixed) or the
        normal local bucket selection (non-DP).
        """
        num_cal_tokens = self.input_data.tokens_cpu.shape[0]
        batch_size = len(self.input_data.seqs)
        buf_idx = self._next_tokens_buf_idx
        self._next_tokens_buf_idx = (buf_idx + 1) % self._num_output_bufs
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

        # ``prepare_input_gpu`` enqueued all H2D + (VL) embed work immediately
        # before this call on ``forward_stream``. FIFO ordering protects the
        # shared input buffers without cross-stream events.
        # DP+EP metadata collectives are issued on the worker's default stream
        # before dispatch and must precede the forward.  The ordinary TP/PP
        # path has no such producer: input H2D and recurrent-state reset/restore
        # are already FIFO-ordered on ``forward_stream``. Recording a
        # default-stream event on
        # every non-DP batch was therefore pure host/API overhead.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.forward_stream):
            if is_dp_attn():
                self.forward_stream.wait_stream(default_stream)
            if is_last_pp_rank():
                has_futures = self.future_map.has_futures(
                    self.input_data.tokens_cpu
                )
            else:
                has_futures = self.future_map.wait_for_inputs(
                    self.input_data.tokens_cpu,
                    self.forward_stream,
                )
            if has_futures:
                self.future_map.resolve_future(
                    self.input_data.tokens[:num_cal_tokens]
                )

            # PP followers receive activations directly on the forward stream.
            # ``irecv`` establishes stream ordering, so the model forward below
            # consumes the buffer only after its upstream producer has sent it;
            # the worker thread remains free to prepare later batches.
            #
            # Staging this in a two-slot ring on its own stream -- to break the
            # write-after-read coupling that makes the pipeline critically
            # damped -- measured 1.2% slower for the send half and 2.3% for both
            # halves, bitwise-identical output either way: at decode microbatch
            # widths (16 tokens, 164 KB) the cross-stream handoff costs more
            # than the slack it buys.
            if not is_first_pp_rank():
                recv_pp_data_async(
                    get_last_pp_rank(),
                    num_cal_tokens,
                    self.input_hidden_states,
                    self.input_residual,
                    self.model.ret_residual,
                )

            # This per-row Python sum only feeds the VL re-embed, which is a
            # no-op unless ``use_mm``. It runs between the previous batch's
            # last GPU work and this batch's forward, so on a text model it
            # was pure bubble.
            if self.use_mm:
                num_decode_tokens = sum(
                    s.to_compute_token_num
                    for s in self.input_data.seqs
                    if s.computed_prompt
                )
                self._fixup_vl_decode_embeddings(num_decode_tokens)

            num_cal_tokens = self._run_forward_on_stream(
                num_cal_tokens, dp_padded_size=dp_padded_size
            )

            if not is_last_pp_rank():
                # The input metadata may be overwritten as soon as this stage's
                # forward completes; activation/token P2P below touches only the
                # output and FutureMap buffers.
                output = (
                    self.output_hidden_states[:num_cal_tokens],
                    self.output_residual[:num_cal_tokens],
                )
                if not self.model.ret_residual:
                    output = output[0]
                send_pp_data_async(output, get_next_pp_rank())
            if not is_last_pp_rank():
                # Post the sampled-token receive now, outside the
                # ``forward_stream`` context: this batch's activations are on
                # their way, so the collective has a full stage time to drain
                # before any collect asks for it. Deferring it to the collect
                # step (or to whenever a successor's input happens to reference
                # the slot) pinned the receive to two launches after dispatch,
                # i.e. always while the last stage was still forwarding this
                # batch -- its NCCL kernel then spun ~one forward and the
                # driver's ``copy_done.synchronize()`` paid for it, no matter
                # how deep the collect lag was.
                copy_done = self.complete_pp_token_feedback(
                    future_slot_ids, batch_size, buf_idx
                )
                return copy_done, batch_size, future_slot_ids, buf_idx, None

            hidden = self.output_hidden_states[:num_cal_tokens]
            logits = self.model.compute_logits(self.input_data, hidden)
            self.input_data.prepare_sample()
            self.memory_manager.apply_resolved_repetition_tokens(self.input_data)
            next_tokens_gpu = None
            # ``lp_k`` doubles as the "logprobs requested this batch" flag for
            # the collect side: ``None`` => none requested (skip staging),
            # >= 0 => number of top alternatives staged. Only the output rank
            # holds full logits, so only it computes/stages logprobs.
            lp_k = None
            lp_gpu = None
            # Determinism/deadlock note: ``compute_logits`` all-gathers, so EVERY
            # TP rank holds full logits and CAN sample -- and under the overlap
            # pipeline every rank MUST. If only the output rank ran the (heavier,
            # multi-round rejection) sampling kernel, the per-rank GPU-time
            # asymmetry would drift ranks out of lockstep and interleave the
            # get_tp_group collective sequence (graph all-reduce -> LM-head
            # all-gather -> token broadcast) across iterations -> NCCL deadlock.
            # All ranks therefore run the sampler for timing symmetry; the
            # broadcast below keeps the output rank's draw authoritative for
            # correctness (sampling amplifies fp all-reduce epsilon).
            _all_greedy = all(s.top_k == 1 for s in self.input_data.seqs)
            # MTP verifies on every TP rank, including greedy batches. Keep
            # grammar histories alive on all ranks already during prefill and
            # ordinary decode, so entering MTP never starts a fresh matcher
            # after tokens have been emitted on the output rank.
            _sample_here = (is_output_rank() or not _all_greedy or any(
                getattr(s, "structured_output", None) is not None
                for s in self.input_data.seqs
            ))
            if _sample_here:
                seqs = self.input_data.seqs
                # The current forward is already enqueued. Advance grammar
                # from the previous authoritative D2H result while it runs;
                # only mask application and sampling follow on the GPU stream.
                structured = self.sampler.prepare_structured(logits, seqs)
                if is_output_rank() and any(s.logprobs_enabled for s in seqs):
                    lp_k = min(
                        self._max_top_logprobs,
                        max(
                            (s.num_top_logprobs for s in seqs if s.logprobs_enabled),
                            default=0,
                        ),
                    )
                    next_tokens_gpu, lp_gpu = self.sampler.forward_gpu(
                        logits, self.input_data, True, lp_k, structured=structured
                    )
                else:
                    _nt = self.sampler.forward_gpu(logits, self.input_data, structured=structured)
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
                    get_rank() - get_tp_rank() if is_dp_attn() else get_output_rank()
                )
                dist.broadcast(
                    next_tokens_gpu,
                    src=tp_src,
                    group=get_tp_group(),
                )
            if next_tokens_gpu is not None:
                self.sampler.stage_structured_feedback(
                    next_tokens_gpu, self.input_data.seqs, stream=self.copy_stream
                )
            # Same MTP head KV maintenance as the synchronous ``step_once``:
            # this is the non-speculative overlap step (pure prefill, an MTP
            # batch-size crossover, ...), and without it those tokens leave a
            # gap the draft would later read off a recycled page.  It must run
            # AFTER the broadcast -- a rank that wrote different head KV would
            # propose different drafts and diverge -- and BEFORE
            # the next batch's FIFO-ordered input writes. ``next_tokens_gpu``
            # stays on the device; ``_mtp_sync_kv``
            # scatters it without a host round-trip.
            if next_tokens_gpu is not None:
                with torch.profiler.record_function("gllm::mtp_head_kv_sync"):
                    self._mtp_sync_kv(
                        self.input_data,
                        hidden,
                        tail_seqs=self.input_data.seqs,
                        tail_next_tokens=next_tokens_gpu,
                    )
            self.future_map.store_to_map(future_indices, next_tokens_gpu)
            if get_pp_size() > 1:
                # Issue the feedback broadcast on ``feedback_stream``, not on
                # ``forward_stream``. ``Work.wait()`` blocks the issuing stream
                # until the collective completes, and a broadcast completes
                # only once every earlier stage has posted its matching
                # receive -- which happens on their collect step, several
                # launches later. Leaving it on ``forward_stream`` therefore
                # gated this stage's *next* forward on the driver's CPU loop
                # and serialized the whole pipeline.
                self.feedback_stream.wait_stream(self.forward_stream)
                next_tokens_gpu.record_stream(self.feedback_stream)
                with torch.cuda.stream(self.feedback_stream):
                    send_pp_tokens_to_previous_stages(next_tokens_gpu)
            # The last stage keeps sampled tokens on GPU and broadcasts them
            # back through the PP group. Only PP0 stages D2H-copy those tokens
            # into their scheduler's pinned output slot; optional logprobs are
            # staged separately on the output rank.
            copy_on_this_rank = (
                is_first_pp_rank() and (get_tp_size() > 1 or is_output_rank())
            ) or lp_gpu is not None
            if copy_on_this_rank:
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_stream(self.forward_stream)
                    # These sources were allocated on ``forward_stream`` and
                    # die with this call. Without ``record_stream`` the caching
                    # allocator hands their blocks to the next batch's
                    # forward-stream allocations while this D2H is still
                    # queued, and the copy reads that batch's int32 metadata
                    # as "tokens".
                    if is_first_pp_rank():
                        next_tokens_gpu.record_stream(self.copy_stream)
                        next_tokens_cpu[:batch_size].copy_(
                            next_tokens_gpu, non_blocking=True
                        )
                    # Stage this batch's logprobs into the same buf_idx slot so
                    # ``_collect_batch`` can read them once ``copy_done`` fires.
                    if lp_gpu is not None:
                        sampled, top_vals, top_ids = lp_gpu
                        for t in lp_gpu:
                            t.record_stream(self.copy_stream)
                        self._lp_sampled_bufs[buf_idx][:batch_size].copy_(
                            sampled, non_blocking=True
                        )
                        if lp_k > 0:
                            self._lp_topval_bufs[buf_idx][:batch_size, :lp_k].copy_(
                                top_vals, non_blocking=True
                            )
                            self._lp_topid_bufs[buf_idx][:batch_size, :lp_k].copy_(
                                top_ids, non_blocking=True
                            )


        copy_done = torch.cuda.Event()
        if (
            is_first_pp_rank() and (get_tp_size() > 1 or is_output_rank())
        ) or lp_gpu is not None:
            copy_done.record(self.copy_stream)
        else:
            copy_done.record(self.forward_stream)
        return copy_done, batch_size, future_slot_ids, buf_idx, lp_k

    def complete_pp_token_feedback(
        self, future_slot_ids: List[int], batch_size: int, buf_idx: int
    ) -> torch.cuda.Event:
        """Post delayed PP sampled-token feedback for one pending batch.

        Delaying this receive until collect keeps the launch-current then
        correct-previous ordering.  FutureMap still records a slot-specific
        event, so a dependent successor can wait on the GPU without a global
        stream synchronization.
        """
        if is_last_pp_rank():
            raise RuntimeError("last PP rank produces rather than receives feedback")
        if not future_slot_ids:
            raise ValueError("PP feedback requires at least one future slot")
        interval = slice(future_slot_ids[0], future_slot_ids[-1] + 1)
        future_indices = FutureIndices(interval=interval)
        future_tokens = self.future_map.token_ids_buf[interval]
        with torch.cuda.stream(self.feedback_stream):
            recv_pp_tokens_from_last_stage(future_tokens)
            feedback_done = torch.cuda.Event()
            feedback_done.record(self.feedback_stream)
        self.future_map.mark_ready(future_indices, feedback_done)

        if is_first_pp_rank():
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(feedback_done)
                self._next_tokens_bufs[buf_idx][:batch_size].copy_(
                    future_tokens, non_blocking=True
                )
            copy_done = torch.cuda.Event()
            copy_done.record(self.copy_stream)
            return copy_done
        return feedback_done

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
