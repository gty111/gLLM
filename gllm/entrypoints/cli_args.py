"""CLI arguments shared by the serving entrypoints.

``api_server`` and ``lm_server`` configure the *same* engine
(:class:`gllm.async_llm_engine.PipeAsyncLLM`); they differ only in what sits in
front of it (an OpenAI-compatible HTTP app vs an encoder-disaggregated LM node).
Every engine knob therefore has to exist in both, and duplicating the
definitions means every new one is added twice, drifts in its default or help
text, or -- as happened with the MTP flags -- lands in one entrypoint only.

The groups below own the shared surface. Each entrypoint adds the groups it
needs plus its own front-end-specific flags, and builds the engine's common
kwargs with :func:`engine_kwargs`, so the argument *and* its plumbing to the
engine are declared once.

Deliberately NOT shared: ``--host`` / ``--port``. Their defaults are part of each
entrypoint's contract (api_server auto-allocates a free port and logs it;
lm_server pins 8000), so folding them in would hide a real behavioural
difference behind a shared helper.
"""

import argparse


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model-path",
        help="Path to the model, either from local disk or from huggingface",
        type=str,
        required=True,
    )
    p.add_argument(
        "--load-format",
        type=str,
        choices=["auto", "dummy"],
        help="auto: actually load model weights; dummy: initialize the model with random values",
        default="auto",
    )
    p.add_argument(
        "--model-max-length",
        type=int,
        help="Maximum sequence length supported by the model (including prompt and generated tokens)",
        default=None,
    )


def add_dist_args(p: argparse.ArgumentParser, *, tp_help: str = None) -> None:
    p.add_argument("--master-addr", type=str, help="NCCL addr", default="0.0.0.0")
    p.add_argument(
        "--master-port",
        type=str,
        help="NCCL rendezvous port (auto-selects a free port when unset).",
        default=None,
    )
    p.add_argument(
        "--tp",
        type=int,
        help=tp_help or "Number of tensor parallel degrees",
        default=1,
    )


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--overlap-scheduling",
        dest="overlap_scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CPU/GPU overlap scheduling with FutureMap (default: on; requires pp=1)",
    )
    p.add_argument(
        "--gpu-memory-util",
        type=float,
        help="GPU memory utilization for KV cache (excluding model weights)",
        default=0.9,
    )
    p.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV cache reuse across requests (default: on)",
    )
    p.add_argument(
        "--page-size", type=int, help="Number of tokens in a page", default=16
    )
    p.add_argument(
        "--attention-backend",
        type=str,
        choices=["auto", "fa3", "flashinfer"],
        default="auto",
        help=(
            "Paged attention backend for ordinary MHA/GQA layers. 'auto' "
            "selects FlashInfer on Blackwell SM100 and sgl-kernel FA3 on "
            "supported SM8x/SM9x GPUs."
        ),
    )
    p.add_argument(
        "--mla-decode-backend",
        type=str,
        choices=["fa3", "flashmla", "triton"],
        default="fa3",
        help=(
            "MLA decode attention backend. 'fa3' (default) uses FA3 absorbed "
            "MLA decode via sgl_kernel (SGLang-compatible); 'flashmla' uses "
            "DeepSeek FlashMLA (auto-bumps page_size to 64); 'triton' uses "
            "the in-tree Triton kernel. Unavailable backends fall back "
            "automatically."
        ),
    )
    p.add_argument(
        "--mamba-ssm-cache-dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help=(
            "Recurrent-state (SSM/GDN) cache precision for hybrid "
            "linear-attention models such as Qwen3.5. 'auto' (default) follows "
            "the checkpoint's mamba_ssm_dtype hint when present and otherwise "
            "uses the activation dtype. An explicit dtype overrides that hint; "
            "float32 uses 2x the state bytes of bf16. The recurrence accumulates "
            "in fp32 inside the kernels either way. No effect on non-hybrid "
            "models."
        ),
    )
    p.add_argument(
        "--ssm-snapshot-stride-tokens",
        type=int,
        default=256,
        help=(
            "Token granularity at which a hybrid model's recurrent state is "
            "cached for prefix reuse (default: 256). Rounded down to whole KV "
            "pages, floored at one page. Smaller means finer restore points on "
            "a prefix-cache hit (less tail recompute) but more state blocks "
            "reserved per prompt; those blocks come from the same pool live "
            "sequences borrow their rolling state from, so going too small "
            "starves sequence admission. Also the grid the scheduler aligns "
            "prefill chunk cuts to (a chunk must END on a boundary for its "
            "state to be cacheable). Only affects hybrid linear-attention "
            "models (e.g. Qwen3.5) with prefix caching enabled."
        ),
    )
    p.add_argument(
        "--mla-cache-dtype",
        type=str,
        choices=["bf16", "fp8"],
        default="bf16",
        help=(
            "MLA latent KV cache precision for DeepSeek Sparse Attention "
            "(V3.2). 'bf16' (default) stores a full-precision latent cache and "
            "runs dense decode (exact for prompts <= index_topk). 'fp8' stores "
            "the FlashMLA FP8-packed cache to drive SM90 sparse decode for long "
            "context. No effect on non-DSA models."
        ),
    )
    p.add_argument(
        "--disable-cuda-graph",
        help=(
            "Disable all CUDA graphs, including decode, MTP draft/verify, "
            "and mixed-MTP piecewise graphs."
        ),
        action="store_true",
    )
    p.add_argument(
        "--max-cuda-graph-bs",
        type=int,
        help=(
            "Maximum batch size for CUDA graph capture. "
            "Larger values allow more decode batches to benefit from CUDA graphs "
            "but increase startup time and GPU memory usage during graph capture. "
            "Default: 512."
        ),
        default=512,
    )


def add_scheduler_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--maxd",
        type=int,
        help="Maximum decode token count per batch (Token Throttling)",
        default=512,
    )
    p.add_argument(
        "--maxp",
        type=int,
        help="Maximum prefill token count per batch (Token Throttling) or token budget in Sarathi-Serve",
        default=8192,
    )
    p.add_argument(
        "--minp",
        type=int,
        help="Minimum prefill token count per batch (Token Throttling)",
        default=32,
    )
    p.add_argument(
        "--iterp",
        type=int,
        help="Number of iterations to process waiting prefill tokens (Token Throttling)",
        default=8,
    )
    p.add_argument(
        "--init-new-token-ratio",
        type=float,
        help="Initial/ceiling fraction of remaining output length reserved for "
        "in-flight decodes (adaptive KV admission control)",
        default=0.7,
    )
    p.add_argument(
        "--min-new-token-ratio",
        type=float,
        help="Floor the new-token-ratio decays toward when the system is stable "
        "(adaptive KV admission control)",
        default=0.1,
    )
    p.add_argument(
        "--schedule-method",
        type=str,
        choices=["split_pd", "chunked_prefill", "token_throttling"],
        help="Specify scheduling method",
        default="chunked_prefill",
    )


def add_mtp_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mtp-enabled",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="MTP speculative decoding: 'auto' (default) enables it iff the "
        "checkpoint ships an MTP head; 'on'/'off' force it. Use 'off' for a "
        "non-speculative baseline.",
    )
    p.add_argument(
        "--mtp-k",
        type=int,
        default=3,
        help="MTP draft-chain length (tokens drafted per target forward).",
    )
    p.add_argument(
        "--mtp-max-batch",
        type=int,
        default=0,
        help="Speculate only while the decode batch has at most this many "
        "sequences. Speculation "
        "multiplies per-step target work by 1+k, so it pays off only while the "
        "batch leaves the GPU under-utilized; past the crossover a plain step is "
        "faster. 0 (default) always speculates. Pick the value from a "
        "concurrency sweep -- it depends on the model and GPU.",
    )


def add_mm_processor_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mm-processor-min-pixels",
        type=int,
        help="Minimum pixels for multimodal processor",
        default=None,
    )
    p.add_argument(
        "--mm-processor-max-pixels",
        type=int,
        help="Maximum pixels for multimodal processor",
        default=None,
    )


def add_frontend_args(p: argparse.ArgumentParser) -> None:
    """OpenAI-frontend flags. ``lm_server`` serves the same app, so it needs these too."""
    p.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=["kimi", "qwen"],
        help="Parser for model-native tool-call output -> structured "
        "tool_calls. Default: auto-detect from model architecture; pass a "
        "name to override.",
    )


def add_engine_args(p: argparse.ArgumentParser, *, tp_help: str = None) -> None:
    """Every engine-facing argument both entrypoints share."""
    add_model_args(p)
    add_dist_args(p, tp_help=tp_help)
    add_runtime_args(p)
    add_scheduler_args(p)
    add_mtp_args(p)
    add_mm_processor_args(p)


def engine_kwargs(args: argparse.Namespace) -> dict:
    """Engine constructor kwargs for the arguments added by :func:`add_engine_args`.

    Keeps the argument and its plumbing in one place -- an entrypoint that adds
    the group but forgets the kwarg is the failure mode this module exists to
    prevent. Entrypoint-specific kwargs (parallelism topology, disagg config,
    ...) are passed by the caller alongside ``**engine_kwargs(args)``.
    """
    return {
        "model_path": args.model_path,
        "load_format": args.load_format,
        "model_max_length": args.model_max_length,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
        "tp_size": args.tp,
        "overlap_scheduling": args.overlap_scheduling,
        "gpu_memory_util": args.gpu_memory_util,
        "enable_prefix_caching": args.enable_prefix_caching,
        "page_size": args.page_size,
        "attention_backend": args.attention_backend,
        "mla_decode_backend": args.mla_decode_backend,
        "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
        "ssm_snapshot_stride_tokens": args.ssm_snapshot_stride_tokens,
        "mla_cache_dtype": args.mla_cache_dtype,
        "disable_cuda_graph": args.disable_cuda_graph,
        "max_cuda_graph_bs": args.max_cuda_graph_bs,
        "maxd": args.maxd,
        "maxp": args.maxp,
        "minp": args.minp,
        "iterp": args.iterp,
        "init_new_token_ratio": args.init_new_token_ratio,
        "min_new_token_ratio": args.min_new_token_ratio,
        "schedule_method": args.schedule_method,
        "mtp_enabled": {"auto": None, "on": True, "off": False}[args.mtp_enabled],
        "mtp_k": args.mtp_k,
        "mtp_max_batch": args.mtp_max_batch,
        "mm_processor_min_pixels": args.mm_processor_min_pixels,
        "mm_processor_max_pixels": args.mm_processor_max_pixels,
    }
