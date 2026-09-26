"""Engine-level configuration, carried as a single object down the stack.

``EngineConfig`` is the single source of truth for the knobs that used to be
re-declared at every layer of the construction chain (``cli_args`` argparse
definitions -> the ``engine_kwargs`` remap table -> ``LLM.__init__`` kwargs ->
``ModelRunner.__init__`` params -> ``Worker.__init__`` positionals). ``LLM``
keeps its public kwargs signature (existing scripts pass them), builds one
``EngineConfig`` from them, and hands the object to the model runner and the
workers; the entrypoints' ``engine_kwargs`` filters ``vars(args)`` against
these field names instead of re-listing them.

The object crosses the ``spawn`` process boundary (pickled with the worker),
so it must stay picklable: plain dataclass, plain field types, and the
already-picklable :class:`gllm.disagg.config.DisaggConfig`.
"""

from dataclasses import dataclass
from typing import Optional

from gllm.disagg.config import DisaggConfig


@dataclass
class EngineConfig:
    """All engine knobs shared by the model runner, workers, and scheduler."""

    # Model
    model_path: str
    load_format: str = "auto"
    model_max_length: Optional[int] = 8192
    # Distributed topology / launch. ``host`` is deliberately NOT here: it is a
    # frontend/server concern (each entrypoint owns its own --host/--port with
    # different defaults), only ``LLM`` itself consumes it.
    master_addr: str = "0.0.0.0"
    master_port: Optional[str] = None
    launch_mode: str = "normal"
    worker_ranks: Optional[str] = None
    pp_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    use_ep: bool = True
    assigned_layers: Optional[list] = None
    # Runtime / cache
    overlap_scheduling: bool = True
    gpu_memory_util: float = 0.9
    enable_prefix_caching: bool = True
    page_size: int = 16
    attention_backend: str = "flashinfer"
    mla_decode_backend: str = "fa4"
    mamba_ssm_cache_dtype: str = "auto"
    ssm_snapshot_stride_tokens: int = 256
    mla_cache_dtype: str = "bf16"
    # CUDA graphs
    disable_cuda_graph: bool = False
    piecewise_cuda_graph: Optional[bool] = True
    max_piecewise_cuda_graph_tokens: Optional[int] = None
    max_cuda_graph_bs: int = 512
    # Scheduler
    maxd: int = 512
    maxp: int = 2048
    minp: int = 32
    iterp: int = 8
    init_new_token_ratio: float = 0.7
    min_new_token_ratio: float = 0.1
    schedule_method: str = "chunked_prefill"
    # MTP speculative decoding
    mtp_enabled: Optional[bool] = None
    mtp_k: int = 3
    mtp_max_batch: int = 0
    # Multimodal processor
    mm_processor_min_pixels: Optional[int] = None
    mm_processor_max_pixels: Optional[int] = None
    # Encoder disaggregation
    disagg_config: Optional[DisaggConfig] = None

    def __post_init__(self):
        # Internal consistency lives here, next to the fields, so entrypoints
        # only forward values and never re-validate them.
        if self.launch_mode not in ("normal", "master", "slave"):
            raise ValueError(f"Invalid launch_mode: {self.launch_mode!r}")
        if self.launch_mode != "normal" and not self.worker_ranks:
            raise ValueError(
                f"launch_mode={self.launch_mode!r} requires --ranks (worker_ranks)"
            )
        for name in ("pp_size", "tp_size", "dp_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
