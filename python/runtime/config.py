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
    # Explicit selection is opt-in; None preserves the legacy MTP policy.
    speculative_backend: Optional[str] = None
    dspark_model_path: Optional[str] = None

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

        backend = self.speculative_backend
        if backend not in (None, "none", "mtp", "dspark"):
            raise ValueError(f"Invalid speculative_backend: {backend!r}")
        if backend == "dspark":
            if not self.dspark_model_path:
                raise ValueError("DSpark requires a local dspark_model_path")
        elif self.dspark_model_path is not None:
            raise ValueError("dspark_model_path requires speculative_backend='dspark'")
        if backend in ("none", "dspark") and self.mtp_enabled is True:
            raise ValueError(f"{backend} conflicts with mtp_enabled=True")
        if backend == "mtp" and self.mtp_enabled is False:
            raise ValueError("mtp conflicts with mtp_enabled=False")
        if backend == "dspark":
            required = {
                "tp_size": 1, "pp_size": 1, "dp_size": 1, "maxd": 1,
                "overlap_scheduling": False, "enable_prefix_caching": False,
                "disable_cuda_graph": True, "piecewise_cuda_graph": False,
                "schedule_method": "chunked_prefill", "attention_backend": "fa3",
                "load_format": "auto",
            }
            for name, value in required.items():
                if getattr(self, name) != value:
                    raise ValueError(f"DSpark requires {name}={value!r}")
            if self.maxp < 8:
                raise ValueError("DSpark requires maxp >= 8")
            if self.disagg_config is not None or self.assigned_layers is not None:
                raise ValueError("DSpark does not support disaggregation or assigned_layers")

    def resolve_mtp_enabled(self, architecture: str, nextn_layers: int) -> bool:
        """Resolve head construction without confusing GDN capacity with MTP."""
        if self.speculative_backend in ("none", "dspark"):
            return False
        if self.speculative_backend == "mtp":
            # V4's joint DSpark stages are not a sequential NextN head.
            if architecture == "DeepseekV4ForCausalLM" or nextn_layers < 1 or self.mtp_k < 1:
                raise ValueError("mtp requires a compatible MTP head and positive mtp_k")
            return True
        if self.mtp_enabled is None and architecture == "DeepseekV4ForCausalLM":
            return False
        return nextn_layers >= 1 if self.mtp_enabled is None else bool(self.mtp_enabled)
