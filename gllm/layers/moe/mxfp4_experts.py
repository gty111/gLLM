"""Native MXFP4 routed experts.

The explicit DeepGEMM expert loop mirrors the DeepSeek-V4 reference and remains
the numerical oracle.  On Blackwell, loaded weights are converted once to the
FlashInfer/TRT-LLM layout and evaluated by one grouped routed-MoE operation.
"""

from __future__ import annotations

import torch

from gllm.distributed.parallel_state import (
    ep_all_reduce,
    get_ep_rank,
    get_ep_size,
    get_tp_rank,
    get_tp_size,
    is_use_ep,
    tensor_model_parallel_all_reduce,
)
from gllm.layers.quantization.mxfp4 import deepgemm_mxfp4_expert
from gllm.models.weight_utils import iter_experts
from gllm.layers.quantization.mxfp4 import prepare_mxfp4_scale


def _can_use_flashinfer_mxfp4_moe() -> bool:
    """Return whether the installed stack has the Blackwell grouped kernel."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


@torch.no_grad()
def prepare_flashinfer_mxfp4_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert native packed checkpoint weights to FlashInfer's MoE layout.

    The checkpoint stores FC1 as ``[gate; up]`` while the TRT-LLM kernel uses
    ``[up; gate]`` and a tiled row order.  Scale tensors contain E8M0 bytes;
    their final E4M3 dtype is only the byte carrier expected by FlashInfer.
    """
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("w13 and w2 must be 3D with the same expert count")
    if w13.shape[1] % 2:
        raise ValueError("w13 output dimension must be even")
    if w13_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("FlashInfer preprocessing requires raw E8M0 w13 scales")
    if w2_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("FlashInfer preprocessing requires raw E8M0 w2 scales")

    device = w13.device
    intermediate_size = w13.shape[1] // 2
    swap_gate_up = torch.cat(
        (
            torch.arange(intermediate_size, 2 * intermediate_size, device=device),
            torch.arange(intermediate_size, device=device),
        )
    )
    cache: dict = {}
    epilogue_tile_m = 128

    w13_permute = _maybe_get_cached_w3_w1_permute_indices(
        cache, w13[0], epilogue_tile_m
    ).to(device)
    w13_scale_permute = _maybe_get_cached_w3_w1_permute_indices(
        cache,
        w13_scale[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(device)
    w2_permute = get_w2_permute_indices_with_cache(
        cache, w2[0], epilogue_tile_m
    ).to(device)
    w2_scale_permute = get_w2_permute_indices_with_cache(
        cache,
        w2_scale[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(device)

    # Compose the gate/up swap with the tiled permutation, avoiding a separate
    # full-size [up; gate] allocation during model loading.
    w13_rows = swap_gate_up[w13_permute]
    w13_scale_rows = swap_gate_up[w13_scale_permute]
    flash_w13 = w13.view(torch.uint8)[:, w13_rows].contiguous()
    flash_w2 = w2.view(torch.uint8)[:, w2_permute].contiguous()
    flash_w13_scale = block_scale_interleave(
        w13_scale.view(torch.uint8)[:, w13_scale_rows].contiguous()
    ).view(torch.float8_e4m3fn).reshape(w13.shape[0], w13.shape[1], -1)
    flash_w2_scale = block_scale_interleave(
        w2_scale.view(torch.uint8)[:, w2_scale_permute].contiguous()
    ).view(torch.float8_e4m3fn).reshape(w2.shape[0], w2.shape[1], -1)
    return flash_w13, flash_w2, flash_w13_scale, flash_w2_scale


def flashinfer_mxfp4_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    expert_offset: int,
    swiglu_limit: float,
    scale_ones: torch.Tensor,
    clamp_limits: torch.Tensor,
) -> torch.Tensor:
    """Run the precomputed routing through FlashInfer's grouped FP4 MoE."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if topk_weights.dtype not in (torch.bfloat16, torch.float32):
        topk_weights = topk_weights.float()
    num_tokens = hidden_states.shape[0]
    tune_max_num_tokens = 1 << max(0, num_tokens - 1).bit_length()
    output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    return trtllm_fp4_block_scale_routed_moe(
        topk_ids=(topk_ids, topk_weights),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp_limits,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=scale_ones,
        output1_scale_gate_scalar=scale_ones,
        output2_scale_scalar=scale_ones,
        num_experts=num_experts,
        top_k=topk_ids.shape[1],
        n_group=1,
        topk_group=1,
        intermediate_size=w2.shape[2] * 2,
        local_expert_offset=expert_offset,
        local_num_experts=w13.shape[0],
        routed_scaling_factor=1.0,
        routing_method_type=5,  # TopK decisions and weights are precomputed.
        activation_type=3,  # clamped SwiGLU
        do_finalize=True,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
    )[0]


def deepgemm_mxfp4_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_offset: int = 0,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    """Evaluate locally owned routed experts from precomputed top-k routing.

    Args:
        hidden_states: ``[M, H]`` BF16 activations.
        w13: ``[E_local, 2I, H/2]`` packed FP4 gate/up weights.
        w2: ``[E_local, H, I/2]`` packed FP4 down weights.
        *_scale: raw E8M0 checkpoint scales or DeepGEMM-packed scales.
        topk_weights/topk_ids: ``[M, topk]`` global routing decisions.
        expert_offset: global id of local expert zero.

    Non-local selections contribute zero.  The caller performs the collective
    reduction when expert parallelism is enabled.
    """
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be a 2D bfloat16 tensor")
    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("w13 and w2 must be 3D with the same expert count")
    if topk_weights.shape != topk_ids.shape or topk_ids.ndim != 2:
        raise ValueError("topk_weights and topk_ids must have the same 2D shape")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing token count must match hidden_states")
    if w13_scale.shape[0] != w13.shape[0] or w2_scale.shape[0] != w2.shape[0]:
        raise ValueError("scale expert count must match its weight tensor")

    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for local_expert in range(w13.shape[0]):
        global_expert = expert_offset + local_expert
        token_idx, topk_slot = torch.where(topk_ids == global_expert)
        if token_idx.numel() == 0:
            continue
        expert_output = deepgemm_mxfp4_expert(
            hidden_states[token_idx],
            w13[local_expert],
            w2[local_expert],
            w13_scale[local_expert],
            w2_scale[local_expert],
            routing_weight=topk_weights[token_idx, topk_slot],
            swiglu_limit=swiglu_limit,
        )
        output.index_add_(0, token_idx, expert_output.float())
    return output.to(hidden_states.dtype)


class NativeMXFP4Experts(torch.nn.Module):
    """Load and execute native DeepSeek-V4 packed MXFP4 expert shards.

    Expert parallelism owns complete experts. With EP disabled, every TP rank
    owns all experts and shards their intermediate dimension instead. The raw
    checkpoint tensors stay packed as two E2M1 values per byte; only E8M0 scale
    metadata is rearranged once for DeepGEMM after loading.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 10.0,
    ) -> None:
        super().__init__()
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.swiglu_limit = swiglu_limit
        self.use_expert_parallel = is_use_ep()
        self.ep_size = get_ep_size() if self.use_expert_parallel else 1
        self.ep_rank = get_ep_rank() if self.use_expert_parallel else 0
        self.tp_size = get_tp_size()

        if self.use_expert_parallel:
            if num_experts % self.ep_size:
                raise ValueError(
                    f"num_experts={num_experts} must divide EP={self.ep_size}"
                )
            self.local_num_experts = num_experts // self.ep_size
            self.expert_offset = self.ep_rank * self.local_num_experts
            self.intermediate_size = intermediate_size
        else:
            if intermediate_size % self.tp_size:
                raise ValueError(
                    f"intermediate_size={intermediate_size} must divide "
                    f"TP={self.tp_size}"
                )
            self.local_num_experts = num_experts
            self.expert_offset = 0
            self.intermediate_size = intermediate_size // self.tp_size

        if hidden_size % 32 or self.intermediate_size % 32:
            raise ValueError("MXFP4 expert dimensions must be divisible by 32")
        e = self.local_num_experts
        h = hidden_size
        i = self.intermediate_size
        self.w13_weight = torch.nn.Parameter(
            torch.empty(e, 2 * i, h // 2, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(e, h, i // 2, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        )
        self.w13_scale = torch.nn.Parameter(
            torch.empty(e, 2 * i, h // 32, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        )
        self.w2_scale = torch.nn.Parameter(
            torch.empty(e, h, i // 32, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        )
        self.register_buffer("_flashinfer_scale_ones", None, persistent=False)
        self.register_buffer("_flashinfer_clamp_limits", None, persistent=False)
        self._backend = "unprepared"
        self._scales_prepared = False

    # Checkpoint field -> (stacked parameter, per-expert source names).  ``w1``
    # (gate) and ``w3`` (up) are stacked into one ``w13`` parameter along the
    # intermediate dimension, exactly as the fused kernels expect them.
    _CHECKPOINT_FIELDS = {
        "w13_weight": (("w1", "w3"), "weight"),
        "w13_scale": (("w1", "w3"), "scale"),
        "w2_weight": (("w2",), "weight"),
        "w2_scale": (("w2",), "scale"),
    }

    @torch.no_grad()
    def load_stacked_param(
        self,
        field: str,
        param: torch.Tensor,
        weights,
        prefix: str,
        *,
        pool=None,
    ) -> None:
        """Fill one stacked routed-expert parameter from per-expert checkpoint
        tensors.

        ``prefix`` is the checkpoint namespace immediately before the expert
        id, e.g. ``"layers.3.ffn.experts"``. Expert-parallel ranks own complete
        experts; ordinary tensor-parallel ranks own an aligned slice of every
        expert's intermediate dimension.  ``pool`` is the loader-wide
        :func:`moe_expert_load_pool`; the lazy safetensors index hands each
        worker thread its own file handle, so the per-expert reads overlap.
        """
        if self._scales_prepared:
            raise RuntimeError("cannot load MXFP4 weights after scale packing")
        try:
            sources, suffix = self._CHECKPOINT_FIELDS[field]
        except KeyError:
            raise ValueError(f"unknown MXFP4 expert parameter {field!r}") from None

        local_i = self.intermediate_size
        if self.use_expert_parallel:
            start = 0
        else:
            start = get_tp_rank() * local_i
        end = start + local_i

        def raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dtype in (torch.int8, torch.uint8, torch.float8_e8m0fnu):
                return tensor.view(torch.uint8)
            raise TypeError(
                "V4 MXFP4 checkpoint tensors must be packed int8/uint8 or "
                f"E8M0, got {tensor.dtype}"
            )

        # ``w2`` is the down projection: its *columns* are the intermediate
        # dimension, and the packed layouts halve (2 FP4 values per byte) or
        # thirty-two (one E8M0 scale per 32 values) that axis.
        if field == "w2_weight":
            window = (slice(None), slice(start // 2, end // 2))
        elif field == "w2_scale":
            window = (slice(None), slice(start // 32, end // 32))
        else:
            window = (slice(start, end),)

        def load_one(local_expert: int) -> None:
            base = f"{prefix}.{self.expert_offset + local_expert}"
            for index, name in enumerate(sources):
                source = raw_bytes(weights[f"{base}.{name}.{suffix}"])[window]
                if len(sources) == 1:
                    param[local_expert].copy_(source)
                else:
                    param[
                        local_expert, index * local_i : (index + 1) * local_i
                    ].copy_(source)

        iter_experts(self.local_num_experts, load_one, pool)


    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        """Convert weights once to the fastest supported persistent layout."""
        if self._scales_prepared:
            return
        e = self.local_num_experts
        h = self.hidden_size
        i = self.intermediate_size
        if _can_use_flashinfer_mxfp4_moe():
            w13, w2, s13, s2 = prepare_flashinfer_mxfp4_moe_weights(
                self.w13_weight.data,
                self.w2_weight.data,
                self.w13_scale.data,
                self.w2_scale.data,
            )
            self.w13_weight.data = w13
            self.w2_weight.data = w2
            self.w13_scale.data = s13
            self.w2_scale.data = s2
            self._flashinfer_scale_ones = torch.ones(
                e, dtype=torch.float32, device=self.w13_weight.device
            )
            self._flashinfer_clamp_limits = torch.full(
                (e,),
                self.swiglu_limit,
                dtype=torch.float32,
                device=self.w13_weight.device,
            )
            self._backend = "flashinfer"
        else:
            self.w13_scale.data = prepare_mxfp4_scale(
                self.w13_scale.data,
                output_size=2 * i,
                input_size=h,
                num_groups=e,
            )
            self.w2_scale.data = prepare_mxfp4_scale(
                self.w2_scale.data,
                output_size=h,
                input_size=i,
                num_groups=e,
            )
            self._backend = "deepgemm"
        self._scales_prepared = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        reduce_results: bool = True,
    ) -> torch.Tensor:
        self.process_weights_after_loading()
        if self._backend == "flashinfer":
            output = flashinfer_mxfp4_moe(
                hidden_states,
                self.w13_weight,
                self.w2_weight,
                self.w13_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                num_experts=self.global_num_experts,
                expert_offset=self.expert_offset,
                swiglu_limit=self.swiglu_limit,
                scale_ones=self._flashinfer_scale_ones,
                clamp_limits=self._flashinfer_clamp_limits,
            )
        else:
            output = deepgemm_mxfp4_moe(
                hidden_states,
                self.w13_weight,
                self.w2_weight,
                self.w13_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                expert_offset=self.expert_offset,
                swiglu_limit=self.swiglu_limit,
            )
        if reduce_results:
            if self.use_expert_parallel and self.ep_size > 1:
                output = ep_all_reduce(output)
            elif not self.use_expert_parallel and self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output)
        return output


__all__ = [
    "NativeMXFP4Experts",
    "deepgemm_mxfp4_moe",
    "flashinfer_mxfp4_moe",
    "prepare_flashinfer_mxfp4_moe_weights",
]
