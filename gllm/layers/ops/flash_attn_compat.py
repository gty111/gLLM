"""Unified varlen prefill adapter for a pre-validated attention backend.

Backend selection belongs to the runtime configuration validation. This
module only dispatches to the final backend and never changes that decision at
kernel-call time. It never loops over individual sequences.
"""

import math
import os
from typing import Optional

import torch
from gllm.layers.ops.flashinfer_utils import ensure_ninja_on_path

ensure_ninja_on_path()

from flashinfer.prefill import (
    BatchPrefillWithRaggedKVCacheWrapper,
    trtllm_ragged_attention_deepseek,
)

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func
except Exception:
    _fa4_varlen_func = None


_WORKSPACE_BYTES = int(
    os.environ.get("GLLM_FLASHINFER_WORKSPACE_SIZE", str(512 * 1024 * 1024))
)
_workspaces: dict[int, torch.Tensor] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workspace = _workspaces.get(device_index)
    if workspace is None:
        # FlashInfer requires its global workspace to be zeroed on first use.
        workspace = torch.zeros(
            _WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=torch.device("cuda", device_index),
        )
        _workspaces[device_index] = workspace
    return workspace


def _planned_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float],
) -> BatchPrefillWithRaggedKVCacheWrapper:
    # Inference-mode metadata has no version counter and may be updated in
    # place. Always plan against the current lengths; only workspace is reused.
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        _workspace(q.device), "NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr=cu_seqlens_q,
        kv_indptr=cu_seqlens_k,
        num_qo_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim_qk=q.shape[-1],
        head_dim_vo=v.shape[-1],
        causal=causal,
        sm_scale=softmax_scale,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=q.dtype,
        non_blocking=True,
    )
    return wrapper


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    backend: str,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    **kwargs,
):
    """Run varlen attention with an already resolved backend."""
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have equal batch size")
    backend = backend.lower()
    if backend not in ("fa4", "flashinfer", "fa3"):
        raise ValueError(
            "backend must be the resolved value 'fa4', 'flashinfer', or 'fa3', "
            f"got {backend!r}"
        )

    if backend == "fa3":
        from sgl_kernel.flash_attn import flash_attn_varlen_func as sgl_varlen

        value_dim = v.shape[-1]
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        if (
            q.shape[-1] != value_dim
            and torch.cuda.get_device_capability(q.device)[0] == 8
        ):
            # Ampere/Ada kernels require equal QK/V dimensions. Zero padding
            # preserves attention scores, using the original softmax scale.
            head_dim = max(q.shape[-1], value_dim)
            q = torch.nn.functional.pad(q, (0, head_dim - q.shape[-1]))
            k = torch.nn.functional.pad(k, (0, head_dim - k.shape[-1]))
            v = torch.nn.functional.pad(v, (0, head_dim - value_dim))
        result = sgl_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            **kwargs,
        )
        if return_softmax_lse:
            # SGL also returns auxiliary tensors; callers expect (out, lse).
            return result[0][..., :value_dim], result[1]
        return result[..., :value_dim]

    if backend == "fa4":
        if _fa4_varlen_func is None:
            raise RuntimeError(
                "resolved attention backend is 'fa4', but flash-attn-4 "
                "could not be imported; configuration validation must run "
                "before attention execution"
            )
        result = _fa4_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=return_softmax_lse,
            **kwargs,
        )
        if return_softmax_lse:
            return result
        return result[0] if isinstance(result, tuple) else result

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported FlashInfer varlen options: {unsupported}")

    head_dims = (q.shape[-1], k.shape[-1], v.shape[-1])
    trtllm_dims = {(128, 128, 128), (192, 192, 128), (256, 256, 256)}
    if head_dims in trtllm_dims:
        seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        result = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=_workspace(q.device),
            seq_lens=seq_lens_k,
            max_q_len=int(max_seqlen_q or q.shape[0]),
            max_kv_len=int(max_seqlen_k or k.shape[0]),
            bmm1_scale=softmax_scale or q.shape[-1] ** -0.5,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=cu_seqlens_q.numel() - 1,
            window_left=-1,
            cum_seq_lens_q=cu_seqlens_q,
            cum_seq_lens_kv=cu_seqlens_k,
            enable_pdl=False,
            is_causal=causal,
            return_lse=return_softmax_lse,
        )
    else:
        # FlashInfer FA2 uses tiled head dimensions. An irregular vision width
        # such as Qwen3-VL's 72 compiles but produces incorrect results on SM80.
        # Pad to a supported multiple of 64, preserving the original QK scale.
        value_dim = v.shape[-1]
        qk_dim = ((q.shape[-1] + 63) // 64) * 64
        vo_dim = ((value_dim + 63) // 64) * 64
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        if q.shape[-1] != qk_dim:
            q = torch.nn.functional.pad(q, (0, qk_dim - q.shape[-1]))
            k = torch.nn.functional.pad(k, (0, qk_dim - k.shape[-1]))
        if value_dim != vo_dim:
            v = torch.nn.functional.pad(v, (0, vo_dim - value_dim))
        wrapper = _planned_wrapper(
            q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale
        )
        result = wrapper.run(q, k, v, return_lse=return_softmax_lse)
        if return_softmax_lse:
            # FlashInfer FA2 reports base-2 LSE; the FA-compatible interface
            # returns natural-log LSE, including when padded dimensions are used.
            result = (result[0][..., :value_dim], result[1] * math.log(2.0))
        else:
            result = result[..., :value_dim]

    if not return_softmax_lse:
        return result
    output, lse = result
    # FA4 uses [heads, total_q]; MLA's adapter turns it back into
    # [total_q, heads] for state merging.
    return output, lse.transpose(0, 1).contiguous()
