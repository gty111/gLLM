"""Unified varlen prefill adapter for a pre-validated attention backend.

Backend selection belongs to the runtime configuration validation. This
module only dispatches to the final backend and never changes that decision at
kernel-call time. It never loops over individual sequences.
"""

from collections import OrderedDict
import os
import weakref
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
_MAX_PLAN_CACHE = 32
_workspaces: dict[int, torch.Tensor] = {}
_plan_cache: OrderedDict[tuple, tuple] = OrderedDict()


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
    # Tensor version + identity lets every vision/MLA layer reuse one plan,
    # while an in-place metadata update or a new request gets a fresh plan.
    key = (
        q.device.index,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_q._version,
        cu_seqlens_k.data_ptr(),
        cu_seqlens_k._version,
        q.shape[1],
        k.shape[1],
        q.shape[-1],
        v.shape[-1],
        q.dtype,
        k.dtype,
        v.dtype,
        causal,
        softmax_scale,
    )
    cached = _plan_cache.get(key)
    if cached is not None:
        q_ref, k_ref, wrapper = cached
        if q_ref() is cu_seqlens_q and k_ref() is cu_seqlens_k:
            _plan_cache.move_to_end(key)
            return wrapper
        del _plan_cache[key]

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
    _plan_cache[key] = (weakref.ref(cu_seqlens_q), weakref.ref(cu_seqlens_k), wrapper)
    while len(_plan_cache) > _MAX_PLAN_CACHE:
        _plan_cache.popitem(last=False)
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
    if backend not in ("fa4", "flashinfer"):
        raise ValueError(
            "backend must be the resolved value 'fa4' or 'flashinfer', "
            f"got {backend!r}"
        )

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
        wrapper = _planned_wrapper(
            q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale
        )
        result = wrapper.run(q, k, v, return_lse=return_softmax_lse)

    if not return_softmax_lse:
        return result
    output, lse = result
    # FA4 uses [heads, total_q]; MLA's adapter turns it back into
    # [total_q, heads] for state merging.
    return output, lse.transpose(0, 1).contiguous()
