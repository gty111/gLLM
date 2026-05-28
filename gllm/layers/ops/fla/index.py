# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from collections import OrderedDict
from typing import Tuple

import torch
import triton

from gllm.layers.ops.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# Hot-path note: ``prepare_chunk_indices`` / ``prepare_chunk_offsets`` are
# called from every GDN layer on every prefill step (18 calls per step on
# Qwen3.5-0.8B). The vendored GPU implementation finishes with ``.tolist()``
# which forces a host-side ``cudaStreamSynchronize`` per layer; profiling
# showed ~3 s of sync time and ~39 ms of host gap per prefill step in the
# 16k/c8 workload.
#
# We use the CPU mirror that ``input_data.get_query_start_loc`` attaches as
# ``cu_seqlens._cpu_view`` to (a) compute the indices on the host without
# touching the GPU stream and (b) keep a small value-keyed cache so identical
# batch shapes (the common case for steady-state benchmarks and many real
# workloads) reuse the same on-device tensor with zero further allocation.
# Callers that don't attach a mirror fall through to the original GPU path
# unchanged.


_INDICES_CACHE: "OrderedDict[Tuple, torch.Tensor]" = OrderedDict()
_OFFSETS_CACHE: "OrderedDict[Tuple, torch.Tensor]" = OrderedDict()
_VALUE_CACHE_MAX = 32


def _cpu_mirror(cu_seqlens: torch.Tensor):
    if not cu_seqlens.is_cuda:
        return cu_seqlens
    mirror = getattr(cu_seqlens, "_cpu_view", None)
    if mirror is None or mirror.numel() != cu_seqlens.numel():
        return None
    return mirror


def _cache_get(cache, key, device):
    cached = cache.get(key)
    if cached is None or cached.device != device:
        return None
    cache.move_to_end(key)
    return cached


def _cache_put(cache, key, value):
    cache[key] = value
    if len(cache) > _VALUE_CACHE_MAX:
        cache.popitem(last=False)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    cpu = _cpu_mirror(cu_seqlens)
    if cpu is not None:
        lens = (cpu[1:] - cpu[:-1]).tolist()
        key = (tuple(lens), chunk_size, cu_seqlens.dtype, str(cu_seqlens.device))
        cached = _cache_get(_INDICES_CACHE, key, cu_seqlens.device)
        if cached is not None:
            return cached
        rows = []
        for seq_id, n in enumerate(lens):
            n_chunks = (n + chunk_size - 1) // chunk_size
            for c in range(n_chunks):
                rows.append((seq_id, c))
        if not rows:
            out = torch.empty(
                (0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
        else:
            out_cpu = torch.tensor(
                rows, dtype=cu_seqlens.dtype,
                pin_memory=cu_seqlens.is_cuda,
            )
            out = out_cpu.to(cu_seqlens.device, non_blocking=True) \
                if cu_seqlens.is_cuda else out_cpu
        _cache_put(_INDICES_CACHE, key, out)
        return out

    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    cpu = _cpu_mirror(cu_seqlens)
    if cpu is not None:
        lens = (cpu[1:] - cpu[:-1]).tolist()
        key = (tuple(lens), chunk_size, cu_seqlens.dtype, str(cu_seqlens.device))
        cached = _cache_get(_OFFSETS_CACHE, key, cu_seqlens.device)
        if cached is not None:
            return cached
        n_chunks = [(n + chunk_size - 1) // chunk_size for n in lens]
        offsets = [0]
        acc = 0
        for nc in n_chunks:
            acc += nc
            offsets.append(acc)
        out_cpu = torch.tensor(
            offsets, dtype=cu_seqlens.dtype,
            pin_memory=cu_seqlens.is_cuda,
        )
        out = out_cpu.to(cu_seqlens.device, non_blocking=True) \
            if cu_seqlens.is_cuda else out_cpu
        _cache_put(_OFFSETS_CACHE, key, out)
        return out

    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
