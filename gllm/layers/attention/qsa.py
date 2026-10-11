# SPDX-License-Identifier: Apache-2.0
# Source reference: vLLM's Qwen4Exp QSA tests and kernels.
"""Eager numerical oracle for Qwen Sparse Attention, independent of paging.

Keys passed to selection are already mean-pooled, Gemma-normalized and rotated
at the FIRST token of each micro-block. Queries are normalized and rotated at
their own positions. Runtime cache writing, projections and RoPE stay outside
these functions. This code preserves QSA sparsity; it never substitutes full
attention when the budget is exceeded.
"""

import math

import torch


def indexer_rotary_cache(positions, rotary):
    """Resolve text or T/H/W positions to the shared rotary frequency pairs."""
    cache = rotary.cos_sin_cache[positions]
    if positions.ndim == 1:
        return cache
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("Expected text positions or three MRoPE axes")
    if rotary.mrope_interleaved:
        axis = rotary.interleaved_axis.view(1, -1, 1).expand(positions.shape[-1], -1, 1)
        return cache.permute(1, 2, 0).gather(2, axis).squeeze(2)
    cos, sin = cache.chunk(2, -1)
    sections = rotary.mrope_section
    return torch.cat(
        [
            torch.cat([part[i] for i, part in enumerate(value.split(sections, -1))], -1)
            for value in (cos, sin)
        ],
        -1,
    )


def apply_indexer_rope(tensor, positions, rotary, *, cos_sin=None):
    """Use the main attention's rotary frequencies with an indexer's head size.

    QSA index heads use the SAME rotary dimension as main Q/K, rather than
    applying partial_rotary_factor to the smaller indexer head dimension.
    The initial serving path accepts logical one-dimensional text positions.
    """
    if tensor.shape[0] != positions.shape[-1]:
        raise ValueError("Indexer RoPE expects one position per row")
    dim = rotary.rotary_dim
    if dim > tensor.shape[-1]:
        raise ValueError("Indexer head is smaller than the shared rotary dimension")
    cache = indexer_rotary_cache(positions, rotary) if cos_sin is None else cos_sin
    cos, sin = cache.chunk(2, dim=-1)
    cos, sin = cos.to(tensor.dtype).unsqueeze(1), sin.to(tensor.dtype).unsqueeze(1)
    left, right = tensor[..., :dim].chunk(2, -1)
    return torch.cat(
        (left * cos - right * sin, right * cos + left * sin, tensor[..., dim:]), -1
    )


def compress_index_keys(raw_keys: torch.Tensor, ratio: int) -> torch.Tensor:
    """Mean-pool complete micro-blocks in fp32, then return the input dtype.

    ``raw_keys`` is [sequence, dim], BEFORE normalization / RoPE. An incomplete
    final group stays in the raw-key history until it is complete.
    """
    if raw_keys.ndim != 2 or ratio <= 0:
        raise ValueError("Expected [sequence, dim] raw keys and positive ratio")
    complete = raw_keys.shape[0] // ratio
    return (
        raw_keys[: complete * ratio]
        .float()
        .reshape(complete, ratio, raw_keys.shape[-1])
        .mean(1)
        .to(raw_keys.dtype)
    )


def select_qsa_tokens(query, compressed_keys, query_positions, *, budget, ratio):
    """Select complete blocks, then append the causally visible raw tail.

    Inputs: query [queries, index_heads, dim], compressed_keys [blocks, dim],
    zero-based logical query positions [queries]. Returns logical token IDs
    [queries, budget + ratio - 1], padded with -1. Invalid cache positions are
    never selected, including blocks that complete later in the same prefill.
    """
    if ratio <= 0 or budget <= 0 or budget % ratio:
        raise ValueError("QSA budget must be a positive multiple of ratio")
    if query.ndim != 3 or compressed_keys.ndim != 2:
        raise ValueError("Expected [queries, heads, dim] and [blocks, dim]")
    if query.shape[-1] != compressed_keys.shape[-1]:
        raise ValueError("Indexer query and key dimensions differ")
    if query_positions.shape != (query.shape[0],) or torch.any(query_positions < 0):
        raise ValueError("Expected one non-negative logical position per query")
    scores = torch.einsum("qhd,bd->qbh", query.float(), compressed_keys.float())
    scores = scores.relu().sum(-1) / math.sqrt(query.shape[-1])
    selected = torch.full(
        (query.shape[0], budget + ratio - 1), -1, dtype=torch.long, device=query.device
    )
    offsets = torch.arange(ratio, device=query.device)
    for row, position in enumerate(query_positions.tolist()):
        visible_blocks = (position + 1) // ratio
        if visible_blocks > compressed_keys.shape[0]:
            raise ValueError("Compressed cache is missing a completed visible block")
        count = min(visible_blocks, budget // ratio)
        if count:
            blocks = torch.topk(scores[row, :visible_blocks], count).indices
            selected[row, : count * ratio] = (
                blocks[:, None] * ratio + offsets
            ).flatten()
        tail_start = visible_blocks * ratio
        tail_count = position + 1 - tail_start
        selected[row, count * ratio : count * ratio + tail_count] = torch.arange(
            tail_start, position + 1, device=query.device
        )
    return selected


def sparse_gqa_attention(query, keys, values, indices, *, scale=None):
    """Attend to selected logical tokens; all softmax accumulation is fp32.

    Query [queries, q_heads, dim], keys/values [sequence, kv_heads, dim].
    Output has the query shape and dtype. The caller owns causal selection.
    """
    if query.ndim != 3 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("Expected three-dimensional query and matching K/V")
    if keys.shape[1] <= 0 or query.shape[1] % keys.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    if query.shape[-1] != keys.shape[-1] or indices.shape[0] != query.shape[0]:
        raise ValueError("Attention dimensions or query row counts differ")
    if (
        indices.ndim != 2
        or torch.any(indices < -1)
        or torch.any(indices >= keys.shape[0])
    ):
        raise ValueError("Selected index lies outside the KV cache")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    repeat = query.shape[1] // keys.shape[1]
    output = torch.zeros_like(query)
    for row in range(query.shape[0]):
        logical = indices[row][indices[row] >= 0].long()
        if not logical.numel():
            continue
        k = keys[logical].float().repeat_interleave(repeat, dim=1)
        v = values[logical].float().repeat_interleave(repeat, dim=1)
        logits = torch.einsum("hd,shd->hs", query[row].float(), k) * scale
        output[row] = torch.einsum("hs,shd->hd", logits.softmax(-1), v).to(query.dtype)
    return output
