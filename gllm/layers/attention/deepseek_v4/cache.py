"""Cache and prepared-input records shared by the V4 serving layer and oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gllm.layers.attention.deepseek_v4.compressor import CompressorState


@dataclass
class DeepseekV4AttentionCache:
    """Numerical-oracle cache for one V4 attention layer.

    Serving code may place the two compressor states in the shared request
    arena; keeping them explicit here makes the online update order directly
    testable without coupling the layer math to one cache allocator.
    """

    window: torch.Tensor
    compressed: torch.Tensor | None
    index_compressed: torch.Tensor | None
    compressor_state: CompressorState | None
    indexer_state: CompressorState | None


@dataclass
class _PreparedPrefill:
    """Bulk start-at-zero attention inputs plus cache materialization data."""

    query: torch.Tensor
    attention_kv: torch.Tensor
    indices: torch.Tensor
    frequencies: torch.Tensor
    raw_kv: torch.Tensor
    compressed_kv: torch.Tensor | None
    index_kv: torch.Tensor | None
    cache: DeepseekV4AttentionCache


@dataclass
class _PreparedDecode:
    """One online token after its cache/state update, before attention."""

    query: torch.Tensor
    attention_kv: torch.Tensor
    indices: torch.Tensor
    frequency: torch.Tensor
    raw_kv: torch.Tensor


__all__ = [
    "DeepseekV4AttentionCache",
    "_PreparedDecode",
    "_PreparedPrefill",
]
