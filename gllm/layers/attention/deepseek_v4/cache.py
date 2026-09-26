"""Cache record for the DeepSeek-V4 attention layer's non-paged state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gllm.layers.attention.deepseek_v4.compressor import CompressorState


@dataclass
class DeepseekV4AttentionCache:
    """Sliding-window, compressed-KV and compressor state for one V4 layer.

    Serving code may place the two compressor states in the shared request
    arena; keeping them explicit here makes the online update order directly
    testable without coupling the layer math to one cache allocator.
    """

    window: torch.Tensor
    compressed: torch.Tensor | None
    index_compressed: torch.Tensor | None
    compressor_state: CompressorState | None
    indexer_state: CompressorState | None


__all__ = [
    "DeepseekV4AttentionCache",
]
