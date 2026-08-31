"""Whole-model DeepSeek-V4 numerical oracle.

Chains the per-layer token-at-a-time oracles into an end-to-end model with
explicit, contiguous per-layer caches.  Nothing here is on a serving path: it
exists so the packed/paged implementation in :mod:`gllm.models.deepseek_v4` has
something exact to be compared against.
"""

from __future__ import annotations

import torch

from gllm.layers.attention.deepseek_v4.cache import DeepseekV4AttentionCache
from gllm.models.deepseek_v4 import DeepseekV4ModelBase


class DeepseekV4ReferenceModel(DeepseekV4ModelBase):
    """Contiguous-cache oracle over the same decoder layers."""

    def forward_prefill(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[DeepseekV4AttentionCache]]:
        hidden_states = self._embed(input_ids)
        caches = []
        for layer in self.layers:
            hidden_states, cache = layer.forward_prefill(
                hidden_states, input_ids
            )
            caches.append(cache)
        return self._head(hidden_states), caches

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        *,
        position: int,
        caches: list[DeepseekV4AttentionCache],
    ) -> torch.Tensor:
        if len(caches) != len(self.layers):
            raise ValueError("V4 decode cache count does not match decoder depth")
        hidden_states = self._embed(input_ids)
        for layer, cache in zip(self.layers, caches):
            hidden_states = layer.forward_decode(
                hidden_states,
                input_ids,
                position=position,
                cache=cache,
            )
        return self._head(hidden_states)


__all__ = ["DeepseekV4ReferenceModel"]
