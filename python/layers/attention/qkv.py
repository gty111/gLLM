"""Explicit-QKV attention cache writes and backend dispatch."""

from typing import TYPE_CHECKING, Optional

import torch

from gllm.runtime.input_data import InputData

if TYPE_CHECKING:
    from gllm.layers.attention.qkv_backends import QKVAttentionBackend


class QKVAttention:
    """Per-layer QKV cache state shared with a worker-level backend."""

    def __init__(
        self,
        layer_id: int,
        scale: float,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        self.scale = scale
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.backend: Optional["QKVAttentionBackend"] = None

    def set_backend(self, backend: "QKVAttentionBackend") -> None:
        self.backend = backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        input_data: InputData,
    ):
        # The KV segment is built only after the memory-profile forward.
        if getattr(input_data.memory_manager, "segment", None) is None:
            return q

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        input_data.memory_manager.batch_store(
            self.layer_id, k, v, input_data.get_slot_mapping()
        )
        k_cache = input_data.memory_manager.segment.k_cache[self.layer_id]
        v_cache = input_data.memory_manager.segment.v_cache[self.layer_id]

        if self.backend is None:
            raise RuntimeError(
                "QKV attention backend was not injected into QKVAttention"
            )
        plan = input_data.forward_metadata_plan
        if plan is None or plan.attention_metadata is None:
            raise RuntimeError(
                "QKV attention metadata plan was not prepared before forward"
            )
        output = self.backend.forward(
            q,
            k_cache,
            v_cache,
            plan.attention_metadata,
            self.scale,
        )
        return output.view(-1, output.shape[-2] * output.shape[-1])
