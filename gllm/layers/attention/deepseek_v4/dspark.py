"""Correctness path for DeepSeek-V4 DSpark block attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gllm.distributed.parallel_state import get_tp_rank, get_tp_size
from gllm.layers.attention.deepseek_v4.ops import (
    precompute_rope_frequencies,
    serving_max_length,
    sparse_attention_reference,
)
from gllm.layers.attention.deepseek_v4.projection import (
    DeepseekV4AttentionProjections,
)


@dataclass
class DeepseekV4DSparkAttentionCache:
    """Target-token sliding-window KV owned by one DSpark stage."""

    window: torch.Tensor


class DeepseekV4DSparkAttention(torch.nn.Module):
    """DSpark attention over target history plus one jointly drafted block.

    During target prefill only ``main_hidden`` is projected into the persistent
    window.  During speculative decode, the current target hidden updates that
    window and every draft query attends the window plus all keys in the draft
    block, exactly as the official DSpark reference does.
    """

    def __init__(self, layer_id: int, config) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.window_size = getattr(
            config, "window_size", getattr(config, "sliding_window", 128)
        )
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.max_sequence_length = serving_max_length(config)
        self.projections = DeepseekV4AttentionProjections(config)

        tp_size = get_tp_size()
        tp_rank = get_tp_rank()
        local_heads = config.num_attention_heads // tp_size
        self.attn_sink = torch.nn.Parameter(
            torch.empty(local_heads, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )
        self._sink_slice = slice(tp_rank * local_heads, (tp_rank + 1) * local_heads)

        frequencies = precompute_rope_frequencies(
            self.rope_dim,
            self.max_sequence_length,
            original_sequence_length=0,
            base=config.rope_theta,
            factor=1.0,
            beta_fast=32,
            beta_slow=1,
            device="cuda",
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def make_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> DeepseekV4DSparkAttentionCache:
        return DeepseekV4DSparkAttentionCache(
            window=torch.zeros(
                batch_size,
                self.window_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
        )


    @staticmethod
    def _store_window(
        cache: DeepseekV4DSparkAttentionCache,
        kv: torch.Tensor,
        *,
        start_pos: int,
    ) -> None:
        window_size = cache.window.shape[1]
        positions = torch.arange(
            start_pos,
            start_pos + kv.shape[1],
            device=kv.device,
        )
        cache.window[:, positions % window_size] = kv

    def prefill_main(
        self,
        main_hidden: torch.Tensor,
        cache: DeepseekV4DSparkAttentionCache | None = None,
        *,
        start_pos: int = 0,
    ) -> DeepseekV4DSparkAttentionCache:
        """Populate the stage cache from authoritative target hidden states."""
        if main_hidden.ndim != 3:
            raise ValueError("DSpark main hidden states must have shape [B,S,H]")
        if start_pos != 0:
            raise ValueError("DSpark bulk prefill currently requires start_pos=0")
        batch, sequence_length, _ = main_hidden.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError("DSpark prefill exceeds configured maximum length")
        if cache is None:
            cache = self.make_cache(batch, device=main_hidden.device)
        frequencies = self.frequencies[:sequence_length]
        main_kv = self.projections.prepare_kv(main_hidden, frequencies)
        keep = min(sequence_length, self.window_size)
        if keep:
            positions = torch.arange(
                sequence_length - keep,
                sequence_length,
                device=main_hidden.device,
            )
            cache.window[:, positions % self.window_size] = main_kv[:, -keep:]
        return cache

    def forward_draft_block(
        self,
        hidden_states: torch.Tensor,
        main_hidden: torch.Tensor,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkAttentionCache,
    ) -> torch.Tensor:
        """Run one official DSpark speculative block attention step."""
        if start_pos <= 0:
            raise ValueError("DSpark draft attention requires start_pos > 0")
        if hidden_states.ndim != 3 or main_hidden.ndim != 3:
            raise ValueError("DSpark attention inputs must be [B,S,H]")
        if main_hidden.shape[1] != 1:
            raise ValueError("DSpark decode requires one current target hidden token")
        if hidden_states.shape[0] != main_hidden.shape[0]:
            raise ValueError("DSpark draft/main batch sizes must match")

        main_frequency = self.frequencies[start_pos : start_pos + 1]
        main_kv = self.projections.prepare_kv(main_hidden, main_frequency)
        self._store_window(cache, main_kv, start_pos=start_pos)

        block_size = hidden_states.shape[1]
        draft_start = start_pos + 1
        draft_frequencies = self.frequencies[
            draft_start : draft_start + block_size
        ]
        _, query, draft_kv = self.projections.prepare_q_kv(
            hidden_states, draft_frequencies
        )

        history_count = min(self.window_size, start_pos + 1)
        history_indices = torch.arange(
            history_count, device=hidden_states.device, dtype=torch.int32
        )
        draft_indices = self.window_size + torch.arange(
            block_size, device=hidden_states.device, dtype=torch.int32
        )
        indices = torch.cat([history_indices, draft_indices]).view(1, 1, -1)
        indices = indices.expand(
            hidden_states.shape[0], block_size, -1
        ).contiguous()
        attention_kv = torch.cat([cache.window, draft_kv], dim=1)
        output = sparse_attention_reference(
            query,
            attention_kv,
            indices,
            self.attn_sink,
            self.softmax_scale,
        )
        return self.projections.project_output(output, draft_frequencies)


__all__ = [
    "DeepseekV4DSparkAttention",
    "DeepseekV4DSparkAttentionCache",
]
