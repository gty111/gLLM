"""Native-precision projections for DeepSeek-V4 sparse attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from gllm.distributed.parallel_state import get_tp_size
from gllm.layers.attention.deepseek_v4.ops import (
    apply_rope_inplace,
    fp8_fake_quantize_inplace,
)
from gllm.layers.layernorm import RMSNorm
from gllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from gllm.layers.quantization.fp8 import fp8LinearMethod


class DeepseekV4AttentionProjections(nn.Module):
    """Q/KV and grouped output projections in checkpoint precision.

    This class deliberately excludes cache management and sparse attention.
    It provides a separately verifiable numerical boundary around all learned
    matrices in a V4 attention layer.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        quant_config = config.quantization_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.num_groups = config.o_groups
        self.eps = config.rms_norm_eps

        tp_size = get_tp_size()
        if self.num_heads % tp_size or self.num_groups % tp_size:
            raise ValueError("V4 attention heads/groups must divide tensor parallelism")
        self.local_num_heads = self.num_heads // tp_size
        self.local_num_groups = self.num_groups // tp_size
        # Built on first use: the weights are not loaded yet at __init__ time.
        self._group_views = None
        if self.local_num_heads % self.local_num_groups:
            raise ValueError("local V4 attention heads must divide output groups")
        self.group_width = (
            self.local_num_heads // self.local_num_groups * self.head_dim
        )

        self.wq_a = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        self.q_norm = RMSNorm(
            self.q_lora_rank, self.eps, params_dtype=torch.bfloat16
        )
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        self.kv_norm = RMSNorm(
            self.head_dim, self.eps, params_dtype=torch.bfloat16
        )

        # Each local output group has its own [o_lora_rank, group_width]
        # matrix. Column parallelism shards complete groups across TP ranks.
        self.wo_a = ColumnParallelLinear(
            self.group_width,
            self.num_groups * self.o_lora_rank,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        self.wo_b = RowParallelLinear(
            self.num_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            reduce_results=True,
        )


    def prepare_q_kv(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized Q latent, attention Q, and QAT-rounded KV."""
        if hidden_states.ndim != 3:
            raise ValueError("V4 attention projections expect [B,S,H] input")
        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).view(
            *hidden_states.shape[:2], self.local_num_heads, self.head_dim
        )
        # Preserve the reference's BF16 reduction/rounding here.
        q.mul_(torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps))
        apply_rope_inplace(q[..., -self.rope_dim :], frequencies)

        kv = self.kv_norm(self.wkv(hidden_states))
        apply_rope_inplace(kv[..., -self.rope_dim :], frequencies)
        fp8_fake_quantize_inplace(kv[..., : -self.rope_dim], group_size=64)
        return qr, q, kv

    def prepare_kv(
        self,
        hidden_states: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the QAT-rounded shared KV projection.

        DSpark prefill consumes hidden states from the target model solely to
        populate its sliding-window KV cache.  Keeping that path separate
        avoids executing the unused query projections and mirrors the official
        reference implementation.
        """
        if hidden_states.ndim != 3:
            raise ValueError("V4 KV projection expects [B,S,H] input")
        kv = self.kv_norm(self.wkv(hidden_states))
        apply_rope_inplace(kv[..., -self.rope_dim :], frequencies)
        fp8_fake_quantize_inplace(kv[..., : -self.rope_dim], group_size=64)
        return kv

    def _grouped_wo_a(self, grouped_input: torch.Tensor) -> torch.Tensor:
        if tuple(grouped_input.shape[-2:]) != (
            self.local_num_groups,
            self.group_width,
        ):
            raise ValueError(
                "grouped attention output must end in "
                f"[{self.local_num_groups}, {self.group_width}]"
            )
        # Hold the per-group weight and scale views instead of re-slicing every
        # forward. ``fp8LinearMethod`` memoizes DeepGEMM's packed scale *on the
        # scale tensor*, so a fresh view per call silently defeated that cache
        # and re-packed 2 groups x 43 layers every decode step.
        if self._group_views is None:
            block_n, block_k = self.wo_a.weight_block_size
            self._group_views = (
                list(
                    self.wo_a.weight.view(
                        self.local_num_groups,
                        self.o_lora_rank,
                        self.group_width,
                    ).unbind(0)
                ),
                list(
                    self.wo_a.weight_scale_inv.view(
                        self.local_num_groups,
                        self.o_lora_rank // block_n,
                        self.group_width // block_k,
                    ).unbind(0)
                ),
            )
        weight, scales = self._group_views
        outputs = []
        for group in range(self.local_num_groups):
            outputs.append(
                fp8LinearMethod(
                    grouped_input[..., group, :],
                    weight[group],
                    block_size=self.wo_a.weight_block_size,
                    weight_scale=scales[group],
                    input_scale=None,
                    round_scale=self.wo_a.use_ue8m0,
                )
            )
        return torch.stack(outputs, dim=-2)

    def project_output(
        self,
        attention_output: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """Apply inverse RoPE and the grouped native-FP8 O projection."""
        if attention_output.shape[-2:] != (self.local_num_heads, self.head_dim):
            raise ValueError("attention output head shape does not match V4 config")
        apply_rope_inplace(
            attention_output[..., -self.rope_dim :],
            frequencies,
            inverse=True,
        )
        grouped = attention_output.view(
            *attention_output.shape[:2],
            self.local_num_groups,
            self.group_width,
        )
        o_lora = self._grouped_wo_a(grouped)
        return self.wo_b(o_lora.flatten(-2))


__all__ = ["DeepseekV4AttentionProjections"]
