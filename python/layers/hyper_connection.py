# SPDX-License-Identifier: Apache-2.0
# Source reference:
# vllm/models/qwen4_exp/common/hyperconnection.py (Apache-2.0).
"""Checkpoint-native Gated Residual, with explicit four-stream state.

Stream normalization, gating and residual combination use fused CUDA kernels.
Unlike DeepSeek's mHC, it has no Sinkhorn mixing matrix. Projections are
replicated across TP ranks; the attention/MLP result must already be reduced.
"""

import torch
import torch.nn.functional as F
from torch import nn

from gllm.layers.ops import hyper_connection as fused
from gllm.layers.ops.projection import DownProjectionWorkspace, supports_fused_down


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, groups=1, eps=1e-6, *, device=None, dtype=None):
        super().__init__()
        if hidden_size <= 0 or groups <= 0 or eps <= 0:
            raise ValueError("Norm dimensions and epsilon must be positive")
        self.hidden_size = hidden_size
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(
            torch.zeros(hidden_size * groups, device=device, dtype=dtype)
        )

    def forward(self, x):
        if x.shape[-1] != self.hidden_size * self.groups:
            raise ValueError("Grouped norm input width does not match its weights")
        return fused.grouped_norm(
            x, self.weight, self.hidden_size, self.groups, self.eps
        )


class GatedResidual(nn.Module):
    def __init__(
        self,
        hidden_size,
        hc_count,
        hc_lowrank,
        eps=1e-6,
        *,
        use_combine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if min(hidden_size, hc_count, hc_lowrank) <= 0:
            raise ValueError("Gated Residual dimensions must be positive")
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hyper_hidden_size = hidden_size * hc_count
        self.hc_norm = GroupedGemmaRMSNorm(
            hidden_size, hc_count, eps, device=device, dtype=dtype
        )
        self.input_mix_weight_down = nn.Linear(
            self.hyper_hidden_size, hc_lowrank, bias=False, device=device, dtype=dtype
        )
        self.input_mix_weight_up = nn.Linear(
            hc_lowrank, self.hyper_hidden_size, bias=False, device=device, dtype=dtype
        )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hyper_hidden_size, hc_count, bias=False, device=device, dtype=dtype
            )
        self.register_buffer("_mix_inject_weight", None, persistent=False)
        self._down_workspace = DownProjectionWorkspace()

    @torch.no_grad()
    def process_weights_after_loading(self):
        """Pack projections once, preserving checkpoint names as storage views."""
        if self._mix_inject_weight is not None or not hasattr(
            self, "block_inject_weight"
        ):
            return
        down = self.input_mix_weight_down.weight
        inject = self.block_inject_weight.weight
        rows = down.shape[0] + inject.shape[0]
        packed = down.new_zeros(((rows + 15) // 16 * 16, down.shape[1]))
        packed[: down.shape[0]].copy_(down)
        packed[down.shape[0] : rows].copy_(inject)
        self._mix_inject_weight = packed
        self.input_mix_weight_down.weight.data = packed[: down.shape[0]]
        self.block_inject_weight.weight.data = packed[down.shape[0] : rows]

    def mix(self, streams):
        normalized = self.hc_norm(streams)
        return self._mix_normalized(streams, normalized)

    def combine_and_mix(self, block_output, residual):
        """Combine the preceding block and normalize for this mixer in one pass."""
        streams, logits = residual
        streams, normalized = fused.combine_norm(
            streams,
            block_output,
            logits,
            self.hc_norm.weight,
            self.hidden_size,
            self.hc_count,
            self.hc_norm.eps,
        )
        return self._mix_normalized(streams, normalized)

    def _mix_normalized(self, streams, normalized):
        self.process_weights_after_loading()
        rank = self.input_mix_weight_down.out_features
        if supports_fused_down(
            normalized, self._mix_inject_weight, rank, self.hc_count
        ):
            lowrank, injection = self._down_workspace.run(
                normalized, self._mix_inject_weight, rank, self.hc_count
            )
        else:
            if self._mix_inject_weight is None:
                down = self.input_mix_weight_down(normalized)
                injection = None
            else:
                projected = F.linear(normalized, self._mix_inject_weight)
                down = projected[..., :rank]
                injection = projected[..., rank : rank + self.hc_count]
            lowrank = fused.scaled_silu(down, self.hc_count)
        gate = self.input_mix_weight_up(lowrank)
        mixed = fused.gate_mix(normalized, gate, self.hidden_size, self.hc_count)
        return mixed, (streams, injection)

    def combine(self, block_output, residual):
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("Final Gated Residual mixer has no injection weights")
        streams, logits = residual
        if block_output.shape != (*streams.shape[:-1], self.hidden_size):
            raise ValueError("Block output must have one hidden vector per stream row")
        return fused.combine(
            streams, block_output, logits, self.hidden_size, self.hc_count
        )

    def forward(self, streams, block):
        mixed, residual = self.mix(streams)
        return self.combine(block(mixed), residual)
