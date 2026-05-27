"""Vendored flash-linear-attention Triton kernels.

Sources: copied verbatim from sglang's ``srt/layers/attention/fla/`` (which
itself is a port of `fla-org/flash-linear-attention
<https://github.com/fla-org/flash-linear-attention>`_, Apache-2.0). The only
modifications are the ``sglang.srt.*`` imports rewritten to point at
:mod:`gllm.layers.ops.fla._sgl_compat` and at the in-tree paths.

Public surface (re-exported here for convenience):

* :func:`chunk_gated_delta_rule` — chunked prefill kernel for the Gated
  DeltaNet linear-attention forward pass.
* :func:`fused_recurrent_gated_delta_rule` /
  :func:`fused_recurrent_gated_delta_rule_packed_decode` — recurrent decode
  kernel for GDN (single-token state update).
* :func:`fused_sigmoid_gating_delta_rule_update` — sigmoid-gated variant of
  the recurrent decode kernel.
* :func:`fused_gdn_gating` — fused ``(A_log, a, b, dt_bias) -> (g, beta)``
  precompute used before :func:`chunk_gated_delta_rule`.
* :func:`rms_norm_gated` / :class:`RMSNormGated` — fused RMSNorm + sigmoid
  gate (``norm_before_gate=True``).
"""

from gllm.layers.ops.fla.chunk import chunk_gated_delta_rule
from gllm.layers.ops.fla.fused_gdn_gating import fused_gdn_gating
from gllm.layers.ops.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from gllm.layers.ops.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from gllm.layers.ops.fla.layernorm_gated import RMSNorm as RMSNormGated
from gllm.layers.ops.fla.layernorm_gated import rms_norm_gated

__all__ = [
    "RMSNormGated",
    "chunk_gated_delta_rule",
    "fused_gdn_gating",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
    "rms_norm_gated",
]
