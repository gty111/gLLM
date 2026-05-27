"""Vendored Mamba/Mamba2 Triton kernels.

Currently exposes the causal 1D conv kernels needed by the GDN linear-
attention layer. The file is copied verbatim from sglang
(``srt/layers/attention/mamba/causal_conv1d_triton.py``); it was originally
authored by the Mamba-SSM and vLLM teams.
"""

from gllm.layers.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
