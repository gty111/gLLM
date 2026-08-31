"""Fused Triton kernels for the DeepSeek-V4 decode path.

Each module here replaces a region that the reference expresses as a chain of
PyTorch ops. At decode sizes those chains are launch-bound -- tens of
microseconds of kernel launches around a few microseconds of arithmetic -- and
the model runs every one of them 41-43 times per step.

All of them are numerically equivalent to the reference they replace, and each
is pinned against a plain-PyTorch oracle in ``tests/``. The reference forms
live in the test files rather than beside the kernels, so an oracle cannot
drift along with the implementation it checks.
"""

from gllm.layers.ops.deepseek_v4.compress import compress_decode_batch_fused
from gllm.layers.ops.deepseek_v4.mhc import (
    hc_split_sinkhorn_fused,
    mhc_mix_and_sumsq,
    mhc_post_fused,
    mhc_pre_combine,
)
from gllm.layers.ops.deepseek_v4.mxfp4_qat import mxfp4_fake_quantize_fused
from gllm.layers.ops.deepseek_v4.rope import apply_rope_inplace_fused
from gllm.layers.ops.deepseek_v4.scatter import scatter_rows_where

__all__ = [
    "apply_rope_inplace_fused",
    "compress_decode_batch_fused",
    "hc_split_sinkhorn_fused",
    "mhc_mix_and_sumsq",
    "mhc_post_fused",
    "mhc_pre_combine",
    "mxfp4_fake_quantize_fused",
    "scatter_rows_where",
]
