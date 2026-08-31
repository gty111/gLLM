"""Fused interleaved complex RoPE for DeepSeek-V4.

The reference does the official BF16 round-trip as a chain of PyTorch ops --
upcast, ``view_as_complex``, complex multiply, ``view_as_real``, flatten,
``copy_`` -- roughly six launches. Six call sites run it (q, kv on two paths,
the indexer query, the compressor, and the inverse rotation in the output
projection), which a single-token decode profile measured at 1.27 ms/step
across 656 kernels.

One program per row does the whole rotation in registers. The arithmetic is
unchanged: fp32 multiply, single bf16 store, so the round-trip rounds exactly
where the reference rounds.

Every call site passes a trailing slice ``x[..., -rope_dim:]`` of a wider
tensor, so rows are strided; the kernel takes explicit strides rather than
forcing a contiguous copy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inplace_kernel(
    x_ptr,
    f_ptr,            # view_as_real(frequencies): (..., P, 2) float32
    S, H,
    sx0, sx1, sx2,    # x strides over (B, S, H); last dim is contiguous
    sf0, sf1, sf2, sf3,   # freq strides over (B, S, H, P); pair dim is last
    P,
    INVERSE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    h = row % H
    bs = row // H
    s = bs % S
    b = bs // S

    p = tl.arange(0, BLOCK_P)
    mask = p < P

    xb = x_ptr + b.to(tl.int64) * sx0 + s.to(tl.int64) * sx1 + h.to(tl.int64) * sx2
    re = tl.load(xb + 2 * p, mask=mask, other=0.0).to(tl.float32)
    im = tl.load(xb + 2 * p + 1, mask=mask, other=0.0).to(tl.float32)

    fb = (
        f_ptr
        + b.to(tl.int64) * sf0
        + s.to(tl.int64) * sf1
        + h.to(tl.int64) * sf2
        + p.to(tl.int64) * sf3
    )
    fr = tl.load(fb, mask=mask, other=0.0)
    fi = tl.load(fb + 1, mask=mask, other=0.0)
    if INVERSE:
        fi = -fi

    tl.store(xb + 2 * p, (re * fr - im * fi).to(tl.bfloat16), mask=mask)
    tl.store(xb + 2 * p + 1, (re * fi + im * fr).to(tl.bfloat16), mask=mask)


def apply_rope_inplace_fused(
    x: torch.Tensor, frequencies: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """In-place interleaved complex RoPE. ``x`` is bf16, ``frequencies`` complex64."""
    if x.ndim == 3:
        b, s, width = x.shape
        h = 1
        sx0, sx1 = x.stride(0), x.stride(1)
        sx2 = 0
        freq = frequencies.view(*frequencies.shape[:2], 1, frequencies.shape[-1])
    else:
        b, s, h, width = x.shape
        sx0, sx1, sx2 = x.stride(0), x.stride(1), x.stride(2)
        freq = frequencies

    pairs = width // 2
    freq = freq.expand(b, s, h, pairs)
    parts = torch.view_as_real(freq)
    sf0, sf1, sf2, sf3, _ = parts.stride()

    _rope_inplace_kernel[(b * s * h,)](
        x, parts, S=s, H=h,
        sx0=sx0, sx1=sx1, sx2=sx2,
        sf0=sf0, sf1=sf1, sf2=sf2, sf3=sf3,
        P=pairs,
        INVERSE=inverse,
        BLOCK_P=triton.next_power_of_2(pairs),
        num_warps=1,
    )
    return x


__all__ = ["apply_rope_inplace_fused"]
