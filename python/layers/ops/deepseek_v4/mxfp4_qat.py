"""Fused MXFP4 fake-quantization for the DeepSeek-V4 lightning indexer.

The reference expresses one call as ~30 elementwise ops -- group amax, an E8M0
power-of-two scale, a seven-way threshold ladder to find the E2M1 code, a
quadratic LUT, then rescale. Every one is launch-bound at decode sizes, and the
indexer runs once per layer: profiling a single-token decode attributed
2.63 ms/step and 1638 kernels to this one function.

The whole thing is per-group and needs no cross-row communication, so it
collapses into one program per row: 128 channels, four groups of 32, all in
registers.

Bit-exact with the reference -- same fp32 arithmetic, same thresholds, same
LUT; ``tests/test_deepseek_v4_indexer.py`` pins it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mxfp4_fake_quant_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    row_width: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,   # row_width padded to a power of two
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    offs = tl.arange(0, BLOCK)
    mask = offs < row_width
    x = tl.load(x_ptr + row * row_width + offs, mask=mask, other=0.0).to(tl.float32)

    # Per-group E8M0 scale: the smallest power of two that maps the group's
    # amax onto E2M1's largest magnitude (6.0).
    g = tl.reshape(x, (BLOCK // GROUP, GROUP))
    amax = tl.max(tl.abs(g), axis=1)
    amax = tl.maximum(amax, 6.0 * tl.exp2(-126.0))
    scale = tl.exp2(tl.ceil(tl.log2(amax / 6.0)))
    scaled = g / scale[:, None]

    # E2M1 magnitudes are [0, .5, 1, 1.5, 2, 3, 4, 6]; the code is how many
    # midpoints the magnitude clears.
    m = tl.abs(scaled)
    code = tl.zeros(m.shape, dtype=tl.float32)
    code += (m > 0.25).to(tl.float32)
    code += (m > 0.75).to(tl.float32)
    code += (m > 1.25).to(tl.float32)
    code += (m > 1.75).to(tl.float32)
    code += (m > 2.5).to(tl.float32)
    code += (m > 3.5).to(tl.float32)
    code += (m > 5.0).to(tl.float32)

    low = code * 0.5
    # The quadratic maps codes 5/6/7 exactly onto magnitudes 3/4/6.
    high = (code * code - 9.0 * code + 26.0) * 0.5
    mag = tl.where(code <= 4.0, low, high)

    # ``torch.sign`` is zero at zero, and so is the quantized magnitude there.
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    q = tl.reshape(mag * sign * scale[:, None], (BLOCK,))
    tl.store(out_ptr + row * row_width + offs, q, mask=mask)


def mxfp4_fake_quantize_fused(
    x: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    """Drop-in fused replacement for the reference ``mxfp4_fake_quantize``."""
    shape = x.shape
    width = shape[-1]
    flat = x.reshape(-1, width)
    out = torch.empty_like(flat)
    block = triton.next_power_of_2(width)
    _mxfp4_fake_quant_kernel[(flat.shape[0],)](
        flat,
        out,
        flat.shape[0],
        row_width=width,
        GROUP=group_size,
        BLOCK=block,
        num_warps=4,
    )
    return out.reshape(shape)


__all__ = ["mxfp4_fake_quantize_fused"]
