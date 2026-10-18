"""Block-scaled FP8 storage for the DeepSeek-V4 sliding-window KV cache.

The window cache is 85% of V4's KV footprint (43 layers x 512 latent dims x
2 bytes = 43 KB per token), and it is storing values that are *already* FP8:
:func:`~gllm.layers.attention.deepseek_v4.ops.fp8_fake_quantize_inplace`
rounds the 448 NoPE dims through E4M3 with a power-of-two scale per 64-wide
group before anything is cached.  Keeping the E4M3 codes and their exponents
instead of the dequantized BF16 is therefore free of precision loss and saves
43% of the bank.

Losslessness is exact, not approximate.  A stored value is ``code * scale``
where ``code`` has at most three mantissa bits (E4M3) and ``scale`` is a power
of two, so it is representable in BF16 with no rounding.  Re-deriving the scale
from those values either recovers it exactly (when the group maximum exceeds
half the E4M3 range) or halves it, in which case every code doubles -- exactly,
since doubling only changes the exponent -- and the product is unchanged.

Layout, matching vLLM's ``fp8_ds_mla`` so a future native-packed kernel can
read it directly (584 bytes per token per layer):

    [0    : 448)  NoPE, E4M3 codes
    [448  : 576)  RoPE, BF16 verbatim (never quantized -- it is positional)
    [576  : 584)  seven UE8M0 group exponents, one padding byte
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Per-group width of V4's own KV quantization. Changing it here without
# changing ``fp8_fake_quantize_inplace`` would silently re-quantize.
GROUP_SIZE = 64
_E4M3_MAX = 448.0
_QUANT_EPS = 1e-10


def supports_packed_layout(head_dim: int, rope_dim: int) -> bool:
    """Can this geometry be packed at all?

    Real V4 checkpoints are 512/64 -> 448 NoPE dims, seven whole groups. Only
    the synthetic geometries used by unit tests fail this.
    """
    nope = head_dim - rope_dim
    return nope > 0 and nope % GROUP_SIZE == 0


def raw_row_bytes(head_dim: int, rope_dim: int) -> int:
    """Packed byte width of one cached token row."""
    nope = head_dim - rope_dim
    if not supports_packed_layout(head_dim, rope_dim):
        raise ValueError(
            f"V4 FP8 KV needs a NoPE width divisible by {GROUP_SIZE}, got {nope}"
        )
    groups = nope // GROUP_SIZE
    # One scale byte per group, rounded up to a multiple of 8 so the row width
    # stays 8-byte aligned.
    scale_bytes = (groups + 7) // 8 * 8
    return nope + rope_dim * 2 + scale_bytes


@triton.jit
def _pack_kernel(
    kv_ptr,
    out_ptr,
    offset_ptr,
    kv_row_stride,
    eps,
    fp8_max,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
):
    token = tl.program_id(0)
    src = kv_ptr + token.to(tl.int64) * kv_row_stride
    dst = out_ptr + tl.load(offset_ptr + token).to(tl.int64)

    for group in tl.range(0, GROUPS):
        cols = group * GROUP + tl.arange(0, GROUP)
        x = tl.load(src + cols).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x)), eps)
        # Power-of-two scale: the exponent is all UE8M0 can store, and it is
        # what ``fp8_fake_quantize_inplace(round_scale=True)`` already used.
        exponent = tl.ceil(tl.log2(absmax / fp8_max))
        scale = tl.exp2(exponent)
        code = tl.clamp(x / scale, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(dst + cols, code.to(tl.uint8, bitcast=True))
        tl.store(
            dst + SCALE_OFFSET + group,
            (exponent.to(tl.int32) + 127).to(tl.uint8),
        )

    # RoPE dims carry position, never pass through the model's QAT, and are
    # copied as raw BF16 bytes.
    rope = tl.arange(0, ROPE)
    values = tl.load(src + NOPE + rope)
    raw = values.to(tl.uint16, bitcast=True)
    tl.store(dst + NOPE + 2 * rope, (raw & 0xFF).to(tl.uint8))
    tl.store(dst + NOPE + 2 * rope + 1, (raw >> 8).to(tl.uint8))


@triton.jit
def _gather_kernel(
    cache_ptr,
    offset_ptr,
    out_ptr,
    out_row_stride,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
):
    token = tl.program_id(0)
    src = cache_ptr + tl.load(offset_ptr + token).to(tl.int64)
    dst = out_ptr + token.to(tl.int64) * out_row_stride

    for group in tl.range(0, GROUPS):
        cols = group * GROUP + tl.arange(0, GROUP)
        code = tl.load(src + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        exponent = tl.load(src + SCALE_OFFSET + group).to(tl.int32)
        scale = tl.exp2((exponent - 127).to(tl.float32))
        tl.store(dst + cols, (code * scale).to(tl.bfloat16))

    rope = tl.arange(0, ROPE)
    low = tl.load(src + NOPE + 2 * rope).to(tl.uint16)
    high = tl.load(src + NOPE + 2 * rope + 1).to(tl.uint16)
    tl.store(dst + NOPE + rope, ((high << 8) | low).to(tl.bfloat16, bitcast=True))


def pack_raw_fp8(
    kv: torch.Tensor,
    cache: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    rope_dim: int,
) -> None:
    """Quantize ``kv`` rows into ``cache`` at ``row_offsets``.

    ``kv`` is ``[tokens, head_dim]`` BF16 straight out of the projection.
    ``row_offsets`` gives each row's start as a flat element offset into
    ``cache``, so this stays agnostic to how the bank is laid out.
    """
    if kv.dtype is not torch.bfloat16:
        raise TypeError(f"V4 KV packing expects bfloat16, got {kv.dtype}")
    if cache.dtype is not torch.uint8:
        raise TypeError(f"V4 packed KV bank must be uint8, got {cache.dtype}")
    if kv.stride(-1) != 1:
        kv = kv.contiguous()
    tokens, head_dim = kv.shape
    nope = head_dim - rope_dim
    groups = nope // GROUP_SIZE
    if tokens == 0:
        return
    if row_offsets.numel() != tokens:
        raise ValueError((row_offsets.numel(), tokens))
    _pack_kernel[(tokens,)](
        kv,
        cache,
        row_offsets,
        kv.stride(0),
        _QUANT_EPS,
        _E4M3_MAX,
        NOPE=nope,
        ROPE=rope_dim,
        GROUP=GROUP_SIZE,
        GROUPS=groups,
        SCALE_OFFSET=nope + rope_dim * 2,
        num_warps=4,
    )


def gather_raw_fp8(
    cache: torch.Tensor,
    row_offsets: torch.Tensor,
    *,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Dequantize the packed rows at ``row_offsets`` into BF16.

    Returns ``[*row_offsets.shape, head_dim]``.  Both sparse-attention kernels
    V4 dispatches to take BF16 KV, so the bank is unpacked on the way into
    their workspace -- which still cuts the bytes read from HBM by 43%.
    """
    if cache.dtype is not torch.uint8:
        raise TypeError(f"V4 packed KV bank must be uint8, got {cache.dtype}")
    shape = tuple(row_offsets.shape)
    flat = row_offsets.reshape(-1).to(torch.int64)
    out = torch.empty(
        (flat.numel(), head_dim), dtype=torch.bfloat16, device=cache.device
    )
    if flat.numel():
        nope = head_dim - rope_dim
        _gather_kernel[(flat.numel(),)](
            cache,
            flat,
            out,
            out.stride(0),
            NOPE=nope,
            ROPE=rope_dim,
            GROUP=GROUP_SIZE,
            GROUPS=nope // GROUP_SIZE,
            SCALE_OFFSET=nope + rope_dim * 2,
            num_warps=4,
        )
    return out.view(*shape, head_dim)


__all__ = [
    "GROUP_SIZE",
    "gather_raw_fp8",
    "pack_raw_fp8",
    "raw_row_bytes",
    "supports_packed_layout",
]
