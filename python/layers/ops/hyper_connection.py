"""Inference kernels for grouped normalization and gated residual streams."""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_norm(
    X, W, O, H: tl.constexpr, G: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    row, group = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    x = tl.load(X + (row * G + group) * H + d, d < H, 0).to(tl.float32)
    w = tl.load(W + group * H + d, d < H, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    tl.store(O + (row * G + group) * H + d, x * inv * (1 + w), d < H)


@triton.jit
def _gate_mix(
    X, Gate, O, H: tl.constexpr, G: tl.constexpr, BG: tl.constexpr, B: tl.constexpr
):
    row, tile = tl.program_id(0), tl.program_id(1)
    g = tl.arange(0, BG)
    d = tile * B + tl.arange(0, B)
    offsets = row * G * H + g[:, None] * H + d[None, :]
    mask = (g[:, None] < G) & (d[None, :] < H)
    x = tl.load(X + offsets, mask, 0)
    gate = tl.load(Gate + offsets, mask, 0).to(tl.float32)
    # Keep eager BF16/FP16 rounding at the sigmoid and product boundaries.
    weight = tl.sigmoid(gate).to(x.dtype)
    product = (x.to(tl.float32) * weight.to(tl.float32)).to(x.dtype)
    mixed = tl.sum(tl.where(mask, product.to(tl.float32), 0), 0) / G
    tl.store(O + row * H + d, mixed, d < H)


@triton.jit
def _combine(
    X,
    Block,
    Inject,
    O,
    INJECT_STRIDE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    B: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    g = tl.arange(0, BG)
    d = tile * B + tl.arange(0, B)
    logits = tl.load(Inject + row * INJECT_STRIDE + g, g < G, 0)
    scaled = (logits.to(tl.float32) / G).to(logits.dtype)
    injection = tl.sigmoid(scaled.to(tl.float32)).to(logits.dtype)
    injection = (2 * injection.to(tl.float32)).to(logits.dtype)
    block = tl.load(Block + row * H + d, d < H, 0)
    offsets = row * G * H + g[:, None] * H + d[None, :]
    mask = (g[:, None] < G) & (d[None, :] < H)
    streams = tl.load(X + offsets, mask, 0)
    product = (block[None, :].to(tl.float32) * injection[:, None].to(tl.float32)).to(
        streams.dtype
    )
    tl.store(O + offsets, streams.to(tl.float32) + product.to(tl.float32), mask)


@triton.jit
def _combine_norm(
    X,
    Block,
    Inject,
    W,
    O,
    N,
    IS: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    offsets = (row * G + group) * H + d
    x = tl.load(X + offsets, d < H, 0)
    block = tl.load(Block + row * H + d, d < H, 0)
    logits = tl.load(Inject + row * IS + group)
    scaled = (logits.to(tl.float32) / G).to(logits.dtype)
    injection = tl.sigmoid(scaled.to(tl.float32)).to(logits.dtype)
    injection = (2 * injection.to(tl.float32)).to(logits.dtype)
    product = (block.to(tl.float32) * injection.to(tl.float32)).to(x.dtype)
    # Materialize the canonical residual with the same rounding as combine.
    combined = (x.to(tl.float32) + product.to(tl.float32)).to(x.dtype)
    tl.store(O + offsets, combined, d < H)
    z = tl.where(d < H, combined.to(tl.float32), 0)
    inv = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
    w = tl.load(W + group * H + d, d < H, 0).to(tl.float32)
    tl.store(N + offsets, z * inv * (1 + w), d < H)


@triton.jit
def _scaled_silu(
    X, O, STRIDE: tl.constexpr, R: tl.constexpr, G: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + row * STRIDE + d, d < R, 0)
    scaled = (x.to(tl.float32) / G).to(x.dtype).to(tl.float32)
    tl.store(O + row * R + d, scaled * tl.sigmoid(scaled), d < R)


def scaled_silu(x, groups):
    rank = x.shape[-1]
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    rows = x.reshape(-1, rank)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    _scaled_silu[(x.numel() // rank,)](
        rows,
        output,
        rows.stride(0),
        rank,
        groups,
        triton.next_power_of_2(rank),
        enable_fp_fusion=False,
    )
    return output


def combine_norm(streams, block, logits, weight, hidden_size, groups, eps):
    streams, block = streams.contiguous(), block.contiguous()
    logits = logits.reshape(-1, groups)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    output, normalized = torch.empty_like(streams), torch.empty_like(streams)
    _combine_norm[(streams.numel() // (hidden_size * groups), groups)](
        streams,
        block,
        logits,
        weight,
        output,
        normalized,
        logits.stride(0),
        hidden_size,
        groups,
        eps,
        triton.next_power_of_2(hidden_size),
        enable_fp_fusion=False,
    )
    return output, normalized


def grouped_norm(x, weight, hidden_size, groups, eps):
    x = x.contiguous()
    output = torch.empty_like(x)
    _grouped_norm[(x.numel() // (hidden_size * groups), groups)](
        x,
        weight,
        output,
        hidden_size,
        groups,
        eps,
        triton.next_power_of_2(hidden_size),
        enable_fp_fusion=False,
    )
    return output


def gate_mix(x, gate, hidden_size, groups):
    x, gate = x.contiguous(), gate.contiguous()
    output = x.new_empty((*x.shape[:-1], hidden_size))
    rows = x.numel() // (hidden_size * groups)
    # Small row counts favor narrow CTAs; larger
    # rows favor wide tiles that avoid repeatedly scheduling the same streams.
    tile, warps = 256, 4
    if hidden_size == 2560 and groups == 4 and x.dtype == torch.bfloat16:
        tile, warps = (128, 8) if rows <= 16 else (1024, 4)
    _gate_mix[(rows, triton.cdiv(hidden_size, tile))](
        x,
        gate,
        output,
        hidden_size,
        groups,
        triton.next_power_of_2(groups),
        tile,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return output


def combine(streams, block, logits, hidden_size, groups):
    streams, block = streams.contiguous(), block.contiguous()
    logits = logits.reshape(-1, groups)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    output = torch.empty_like(streams)
    _combine[
        (streams.numel() // (hidden_size * groups), triton.cdiv(hidden_size, 256))
    ](
        streams,
        block,
        logits,
        output,
        logits.stride(0),
        hidden_size,
        groups,
        triton.next_power_of_2(groups),
        256,
        enable_fp_fusion=False,
    )
    return output
