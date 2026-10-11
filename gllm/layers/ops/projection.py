"""Small-batch split-K projection with a rounded scaled-SiLU epilogue."""

from functools import cache

import torch
import triton
import triton.language as tl


@cache
def _device_major(device):
    return torch.cuda.get_device_capability(device)[0]


def supports_fused_down(x, weight, rank, groups):
    # Dispatch only the supported device and measured shape. Larger token
    # counts and other architectures retain their existing GEMM implementation.
    return (
        weight is not None
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == 2
        and 0 < x.shape[0] <= 32
        and x.shape[1] == 10240
        and rank == 320
        and groups == 4
        and weight.shape == (336, 10240)
        and x.stride(-1) == 1
        and weight.is_contiguous()
        and _device_major(x.device) == 10
    )


@triton.jit
def _down_gemm(
    X,
    W,
    P,
    M,
    SX,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    blocks: tl.constexpr = triton.cdiv(K, SPLIT * BK)
    kk = part * blocks * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(blocks):
        k = kk + block * BK
        x = tl.load(
            X + rows[:, None] * SX + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(x, w, acc)
    tl.store(
        P + (part * M + rows[:, None]) * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _finish_down(
    P,
    L,
    I,
    M,
    N: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
    SPLIT: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    parts = tl.arange(0, SPLIT)
    values = tl.load(
        P + (parts[:, None] * M + row) * N + cols[None, :],
        cols[None, :] < N,
        other=0,
    )
    # Preserve both BF16 rounding points: GEMM output, then division by G.
    projected = tl.sum(values, axis=0).to(L.dtype.element_ty).to(tl.float32)
    scaled = (projected / G).to(L.dtype.element_ty).to(tl.float32)
    activated = scaled * tl.sigmoid(scaled)
    tl.store(L + row * R + cols, activated, cols < R)
    tl.store(I + row * G + cols - R, projected, (cols >= R) & (cols < N))


class DownProjectionWorkspace:
    """Own scratch and results for every captured batch size until model unload."""

    def __init__(self):
        self._buffers = {}

    def run(self, x, weight, rank, groups):
        m, k = x.shape
        n = rank + groups
        if (
            x.stride(-1) != 1
            or not weight.is_contiguous()
            or weight.shape[1] != k
            or weight.shape[0] < n
            or x.dtype != weight.dtype
        ):
            raise ValueError(
                "Fused down requires matching compact weight/input features"
            )
        key = (x.device, x.dtype, m, k, rank, groups)
        if key not in self._buffers:
            self._buffers[key] = (
                torch.empty((16, m, n), device=x.device, dtype=torch.float32),
                torch.empty((m, rank), device=x.device, dtype=x.dtype),
                torch.empty((m, groups), device=x.device, dtype=x.dtype),
            )
        partial, lowrank, injection = self._buffers[key]
        _down_gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32), 16)](
            x,
            weight,
            partial,
            m,
            x.stride(0),
            k,
            n,
            16,
            32,
            128,
            16,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
        _finish_down[(m, triton.cdiv(n, 128))](
            partial,
            lowrank,
            injection,
            m,
            n,
            rank,
            groups,
            16,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return lowrank, injection
