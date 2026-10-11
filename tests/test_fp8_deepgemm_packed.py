"""DeepGEMM's Blackwell packed-UE8M0 block-FP8 path.

On SM100 ``fp8_gemm_nt`` runs the ``gemm_1d1d`` kernel, which needs per-row
scales packed four to an int32. These tests pin the two things that would
otherwise fail silently: that the packed path agrees numerically with the
Triton fallback, and that the derived weight scales are built once rather than
on every forward.
"""

import pytest
import torch

from gllm.layers.quantization.fp8 import (
    _PACKED_SCALE_ATTR,
    deepgemm_packed_available,
    fp8LinearMethod,
    packed_weight_scale,
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)

BLOCK = [128, 128]

packed_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not deepgemm_packed_available(),
    reason="requires a GPU with DeepGEMM's packed-UE8M0 backend",
)


def _weights(N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) / 8).to(
        torch.float8_e4m3fn
    )
    raw = torch.rand(N // 128, K // 128, device="cuda", dtype=torch.float32, generator=g)
    # checkpoint weight scales for scale_fmt="ue8m0" are powers of two
    return w, torch.exp2(torch.ceil(torch.log2(raw + 0.5)))


@packed_only
@pytest.mark.parametrize("M", [1, 7, 16, 32])
@pytest.mark.parametrize("N,K", [(1024, 4096), (8192, 1024), (512, 4096), (4096, 2048)])
def test_packed_matches_triton(M, N, K):
    """The packed kernel must agree with the Triton fallback it replaces."""
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w, ws = _weights(N, K)

    got = fp8LinearMethod(x, w, BLOCK, ws, round_scale=True)

    q, s = per_token_group_quant_fp8(
        x, 128, column_major_scales=False, round_scale=True
    )
    want = w8a8_block_fp8_matmul(q, w, s, ws, BLOCK, torch.bfloat16)

    # Both consume identical UE8M0 scales; they differ only in FP8 accumulation
    # order, which is worth about one bf16 ULP.
    torch.testing.assert_close(got, want, rtol=8e-3, atol=8e-3)


@packed_only
def test_packed_weight_scale_is_built_once():
    """Building the packed scales per forward would undo the speedup."""
    _, ws = _weights(1024, 4096)
    assert getattr(ws, _PACKED_SCALE_ATTR, None) is None

    first = packed_weight_scale(ws, 1024, 128)
    assert packed_weight_scale(ws, 1024, 128) is first
    assert getattr(ws, _PACKED_SCALE_ATTR, None) is first
    assert first.dtype == torch.int32


@packed_only
def test_packed_weight_scale_covers_partial_last_block():
    """A merged projection may end on a row count below the block size."""
    N, K = 1024 + 64, 4096
    ws = torch.ones((N + 127) // 128, K // 128, device="cuda", dtype=torch.float32)
    packed = packed_weight_scale(ws, N, 128)
    # mn-major: the packed tensor is indexed by row, so it must not overrun N.
    assert packed.shape[0] == N


def test_gate_requires_ue8m0_scales():
    """Without UE8M0 activation scales the packed path would silently reround."""
    import inspect

    from gllm.layers.quantization import fp8

    src = inspect.getsource(fp8.fp8LinearMethod)
    assert "round_scale and deepgemm_packed_available()" in src
