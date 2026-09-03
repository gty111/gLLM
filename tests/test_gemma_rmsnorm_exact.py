"""Accuracy and internal consistency of the Gemma RMSNorm path.

This used to assert ``torch.equal`` against ``aten::mean``'s reduction tree,
which is what forced a hand-written CUDA kernel: reproducing that tree meant
staging every square in shared memory and reducing them in one warp. The
guarantee that actually matters is weaker -- all of gLLM's own paths agree with
each other, and every path stays within FP32-reference accuracy -- and dropping
the ``torch.mean`` half made the kernel 1.5-1.9x faster. These checks encode the
weaker guarantee: a tight bound against an FP32 reference, plus equality across
the shapes that dispatch differently (decode-height vs prefill-height rows, and
the strided gated-QKV view).
"""

import pytest
import torch

from gllm.layers.ops.gemma_rmsnorm import (
    gemma_fused_add_rms_norm,
    gemma_rms_norm,
)

# bf16 carries ~3 decimal digits, so a per-element deviation of a few ulps of
# the output's own scale is the floor any implementation lives at. The old
# aten-matching kernel measured 1.56e-2 by this metric at width 5120; anything
# materially worse is a real regression, not a reduction-order difference.
_TOLERANCE = 3e-2

_SHAPES = ((1, 128), (127, 128), (128, 2048), (513, 2048), (32, 5120), (4096, 5120))


def _reference(x, weight, eps):
    """FP32 ground truth, independent of any kernel under test."""
    wide = x.float()
    normalized = wide * torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * (weight.float() + 1.0)


def _deviation(actual, expected):
    """Max per-element error as a fraction of the reference's RMS."""
    return (
        (actual.float() - expected).abs().max()
        / expected.square().mean().sqrt()
    ).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows,width", _SHAPES)
@pytest.mark.parametrize("weight_dtype", (torch.bfloat16, torch.float32))
def test_matches_fp32_reference(rows, width, weight_dtype):
    torch.manual_seed(23)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=weight_dtype) * 0.05

    actual = torch.empty_like(x)
    gemma_rms_norm(actual, x, weight, 1e-6)
    assert _deviation(actual, _reference(x, weight, 1e-6)) < _TOLERANCE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows,width", _SHAPES)
def test_fused_add_matches_fp32_reference(rows, width):
    torch.manual_seed(29)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.05

    folded = (residual.float() + x.float()).to(x.dtype)
    expected = _reference(folded, weight, 1e-6)

    got_input, got_residual = x.clone(), residual.clone()
    gemma_fused_add_rms_norm(got_input, got_residual, weight, 1e-6)

    assert torch.equal(got_residual, folded), "residual fold must be exact"
    assert _deviation(got_input, expected) < _TOLERANCE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_row_count_does_not_change_the_result():
    """Decode-height and prefill-height rows must agree.

    The dispatch used to send <=256 rows to a CUDA kernel and taller inputs to
    an aten path, so prefill and decode computed the norm differently and were
    held together only by the bitwise guarantee. One implementation makes the
    agreement structural; this test is what keeps it that way.
    """
    torch.manual_seed(31)
    width = 5120
    x = torch.randn(512, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.05

    tall = torch.empty_like(x)
    gemma_rms_norm(tall, x, weight, 1e-6)

    short = torch.empty_like(x)
    for start in range(0, 512, 32):
        chunk = x[start : start + 32]
        out = torch.empty_like(chunk)
        gemma_rms_norm(out, chunk, weight, 1e-6)
        short[start : start + 32] = out

    assert torch.equal(tall, short)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_strided_gated_qkv_view_matches_a_contiguous_copy():
    """Qwen3.5 hands Q/K in as a [token, head, dim] slice of a fused buffer.

    Its feature dimension is contiguous but the token stride spans the
    neighbouring gate/K/V fields, and reshaping it would materialize a copy of
    the activations -- so the kernel addresses it by stride instead.
    """
    torch.manual_seed(37)
    tokens, heads, dim = 24, 8, 128
    fused = torch.randn(tokens, heads * 4, dim, device="cuda", dtype=torch.bfloat16)
    view = fused[:, :heads, :]
    assert not view.is_contiguous()
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.05

    # ``out`` follows ``input``'s shape; the strided view is addressed in place.
    strided_out = torch.empty_like(view)
    gemma_rms_norm(strided_out, view, weight, 1e-6)

    packed = view.contiguous()
    packed_out = torch.empty_like(packed)
    gemma_rms_norm(packed_out, packed, weight, 1e-6)

    assert torch.equal(strided_out, packed_out)
