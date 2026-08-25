"""Exactness checks for the launch-reduced Gemma RMSNorm path."""

import pytest
import torch

from gllm.layers.ops.gemma_rmsnorm import (
    gemma_fused_add_rms_norm_reference_reduction,
    gemma_rms_norm_reference_reduction,
)


def _reference(out, x, weight, eps):
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(
        x_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    out.copy_((normalized * (weight.float() + 1.0)).to(x.dtype))


def _check_exactness():
    torch.manual_seed(23)
    for rows, width in (
        (1, 128),
        (127, 128),
        (128, 2048),
        (513, 2048),
        (32, 5120),
    ):
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        for weight_dtype in (torch.bfloat16, torch.float32):
            weight = (
                torch.randn(width, device="cuda", dtype=weight_dtype) * 0.05
            )
            expected = torch.empty_like(x)
            actual = torch.empty_like(x)
            _reference(expected, x, weight, 1e-6)
            gemma_rms_norm_reference_reduction(actual, x, weight, 1e-6)
            assert torch.equal(actual, expected), (rows, width, weight_dtype)

            residual = torch.randn_like(x)
            expected_residual = residual.clone()
            expected_residual.add_(x)
            _reference(expected, expected_residual, weight, 1e-6)
            actual_residual = residual.clone()
            actual_input = x.clone()
            gemma_fused_add_rms_norm_reference_reduction(
                actual_input, actual_residual, weight, 1e-6
            )
            assert torch.equal(actual_residual, expected_residual)
            assert torch.equal(actual_input, expected), (
                rows,
                width,
                weight_dtype,
            )

    for input_dtype in (torch.float16, torch.float32):
        x = torch.randn(17, 128, device="cuda", dtype=input_dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(128, device="cuda", dtype=input_dtype) * 0.05
        expected_residual = residual.clone()
        expected_residual.add_(x)
        expected = torch.empty_like(x)
        _reference(expected, expected_residual, weight, 1e-6)
        actual_input = x.clone()
        actual_residual = residual.clone()
        gemma_fused_add_rms_norm_reference_reduction(
            actual_input, actual_residual, weight, 1e-6
        )
        assert torch.equal(actual_residual, expected_residual), input_dtype
        assert torch.equal(actual_input, expected), input_dtype

    # Qwen3.5 gated attention lays Q/K out as strided [token, head, dim]
    # views of a fused Q/gate/K/V projection.  The optimized path must read
    # those strides directly and remain bitwise identical to the compact
    # reference that it replaces.
    for tokens, heads in ((16, 24), (257, 4), (341, 24)):
        width = 256
        trailing = 2 * heads * width + 2 * 4 * width
        qkv = torch.randn(
            tokens, trailing, device="cuda", dtype=torch.bfloat16
        )
        q_gate = qkv[:, : 2 * heads * width].view(tokens, heads, 2 * width)
        q, _ = torch.chunk(q_gate, 2, dim=-1)
        weight = torch.randn(width, device="cuda", dtype=torch.float32) * 0.05
        expected = torch.empty_like(q)
        actual = torch.empty_like(q)
        _reference(expected, q, weight, 1e-6)
        gemma_rms_norm_reference_reduction(actual, q, weight, 1e-6)
        assert not q.is_contiguous()
        assert actual.is_contiguous()
        assert torch.equal(actual, expected), (tokens, heads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gemma_rms_norm_is_bitwise_exact():
    _check_exactness()


def main():
    _check_exactness()

    print("Gemma RMSNorm optimized path exactly matches the FP32 reference")


if __name__ == "__main__":
    main()
