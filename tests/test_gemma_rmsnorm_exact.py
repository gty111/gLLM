"""Exactness checks for the launch-reduced Gemma RMSNorm path."""

import torch

from gllm.layers.ops.gemma_rmsnorm import gemma_rms_norm_reference_reduction


def _reference(out, x, weight, eps):
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(
        x_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    out.copy_((normalized * (weight.float() + 1.0)).to(x.dtype))


def main():
    torch.manual_seed(23)
    for rows, width in ((1, 128), (127, 128), (128, 2048), (513, 2048)):
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
            actual_residual.add_(x)
            gemma_rms_norm_reference_reduction(
                actual, actual_residual, weight, 1e-6
            )
            assert torch.equal(actual_residual, expected_residual)
            assert torch.equal(actual, expected), (rows, width, weight_dtype)

    print("Gemma RMSNorm optimized path exactly matches the FP32 reference")


if __name__ == "__main__":
    main()
