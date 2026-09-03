import pytest
import torch

from gllm.layers.layernorm import GemmaRMSNorm

# BF16 keeps 8 mantissa bits, so one ulp is 2**-8 relative. The norm's own
# FP32 reduction order is not required to match ``torch.mean``'s any more (see
# ``gllm/layers/ops/gemma_rmsnorm.py``), which moves the variance estimate by
# well under an ulp and the output by at most a couple. The previous
# ``atol=4e-3, rtol=0`` encoded the old bitwise guarantee -- it left room for
# exactly one ulp at magnitude 1 and none at all above it.
_RTOL = 2**-7
_ATOL = 2**-8


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return (
        x32
        * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
        * (weight.float() + 1.0)
    ).to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 37, 1024])
def test_gemma_rmsnorm_matches_fp32_offset(num_tokens: int):
    torch.manual_seed(7)
    x = torch.randn(num_tokens, 2048, device="cuda", dtype=torch.bfloat16)
    weight = 0.05 * torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    norm = GemmaRMSNorm(2048, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(weight)

    actual = norm(x)
    expected = _reference(x, weight, norm.variance_epsilon)
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_gemma_fused_add_rmsnorm_matches_fp32_offset():
    torch.manual_seed(11)
    x = torch.randn(37, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = 0.05 * torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    norm = GemmaRMSNorm(2048, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(weight)

    expected_residual = (residual + x).to(torch.bfloat16)
    expected = _reference(expected_residual, weight, norm.variance_epsilon)
    actual, actual_residual = norm(x.clone(), residual.clone())

    # The residual fold is still exact -- the kernel rounds once, to storage
    # precision, and publishes that value.
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)
