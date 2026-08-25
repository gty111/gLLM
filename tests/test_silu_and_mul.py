import pytest
import torch
from flashinfer.activation import silu_and_mul as flashinfer_silu_and_mul

from gllm import _custom_ops as ops
from gllm.layers.ops.silu_and_mul import silu_and_mul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 32, 128])
def test_silu_and_mul_matches_previous_flashinfer_backend(rows):
    torch.manual_seed(13)
    width = 17_408
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    flashinfer_silu_and_mul(x, out=expected)
    silu_and_mul(actual, x)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_fast_path_rejects_non_bf16():
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    out = torch.empty(2, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="requires BF16"):
        silu_and_mul(out, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_supports_leading_dimensions():
    torch.manual_seed(19)
    x = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    flashinfer_silu_and_mul(x, out=expected)
    ops.silu_and_mul(actual, x)
    assert torch.equal(actual, expected)
