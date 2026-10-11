import pytest
import torch

from gllm.layers.quantization.mxfp4 import (
    deepgemm_mxfp4_expert,
    deepgemm_mxfp4_linear,
    e8m0_to_float32,
    prepare_mxfp4_scale,
)


def test_e8m0_to_float32_decodes_exponents():
    raw = torch.tensor([127, 128, 126], dtype=torch.uint8)
    actual = e8m0_to_float32(raw)
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0, 0.5]))


def test_e8m0_to_float32_rejects_non_e8m0():
    with pytest.raises(TypeError, match="MXFP4 scales"):
        e8m0_to_float32(torch.ones(2, dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepgemm_mxfp4_linear_matches_dequantized_oracle():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native FP8-by-FP4 DeepGEMM requires Blackwell")

    pytest.importorskip("deep_gemm")
    from deep_gemm.utils import (
        cast_back_from_fp4,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    torch.manual_seed(7)
    m, n, k = 128, 256, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w_q, w_scale = per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
    packed_scale = prepare_mxfp4_scale(
        w_scale, output_size=n, input_size=k
    )

    actual = deepgemm_mxfp4_linear(x, w_q, packed_scale).float()
    x_q, x_scale = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128)
    x_dequant = x_q.float() * x_scale.repeat_interleave(128, dim=1)
    w_dequant = cast_back_from_fp4(w_q, w_scale, gran_k=32)
    expected = x_dequant @ w_dequant.T

    # The operands and scaling recipe are identical.  The remaining error is
    # the native kernel's BF16 output rounding and accumulation order.
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepgemm_mxfp4_expert_matches_reference_order():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native FP8-by-FP4 DeepGEMM requires Blackwell")

    pytest.importorskip("deep_gemm")
    from deep_gemm.utils import (
        cast_back_from_fp4,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    torch.manual_seed(11)
    m, hidden_size, intermediate_size = 128, 512, 256
    x = torch.randn(m, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.2
    w13 = torch.randn(
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.05
    w2 = torch.randn(
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.05
    route = torch.rand(m, device="cuda", dtype=torch.float32)
    w13_q, w13_s = per_token_cast_to_fp4(w13, use_ue8m0=True, gran_k=32)
    w2_q, w2_s = per_token_cast_to_fp4(w2, use_ue8m0=True, gran_k=32)

    actual = deepgemm_mxfp4_expert(
        x, w13_q, w2_q, w13_s, w2_s, routing_weight=route
    ).float()

    def oracle_linear(a, w_q, w_s):
        a_q, a_s = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
        a_dq = a_q.float() * a_s.repeat_interleave(128, dim=1)
        w_dq = cast_back_from_fp4(w_q, w_s, gran_k=32)
        return a_dq @ w_dq.T

    gate, up = oracle_linear(x, w13_q, w13_s).split(
        intermediate_size, dim=-1
    )
    intermediate = (
        torch.nn.functional.silu(gate.clamp(max=10.0))
        * up.clamp(min=-10.0, max=10.0)
        * route[:, None]
    ).to(torch.bfloat16)
    expected = oracle_linear(intermediate, w2_q, w2_s)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)
