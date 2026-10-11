from types import SimpleNamespace

import pytest
import torch

from gllm.layers.moe.deepseek_v4 import (
    DeepseekV4MoE,
    DeepseekV4SharedExpert,
    deepseek_v4_swiglu,
)
from gllm.layers.moe.mxfp4_experts import deepgemm_mxfp4_moe
from gllm.layers.moe.topk import deepseek_v4_topk


FP8_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
}


def test_deepseek_v4_swiglu_matches_reference_order():
    gate_up = torch.tensor(
        [[20.0, -20.0, 3.0, 15.0, -15.0, 2.0]], dtype=torch.bfloat16
    )
    actual = deepseek_v4_swiglu(gate_up, limit=10.0)
    gate, up = gate_up.float().chunk(2, dim=-1)
    expected = (
        torch.nn.functional.silu(gate.clamp(max=10.0))
        * up.clamp(min=-10.0, max=10.0)
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _load_fp8_linear(linear, weight):
    from deep_gemm.utils import per_block_cast_to_fp8

    quantized, scale = per_block_cast_to_fp8(
        weight, use_ue8m0=True, gran_k=128
    )
    linear.weight.data.copy_(quantized)
    linear.weight_scale_inv.data.copy_(scale)
    return quantized, scale


def _load_mxfp4_experts(module, w13, w2):
    from deep_gemm.utils import per_token_cast_to_fp4

    w13_q, w13_s, w2_q, w2_s = [], [], [], []
    for expert in range(w13.shape[0]):
        q, s = per_token_cast_to_fp4(w13[expert], use_ue8m0=True, gran_k=32)
        w13_q.append(q.view(torch.uint8))
        w13_s.append(s.to(torch.float8_e8m0fnu).view(torch.uint8))
        q, s = per_token_cast_to_fp4(w2[expert], use_ue8m0=True, gran_k=32)
        w2_q.append(q.view(torch.uint8))
        w2_s.append(s.to(torch.float8_e8m0fnu).view(torch.uint8))
    raw = tuple(map(torch.stack, (w13_q, w13_s, w2_q, w2_s)))
    module.w13_weight.data.copy_(raw[0])
    module.w13_scale.data.copy_(raw[1])
    module.w2_weight.data.copy_(raw[2])
    module.w2_scale.data.copy_(raw[3])
    return raw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_shared_expert_uses_native_fp8_order():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native block FP8 path requires Blackwell")
    pytest.importorskip("deep_gemm")
    from deep_gemm.utils import per_token_cast_to_fp8

    torch.manual_seed(23)
    h, i, m = 512, 256, 32
    module = DeepseekV4SharedExpert(
        h, i, quant_config=FP8_CONFIG, swiglu_limit=10.0
    )
    w13 = torch.randn(2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(h, i, device="cuda", dtype=torch.bfloat16) * 0.05
    w13_q, w13_s = _load_fp8_linear(module.gate_up_proj, w13)
    w2_q, w2_s = _load_fp8_linear(module.down_proj, w2)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16) * 0.2

    def oracle_linear(a, weight, scale):
        aq, a_scale = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
        a_dq = aq.float() * a_scale.repeat_interleave(128, dim=1)
        w_dq = weight.float() * scale.repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        return (a_dq @ w_dq.T).to(torch.bfloat16)

    gate_up = oracle_linear(x, w13_q, w13_s)
    expected = oracle_linear(
        deepseek_v4_swiglu(gate_up, limit=10.0), w2_q, w2_s
    )
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_moe_combines_native_routes_and_shared_branch():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native FP8-by-FP4 path requires Blackwell")
    pytest.importorskip("deep_gemm")

    torch.manual_seed(29)
    h, i, m, e = 512, 256, 32, 3
    config = SimpleNamespace(
        hidden_size=h,
        moe_intermediate_size=i,
        n_routed_experts=e,
        num_experts_per_tok=2,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hash_layers=0,
        vocab_size=128,
        quantization_config=FP8_CONFIG,
    )
    module = DeepseekV4MoE(3, config)
    module.gate.weight.data.normal_(std=0.05)
    module.e_score_correction_bias.data.normal_(std=0.01)
    w13 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.05
    raw_w13, raw_s13, raw_w2, raw_s2 = _load_mxfp4_experts(
        module.experts, w13, w2
    )
    shared_w13 = torch.randn(
        2 * i, h, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    shared_w2 = torch.randn(h, i, device="cuda", dtype=torch.bfloat16) * 0.05
    _load_fp8_linear(module.shared_experts.gate_up_proj, shared_w13)
    _load_fp8_linear(module.shared_experts.down_proj, shared_w2)
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16) * 0.2

    logits = torch.nn.functional.linear(x.float(), module.gate.weight.float())
    route, ids = deepseek_v4_topk(
        logits,
        2,
        renormalize=True,
        routed_scaling_factor=1.5,
        correction_bias=module.e_score_correction_bias,
    )
    expected_routed = deepgemm_mxfp4_moe(
        x, raw_w13, raw_w2, raw_s13, raw_s2, route, ids
    )
    expected = expected_routed + module.shared_experts(x)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)
