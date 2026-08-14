import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="FlashInfer TRT-LLM MXINT4 MoE requires the SM100 family",
)


def _checkpoint_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = weight.float().reshape(-1, 32)
    upper = blocks.max(dim=-1, keepdim=True).values * (8.0 / 7.0)
    lower = blocks.min(dim=-1, keepdim=True).values
    scales = torch.where(upper > -lower, upper, -lower) / 8.0
    signed = (blocks / scales).round().clamp(-8, 7).to(torch.int8)

    signed = signed.reshape_as(weight)
    biased = (signed.to(torch.int32) + 8).reshape(
        *weight.shape[:-1], weight.shape[-1] // 8, 8
    )
    shifts = torch.arange(8, device=weight.device, dtype=torch.int32) * 4
    packed = torch.sum(biased << shifts, dim=-1, dtype=torch.int32)
    scales = scales.to(torch.bfloat16).reshape(
        *weight.shape[:-1], weight.shape[-1] // 32
    )
    dequantized = (
        signed.float().reshape(*weight.shape[:-1], -1, 32)
        * scales.float().unsqueeze(-1)
    ).reshape_as(weight)
    return packed, scales, dequantized


@torch.inference_mode()
def test_flashinfer_mxint4_checkpoint_conversion_and_moe():
    from gllm import _custom_ops as ops

    torch.manual_seed(11)
    experts, hidden, intermediate = 4, 256, 256
    tokens, top_k = 2, 2

    w13 = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    q13, s13, dq13 = _checkpoint_quantize(w13)
    q2, s2, dq2 = _checkpoint_quantize(w2)

    q13, s13 = ops.prepare_flashinfer_mxint4_moe_weight(
        q13, s13, gated=True
    )
    q2, s2 = ops.prepare_flashinfer_mxint4_moe_weight(q2, s2, gated=False)

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.2
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    bias = torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1
    output = ops.flashinfer_mxint4_moe(
        x,
        logits,
        q13,
        s13,
        q2,
        s2,
        global_num_experts=experts,
        local_num_experts=experts,
        local_expert_offset=0,
        top_k=top_k,
        intermediate_size=intermediate,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        correction_bias=bias,
    )

    scores = logits.sigmoid()
    topk_ids = (scores + bias).topk(top_k, dim=-1).indices
    topk_weights = scores.gather(1, topk_ids)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    reference = torch.zeros_like(output)
    for token in range(tokens):
        for choice in range(top_k):
            expert = int(topk_ids[token, choice])
            gate_up = x[token].float() @ dq13[expert].float().T
            gate, up = gate_up.chunk(2)
            activated = torch.nn.functional.silu(gate) * up
            expert_output = activated @ dq2[expert].float().T
            reference[token] += (
                topk_weights[token, choice] * expert_output
            ).to(reference.dtype)

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=0.08, atol=5e-4)
