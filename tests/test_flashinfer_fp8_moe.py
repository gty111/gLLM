"""TRTLLM FP8 MoE: expert-parallel routing, numerics and graph replay."""

from types import SimpleNamespace

import pytest
import torch

from gllm.layers.moe.flashinfer_trtllm import fp8_moe_support_reason
from gllm.layers.moe.fused_moe_triton.layer import Fp8MoEMethod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "rows,experts,local,hidden,intermediate",
    [(1, 8, 4, 256, 256), (32, 512, 128, 2560, 640), (128, 512, 128, 2560, 640)],
)
@torch.inference_mode()
def test_fp8_moe_matches_triton_and_replays(rows, experts, local, hidden, intermediate):
    torch.manual_seed(37)
    device = "cuda"
    method = Fp8MoEMethod(
        {"weight_block_size": [128, 128], "activation_scheme": "dynamic"}
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.randn(local, 2 * intermediate, hidden, device=device).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.randn(local, hidden, intermediate, device=device).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        ),
        w13_weight_scale_inv=torch.nn.Parameter(
            torch.full(
                (local, 2 * intermediate // 128, hidden // 128), 0.02, device=device
            ),
            requires_grad=False,
        ),
        w2_weight_scale_inv=torch.nn.Parameter(
            torch.full(
                (local, hidden // 128, intermediate // 128), 0.02, device=device
            ),
            requires_grad=False,
        ),
        global_num_experts=experts,
        local_num_experts=local,
        top_k=min(10, experts - 1),
        ep_rank=1,
        activation="silu",
        apply_router_weight_on_input=False,
        scoring_func="softmax",
        renormalize=True,
        use_grouped_topk=False,
        e_score_correction_bias=None,
        intermediate_size_per_partition=intermediate,
        inplace=False,
    )
    layer.w13_weight_scale_inv.uniform_(0.01, 0.03)
    layer.w2_weight_scale_inv.uniform_(0.01, 0.03)
    reason = fp8_moe_support_reason(layer, [128, 128])
    if reason:
        pytest.skip(reason)
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    logits = torch.randn(rows, experts, device=device)
    expert_map = torch.full((experts,), -1, dtype=torch.int32, device=device)
    expert_map[local : 2 * local] = torch.arange(
        local, device=device, dtype=torch.int32
    )

    def apply():
        return method.apply(
            layer,
            x,
            logits,
            layer.top_k,
            True,
            global_num_experts=experts,
            expert_map=expert_map,
        )

    probabilities = logits.float().softmax(-1)
    weights, ids = probabilities.topk(layer.top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    oracle = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(local):
        router_weight = (weights * (ids == local + expert)).sum(-1, keepdim=True)
        w13 = layer.w13_weight[expert].float() * layer.w13_weight_scale_inv[
            expert
        ].repeat_interleave(128, 0).repeat_interleave(128, 1)
        w2 = layer.w2_weight[expert].float() * layer.w2_weight_scale_inv[
            expert
        ].repeat_interleave(128, 0).repeat_interleave(128, 1)
        projected = x.float() @ w13.t()
        gate, up = projected.chunk(2, dim=-1)
        oracle += (torch.nn.functional.silu(gate) * up) @ w2.t() * router_weight
    method._weights_processed = True
    reference = apply().clone()
    method._weights_processed = False
    method.process_weights_after_loading(layer)
    assert method.backend == "flashinfer_trtllm_fp8"
    result = apply()
    assert torch.isfinite(result).all()
    error = (result.float() - reference.float()).square().mean().sqrt()
    rms = reference.float().square().mean().sqrt()
    oracle_rms = oracle.square().mean().sqrt().clamp_min(1e-8)
    triton_error = (reference.float() - oracle).square().mean().sqrt() / oracle_rms
    trtllm_error = (result.float() - oracle).square().mean().sqrt() / oracle_rms
    print(
        f"FP8 rows={rows}: relative RMSE triton={triton_error.item():.6f}, trtllm={trtllm_error.item():.6f}, between={(error / rms).item():.6f}",
        flush=True,
    )
    # Quantizing activations at both GEMMs introduces independent error in
    # each backend. Bound both against the FP32 equation, rather than treating
    # another quantized backend as an exact oracle.
    assert triton_error < 0.07 and trtllm_error < 0.07
    assert error / rms.clamp_min(1e-8) < 0.07

    for _ in range(3):
        apply()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply()
    x.normal_()
    logits.normal_()
    graph.replay()
    torch.testing.assert_close(captured, apply(), atol=0, rtol=0)
