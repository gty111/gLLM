import pytest
import torch

from gllm.layers.moe.mxfp4_experts import NativeMXFP4Experts, deepgemm_mxfp4_moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_moe_matches_explicit_reference_loop():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native FP8-by-FP4 DeepGEMM requires Blackwell")

    pytest.importorskip("deep_gemm")
    from deep_gemm.utils import (
        cast_back_from_fp4,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    torch.manual_seed(17)
    tokens, hidden_size, intermediate_size = 32, 512, 256
    num_experts, topk = 3, 2
    x = torch.randn(
        tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    w13_bf16 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.05
    w2_bf16 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.05
    w13, s13, w2, s2 = [], [], [], []
    for expert in range(num_experts):
        q, s = per_token_cast_to_fp4(
            w13_bf16[expert], use_ue8m0=True, gran_k=32
        )
        w13.append(q)
        s13.append(s)
        q, s = per_token_cast_to_fp4(
            w2_bf16[expert], use_ue8m0=True, gran_k=32
        )
        w2.append(q)
        s2.append(s)
    w13, s13 = torch.stack(w13), torch.stack(s13)
    w2, s2 = torch.stack(w2), torch.stack(s2)
    ids = torch.stack(
        [
            torch.arange(tokens, device="cuda") % num_experts,
            (torch.arange(tokens, device="cuda") + 1) % num_experts,
        ],
        dim=1,
    ).to(torch.int32)
    route = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    route /= route.sum(dim=1, keepdim=True)

    actual = deepgemm_mxfp4_moe(
        x, w13, w2, s13, s2, route, ids
    ).float()

    def linear(a, weight, scale):
        aq, a_scale = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
        a_dq = aq.float() * a_scale.repeat_interleave(128, dim=1)
        w_dq = cast_back_from_fp4(weight, scale, gran_k=32)
        return a_dq @ w_dq.T

    expected = torch.zeros_like(actual)
    for expert in range(num_experts):
        token_idx, slot = torch.where(ids == expert)
        gate, up = linear(x[token_idx], w13[expert], s13[expert]).split(
            intermediate_size, dim=-1
        )
        mid = (
            torch.nn.functional.silu(gate.clamp(max=10.0))
            * up.clamp(min=-10.0, max=10.0)
            * route[token_idx, slot, None]
        ).to(torch.bfloat16)
        expected.index_add_(
            0, token_idx, linear(mid, w2[expert], s2[expert])
        )
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_mxfp4_experts_module_keeps_checkpoint_packing():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("native FP8-by-FP4 DeepGEMM requires Blackwell")
    pytest.importorskip("deep_gemm")
    from deep_gemm.utils import per_token_cast_to_fp4

    torch.manual_seed(19)
    experts = NativeMXFP4Experts(
        num_experts=2,
        hidden_size=512,
        intermediate_size=256,
    )
    w13_bf16 = torch.randn(
        2, 512, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    w2_bf16 = torch.randn(
        2, 512, 256, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    raw_w13, raw_s13, raw_w2, raw_s2 = [], [], [], []
    for expert in range(2):
        q, s = per_token_cast_to_fp4(
            w13_bf16[expert], use_ue8m0=True, gran_k=32
        )
        raw_w13.append(q.view(torch.uint8))
        raw_s13.append(s.to(torch.float8_e8m0fnu).view(torch.uint8))
        q, s = per_token_cast_to_fp4(
            w2_bf16[expert], use_ue8m0=True, gran_k=32
        )
        raw_w2.append(q.view(torch.uint8))
        raw_s2.append(s.to(torch.float8_e8m0fnu).view(torch.uint8))
    experts.w13_weight.data.copy_(torch.stack(raw_w13))
    experts.w2_weight.data.copy_(torch.stack(raw_w2))
    experts.w13_scale.data.copy_(torch.stack(raw_s13))
    experts.w2_scale.data.copy_(torch.stack(raw_s2))

    x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.stack(
        [
            torch.arange(16, device="cuda") % 2,
            (torch.arange(16, device="cuda") + 1) % 2,
        ],
        dim=1,
    ).to(torch.int32)
    weights = torch.rand(16, 2, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    expected = deepgemm_mxfp4_moe(
        x,
        torch.stack(raw_w13),
        torch.stack(raw_w2),
        torch.stack(raw_s13),
        torch.stack(raw_s2),
        weights,
        ids,
    )
    actual = experts(x, weights, ids)
    # The grouped FlashInfer kernel uses a fused implementation, so it is not
    # bitwise identical to the explicit per-expert oracle.  Keep the numerical
    # contract tight enough to catch layout, gate/up-order, or routing errors.
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)
    assert experts.w13_weight.dtype == torch.uint8
    assert experts.w2_weight.dtype == torch.uint8
    if experts._backend == "flashinfer":
        assert experts.w13_scale.dtype == torch.float8_e4m3fn
        assert experts.w2_scale.dtype == torch.float8_e4m3fn
    else:
        assert experts.w13_scale.dtype == torch.int32
        assert experts.w2_scale.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_mxfp4_experts_cuda_graph_replays_changed_routing():
    """A captured routed-MoE must not bake the warm-up expert ids.

    Full-model decode graphs are captured with synthetic tokens and replayed
    with the request's routing decisions.  Exercise that exact contract at the
    component boundary so a graph-unsafe FlashInfer kernel fails here instead
    of crashing the online worker on its first request.
    """
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("FlashInfer MXFP4 MoE requires Blackwell")
    pytest.importorskip("deep_gemm")
    pytest.importorskip("flashinfer.fused_moe")
    from deep_gemm.utils import per_token_cast_to_fp4

    torch.manual_seed(23)
    num_experts = 4
    experts = NativeMXFP4Experts(
        num_experts=num_experts,
        hidden_size=512,
        intermediate_size=256,
    )
    w13_bf16 = torch.randn(
        num_experts, 512, 512, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    w2_bf16 = torch.randn(
        num_experts, 512, 256, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    for expert in range(num_experts):
        q13, s13 = per_token_cast_to_fp4(
            w13_bf16[expert], use_ue8m0=True, gran_k=32
        )
        q2, s2 = per_token_cast_to_fp4(
            w2_bf16[expert], use_ue8m0=True, gran_k=32
        )
        experts.w13_weight.data[expert].copy_(q13.view(torch.uint8))
        experts.w13_scale.data[expert].copy_(
            s13.to(torch.float8_e8m0fnu).view(torch.uint8)
        )
        experts.w2_weight.data[expert].copy_(q2.view(torch.uint8))
        experts.w2_scale.data[expert].copy_(
            s2.to(torch.float8_e8m0fnu).view(torch.uint8)
        )
    experts.process_weights_after_loading()
    if experts._backend != "flashinfer":
        pytest.skip("installed stack did not select FlashInfer MXFP4 MoE")

    x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.float32)

    # Initialize all lazy kernel state before capture, as ModelRunner does.
    experts(x, weights, ids, reduce_results=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = experts(x, weights, ids, reduce_results=False)

    replay_x = torch.randn_like(x) * 0.2
    replay_ids = torch.tensor([[2, 3]], device="cuda", dtype=torch.int32)
    replay_weights = torch.tensor(
        [[0.25, 0.75]], device="cuda", dtype=torch.float32
    )
    x.copy_(replay_x)
    ids.copy_(replay_ids)
    weights.copy_(replay_weights)
    graph.replay()
    torch.cuda.synchronize()
    actual = graph_output.clone()
    expected = experts(
        replay_x, replay_weights, replay_ids, reduce_results=False
    )
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_mxfp4_expert_checkpoint_loader_fuses_w1_w3_without_unpacking():
    experts = NativeMXFP4Experts(
        num_experts=2,
        hidden_size=128,
        intermediate_size=64,
    )
    weights = {}
    for expert in range(2):
        base = f"layers.3.ffn.experts.{expert}"
        weights[f"{base}.w1.weight"] = torch.full(
            (64, 64), expert + 1, dtype=torch.int8
        )
        weights[f"{base}.w3.weight"] = torch.full(
            (64, 64), expert + 11, dtype=torch.int8
        )
        weights[f"{base}.w2.weight"] = torch.full(
            (128, 32), expert + 21, dtype=torch.int8
        )
        weights[f"{base}.w1.scale"] = torch.full(
            (64, 4), 127 + expert, dtype=torch.uint8
        )
        weights[f"{base}.w3.scale"] = torch.full(
            (64, 4), 129 + expert, dtype=torch.uint8
        )
        weights[f"{base}.w2.scale"] = torch.full(
            (128, 2), 131 + expert, dtype=torch.uint8
        )

    for field in experts._CHECKPOINT_FIELDS:
        experts.load_stacked_param(
            field, getattr(experts, field), weights, "layers.3.ffn.experts"
        )
    for expert in range(2):
        assert torch.all(experts.w13_weight[expert, :64] == expert + 1)
        assert torch.all(experts.w13_weight[expert, 64:] == expert + 11)
        assert torch.all(experts.w2_weight[expert] == expert + 21)
        assert torch.all(experts.w13_scale[expert, :64] == 127 + expert)
        assert torch.all(experts.w13_scale[expert, 64:] == 129 + expert)
        assert torch.all(experts.w2_scale[expert] == 131 + expert)
