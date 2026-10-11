"""Fused HC operations preserve eager low-precision rounding boundaries."""

import pytest
import torch

from gllm.layers.ops.hyper_connection import (
    combine,
    gate_mix,
    grouped_norm,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "rows,hidden,groups", [(1, 2560, 4), (32, 2560, 4), (128, 2560, 4), (1024, 128, 3)]
)
@torch.inference_mode()
def test_fused_hyper_connection(rows, hidden, groups):
    torch.manual_seed(29)
    x = torch.randn((rows, hidden * groups), device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hidden * groups, device="cuda", dtype=x.dtype)
    z = x.float().reshape(rows, groups, hidden)
    expected = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).flatten(
        1
    ) * (1 + w.float())
    expected = expected.to(x.dtype)
    norm = grouped_norm(x, w, hidden, groups, 1e-6)
    torch.testing.assert_close(norm, expected, atol=0, rtol=0)
    gate = torch.randn_like(x)
    expected_mix = (norm * gate.sigmoid()).reshape(rows, groups, hidden).mean(1)
    torch.testing.assert_close(
        gate_mix(norm, gate, hidden, groups), expected_mix, atol=0, rtol=0
    )
    block = torch.randn((rows, hidden), device="cuda", dtype=x.dtype)
    logits = torch.randn((rows, groups), device="cuda", dtype=x.dtype)
    injection = 2 * (logits / groups).sigmoid()
    expected_combine = (
        x.reshape(rows, groups, hidden) + block[:, None] * injection[:, :, None]
    ).flatten(1)
    torch.testing.assert_close(
        combine(x, block, logits, hidden, groups), expected_combine, atol=0, rtol=0
    )


@pytest.mark.parametrize("rows", [1, 32, 128, 256, 1024])
@torch.inference_mode()
def test_packed_projections_preserve_checkpoint_and_graph(rows):
    from torch.nn import functional as F

    from gllm.layers.hyper_connection import GatedResidual

    torch.manual_seed(17)
    module = GatedResidual(2560, 4, 320, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    norm = module.hc_norm(x)
    down = F.linear(norm, module.input_mix_weight_down.weight)
    injection = F.linear(norm, module.block_inject_weight.weight)
    expected = gate_mix(norm, module.input_mix_weight_up(F.silu(down / 4)), 2560, 4)
    keys = set(module.state_dict())
    module.process_weights_after_loading()
    mixed, residual = module.mix(x)
    torch.testing.assert_close(mixed, expected, atol=0.008, rtol=0.015)
    torch.testing.assert_close(residual[1], injection, atol=0.004, rtol=0.015)
    assert set(module.state_dict()) == keys
    assert module._mix_inject_weight.shape == (336, 10240)
    assert (
        module.input_mix_weight_down.weight.untyped_storage().data_ptr()
        == module._mix_inject_weight.untyped_storage().data_ptr()
    )
    block = torch.randn(rows, 2560, device="cuda", dtype=x.dtype)
    for _ in range(3):
        module.combine(block, module.mix(x)[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = module.combine(block, module.mix(x)[1])
    x.normal_()
    graph.replay()
    torch.testing.assert_close(
        result, module.combine(block, module.mix(x)[1]), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "rows,hidden,groups", [(1, 2560, 4), (32, 2560, 4), (128, 2560, 4), (1024, 128, 3)]
)
@torch.inference_mode()
def test_combine_norm_and_scaled_silu_rounding_and_graph(rows, hidden, groups):
    from torch.nn import functional as F

    from gllm.layers.ops.hyper_connection import combine_norm, scaled_silu

    torch.manual_seed(41)
    x = torch.randn(rows, hidden * groups, device="cuda", dtype=torch.bfloat16)
    block = torch.randn(rows, hidden, device="cuda", dtype=x.dtype)
    packed = torch.randn(rows, 336, device="cuda", dtype=x.dtype)
    injection = packed[:, 320 : 320 + groups]
    weight = torch.randn(hidden * groups, device="cuda", dtype=x.dtype)

    def run():
        return combine_norm(
            x, block, injection, weight, hidden, groups, 1e-6
        ), scaled_silu(packed[:, :320], groups)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (combined, normalized), activated = run()
    for _ in range(2):
        x.normal_()
        block.normal_()
        packed.normal_()
        graph.replay()
        expected = combine(x, block, injection, hidden, groups)
        torch.testing.assert_close(combined, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            normalized,
            grouped_norm(expected, weight, hidden, groups, 1e-6),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            activated, F.silu(packed[:, :320] / groups), atol=0, rtol=0
        )
