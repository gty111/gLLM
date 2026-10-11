"""Fused HC operations preserve eager low-precision rounding boundaries."""

import pytest
import torch
import torch.nn.functional as F

from gllm.layers.ops.hyper_connection import (
    combine,
    gate_mix,
    grouped_norm,
)
from gllm.layers.ops.projection import DownProjectionWorkspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("rows", [1, 32, 128])
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    "rows,dtype,strided",
    [
        (1, torch.bfloat16, False),
        (7, torch.bfloat16, True),
        (16, torch.bfloat16, False),
        (23, torch.bfloat16, True),
        (32, torch.bfloat16, False),
        (7, torch.float16, True),
    ],
)
@torch.inference_mode()
def test_down_projection_fp64_oracle_and_graph(rows, dtype, strided):
    torch.manual_seed(177)
    width, rank, groups = 10240, 320, 4
    storage = torch.randn(
        rows, width + (17 if strided else 0), device="cuda", dtype=dtype
    )
    x = storage[:, :width]
    weight = torch.randn(336, width, device="cuda", dtype=dtype) * 0.01
    workspace = DownProjectionWorkspace()

    def check(lowrank, injection):
        # FP64 accumulation isolates the numerical effect of the GEMM reduction
        # order. Projection and division still round before the SiLU operation.
        projected = F.linear(x.double(), weight[: rank + groups].double()).to(dtype)
        expected = F.silu((projected[:, :rank] / groups).to(dtype))
        rtol = 0.015 if dtype == torch.bfloat16 else 0.003
        torch.testing.assert_close(lowrank, expected, atol=1e-4, rtol=rtol)
        torch.testing.assert_close(injection, projected[:, rank:], atol=1e-4, rtol=rtol)

    lowrank, injection = workspace.run(x, weight, rank, groups)
    check(lowrank, injection)
    pointers = (lowrank.data_ptr(), injection.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = workspace.run(x, weight, rank, groups)
    assert pointers == tuple(t.data_ptr() for t in captured)
    x.normal_()
    weight.normal_().mul_(0.01)
    graph.replay()
    check(*captured)
    # A different batch allocation must not release captured scratch/results.
    larger = torch.randn(64, width, device="cuda", dtype=dtype)
    workspace.run(larger, weight, rank, groups)
    x.zero_()
    graph.replay()
    torch.testing.assert_close(
        captured[0], torch.zeros_like(captured[0]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        captured[1], torch.zeros_like(captured[1]), atol=0, rtol=0
    )
