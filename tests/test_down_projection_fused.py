"""Split-K projection preserves rounding and captured scratch ownership."""

import pytest
import torch
import torch.nn.functional as F

from gllm.layers.ops.projection import DownProjectionWorkspace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 7, 16, 23, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
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
