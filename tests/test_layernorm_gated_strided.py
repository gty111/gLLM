import pytest
import torch

from gllm.layers.ops.fla.layernorm_gated import rms_norm_gated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 64])
def test_rms_norm_gated_reads_strided_3d_gate_exactly(tokens):
    torch.manual_seed(2026 + tokens)
    heads, dim = 6, 128
    prefix = 5 * dim
    projection = torch.randn(
        tokens,
        prefix + heads * dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    z = projection[:, prefix:].view(tokens, heads, dim)
    x = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    assert z.stride()[1:] == (dim, 1)
    if tokens > 1:
        assert z.stride(0) == prefix + heads * dim
        assert not z.is_contiguous()
    reference = rms_norm_gated(
        x=x.reshape(-1, dim),
        weight=weight,
        bias=None,
        z=z.contiguous().reshape(-1, dim),
        eps=1e-6,
        is_rms_norm=True,
    ).view_as(x)
    actual = rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        is_rms_norm=True,
    )

    torch.cuda.synchronize()
    assert actual.is_contiguous()
    assert torch.equal(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rms_norm_gated_strided_3d_is_cuda_graph_safe():
    tokens, heads, dim, prefix = 4, 6, 128, 640
    projection = torch.randn(
        tokens,
        prefix + heads * dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    z = projection[:, prefix:].view(tokens, heads, dim)
    x = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    expected = rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-6,
        is_rms_norm=True,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rms_norm_gated(
            x=x,
            weight=weight,
            bias=None,
            z=z,
            eps=1e-6,
            is_rms_norm=True,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(captured, expected)
