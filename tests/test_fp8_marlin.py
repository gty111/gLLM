import pytest
import torch


def test_backend_dispatch_preserves_native_fp8(monkeypatch):
    from gllm.layers import linear
    from gllm.layers.quantization.fp8 import fp8LinearMethod
    from gllm.layers.quantization.fp8_marlin import FP8MarlinMethod

    layer = linear.LinearBase(256, 256, quant_config={"quant_method": "fp8"})
    layer.block_quant = True
    layer.weight_block_size = [128, 128]
    layer.weight_scale_inv = torch.ones(2, 2)
    layer.input_scale = None
    layer.use_ue8m0 = False
    for capability in (80, 86):
        monkeypatch.setattr(linear, "get_device_capability", lambda: capability)
        assert isinstance(layer.dispatch_quant_method(), FP8MarlinMethod)
    for capability in (89, 90, 100, 120):
        monkeypatch.setattr(linear, "get_device_capability", lambda: capability)
        assert layer.dispatch_quant_method().func is fp8LinearMethod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["RowParallelLinear", "ColumnParallelLinear", "ReplicatedLinear"])
@torch.inference_mode()
def test_linear_layer_prepares_marlin_after_loading(monkeypatch, kind):
    from gllm.layers import linear

    monkeypatch.setattr(linear, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tp_size", lambda: 1)
    monkeypatch.setattr(linear, "get_device_capability", lambda: 80)
    layer = getattr(linear, kind)(256, 256, params_dtype=torch.bfloat16, quant_config={
        "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
    })
    # Simulate a checkpoint load before invoking ModelLoader's post-load hook.
    layer.weight.copy_(torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn))
    layer.weight_scale_inv.fill_(0.02)
    layer.bias.normal_(0, 0.1)
    original = layer.weight.float() * 0.02
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(17, 256, dtype=torch.bfloat16, device="cuda")
    actual = layer(x)
    expected = (x.float() @ original.T).to(x.dtype) + layer.bias
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,k", [(128, 128), (192, 256), (512, 1024)])
@torch.inference_mode()
def test_marlin_matches_dequantized_reference(dtype, n, k):
    from gllm.layers.quantization.fp8_marlin import FP8MarlinMethod

    torch.manual_seed(123)
    layer = torch.nn.Module()
    layer.params_dtype = dtype
    layer.weight_block_size = [128, 128]
    w = (torch.randn(n, k, device="cuda") * 48).to(torch.float8_e4m3fn)
    s = torch.rand((n + 127) // 128, k // 128, device="cuda") * 0.015 + 0.005
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
    method = FP8MarlinMethod()
    method.process_weights_after_loading(layer)
    packed_ptr = layer.weight.data_ptr()
    method.process_weights_after_loading(layer)
    assert layer.weight.data_ptr() == packed_ptr
    assert layer.weight.dtype == torch.int32
    assert not hasattr(layer, "weight_scale_inv")
    reference_weight = w.float() * s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    bias = torch.randn(n, dtype=dtype, device="cuda")
    for m in (0, 1, 8, 17, 128, 8192):
        x = torch.randn(m, k, dtype=dtype, device="cuda")
        actual = method(x, layer.weight, bias)
        expected = (x.float() @ reference_weight.T).to(dtype) + bias
        assert actual.shape == expected.shape
        if m:
            error = (actual.float() - expected.float()).square().mean().sqrt()
            rms = expected.float().square().mean().sqrt()
            assert error / rms < (0.012 if dtype == torch.bfloat16 else 0.002)
    x = torch.randn(2, 4, k, dtype=dtype, device="cuda")
    eager = method(x, layer.weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = method(x, layer.weight)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, method(x, layer.weight), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k", [(16384, 5120), (5120, 6144), (34816, 5120)])
def test_qwen38_gdn_and_mlp_shapes(n, k):
    test_marlin_matches_dequantized_reference(torch.bfloat16, n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_marlin_compiles_fullgraph():
    from gllm.layers.quantization.fp8_marlin import FP8MarlinMethod

    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight_block_size = [128, 128]
    layer.weight = torch.nn.Parameter(torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(2, 2, device="cuda"), requires_grad=False)
    method = FP8MarlinMethod()
    method.process_weights_after_loading(layer)

    @torch.compile(fullgraph=True, dynamic=True)
    def run(x):
        return method(x, layer.weight)

    for m in (1, 8, 33):
        x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(run(x), method(x, layer.weight), rtol=0, atol=0)
