"""FlashInfer's SM100 groupwise FP8 GEMM matches the Triton block-FP8 path."""
import pytest
import torch

from gllm.layers.quantization import fp8


def _quant_weight(w):
    n, k = w.shape
    blk = w.view(n // 128, 128, k // 128, 128).float()
    scale = blk.abs().amax(dim=(1, 3)).clamp(min=1e-6) / 448.0
    q = (blk / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.contiguous()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8.flashinfer_groupwise_available(),
    reason="needs FlashInfer's SM100 groupwise FP8 GEMM",
)
@pytest.mark.parametrize("m", [1, 2, 7, 33, 137, 300, 2048])
def test_groupwise_matches_triton_block_fp8(m):
    torch.manual_seed(m)
    n, k = 1024, 2048
    wq, ws = _quant_weight(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    xq, xs = fp8.per_token_group_quant_fp8(x, 128)
    ref = fp8.w8a8_block_fp8_matmul(xq, wq, xs, ws, [128, 128], torch.bfloat16)
    out = fp8.w8a8_block_fp8_matmul_flashinfer_groupwise(
        xq, xs, wq, fp8._groupwise_weight_scale(ws)
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2 * ref.abs().max().item())
    # The transposed scale is built once and reused.
    assert fp8._groupwise_weight_scale(ws) is fp8._groupwise_weight_scale(ws)


def test_groupwise_unavailable_off_sm100(monkeypatch):
    monkeypatch.setattr(fp8, "_groupwise_probe", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert fp8.flashinfer_groupwise_available() is False
