import pytest
import torch

from gllm.layers import sampler


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_fallback_returns_original_token_ids(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(sampler, "top_k_top_p_sampling_from_probs", None)
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]], device=device)
    result = sampler._fused_top_k_top_p_sample(
        probs, torch.tensor([1, 1], device=device), torch.tensor([1.0, 1.0], device=device)
    )
    assert result.dtype == torch.int64
    assert result.tolist() == [1, 0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_fallback_top_p_and_unbounded_limits(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.0, 0.0, 1.0]], device=device)
    result = sampler._top_k_top_p_torch(
        probs, torch.tensor([3, 0], device=device), torch.tensor([0.5, 0.0], device=device)
    )
    assert result.tolist() == [1, 2]
