import pytest
import torch

from gllm.layers import sampler


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_torch_fallback_top_p_and_unbounded_limits(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.0, 0.0, 1.0]], device=device)
    result = sampler._top_k_top_p_torch(
        probs, torch.tensor([3, 0], device=device), torch.tensor([0.5, 0.0], device=device)
    )
    assert result.tolist() == [1, 2]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("probabilities,k,p,expected", [
    ([0.5, 0.25, 0.125, 0.125], 2, 0.6, [2 / 3, 1 / 3, 0, 0]),
    ([0.25, 0.25, 0.25, 0.25], 2, 1.0, [0.25] * 4),
    ([0.5, 0.25, 0.25], 3, 0.5, [1, 0, 0]),
    ([1 / 129] * 129, 20, 0.1, [1 / 129] * 129),
    ([0.5, 0.25, 0.25], 2, 1.0, [0.5, 0.25, 0.25]),
    ([0, 0, 1], 0, 0.0, [0, 0, 1]),
], ids=["joint", "top-k-ties", "top-p-equality", "wide-ties",
        "partial-ties", "unbounded"])
def test_torch_fallback_joint_probabilities(
    monkeypatch, device, probabilities, k, p, expected,
):
    from gllm.layers import sampler

    captured = []
    def sample(weights, num_samples):
        assert num_samples == 1
        captured.append(weights.clone())
        return weights.argmax(dim=-1, keepdim=True)

    # Inspect the actual distribution, independently of random draw variance.
    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(sampler, "top_k_top_p_sampling_from_probs", None)
    probs = torch.tensor([probabilities], device=device, dtype=torch.float32)
    result = sampler._fused_top_k_top_p_sample(
        probs, torch.tensor([k], device=device), torch.tensor([p], device=device),
    )
    reference = torch.tensor([expected], device=device, dtype=probs.dtype)
    torch.testing.assert_close(captured[0], reference)
    torch.testing.assert_close(captured[0].sum(dim=-1), torch.ones(1, device=device))
    assert torch.equal(captured[0] > 0, reference > 0)
    assert result.dtype == torch.int64
    assert result.item() == reference.argmax(dim=-1).item()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_fused_sampling_preserves_flashinfer_delegation(monkeypatch, device):
    from gllm.layers import sampler

    calls = []
    def native(probs, top_ks, top_ps, *, filter_apply_order):
        calls.append((probs, top_ks, top_ps, filter_apply_order))
        return torch.tensor([1], device=device, dtype=torch.int32)

    monkeypatch.setattr(sampler, "top_k_top_p_sampling_from_probs", native)
    probs = torch.tensor([[0.25, 0.75]], device=device)
    ks = torch.tensor([2], device=device)
    ps = torch.tensor([0.9], device=device)
    result = sampler._fused_top_k_top_p_sample(probs, ks, ps)
    assert len(calls) == 1 and calls[0][3] == "joint"
    torch.testing.assert_close(calls[0][0], probs)
    torch.testing.assert_close(calls[0][1], ks.to(torch.int32))
    torch.testing.assert_close(calls[0][2], ps)
    assert result.dtype == torch.int64 and result.tolist() == [1]
