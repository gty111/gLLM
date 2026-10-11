"""Zero-temperature requests stay greedy with model defaults and mixed batches."""

from types import SimpleNamespace

import pytest
import torch

from gllm.layers import sampler as sampler_module
from gllm.runtime.input_data import InputData
from gllm.runtime.sequence import GenerationSequence


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_temperature_uses_argmax_without_sampling(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    seq = GenerationSequence(1, [1], [], output_len=4, temperature=0, top_k=20, top_p=0.95)
    assert seq.top_k == 1
    data = SimpleNamespace(seqs=[seq])
    logits = torch.tensor([[2.0, 5.0, 1.0]], device=device)

    def unexpected_sampling(*args):
        pytest.fail("Zero temperature must bypass random sampling")

    monkeypatch.setattr(sampler_module, "_fused_top_k_top_p_sample", unexpected_sampling)
    result, logprobs = sampler_module.Sampler().forward_gpu(
        logits, data, return_logprobs=True, num_logprobs=2
    )
    assert result.tolist() == [1]
    torch.testing.assert_close(logprobs[0], logits.float().log_softmax(-1)[:, 1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_batch_keeps_zero_temperature_rows_greedy(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    seqs = [
        GenerationSequence(1, [1], [], output_len=4, temperature=0, top_k=20, top_p=0),
        GenerationSequence(2, [1], [], output_len=4, temperature=0.5, top_k=3, top_p=0.7),
    ]
    data = SimpleNamespace(
        seqs=seqs,
        temperature=torch.tensor([1.0, 0.5], device=device),
        top_k=torch.tensor([seq.top_k for seq in seqs], device=device),
        top_p=torch.tensor([seq.top_p for seq in seqs], device=device),
    )
    # A tie tests that greedy rows use argmax rather than a top-k sampler's
    # handling of equal probabilities. The sampled row must retain its draw.
    logits = torch.tensor([[5.0, 5.0, 1.0], [1.0, 3.0, 2.0]], device=device)
    expected = torch.stack([logits[0], logits[1] / 0.5]).softmax(-1)

    def sample(probs, top_k, top_p):
        assert torch.isfinite(probs).all()
        torch.testing.assert_close(probs, expected)
        assert top_k.tolist() == [1, 3]
        torch.testing.assert_close(top_p, torch.tensor([1.0, 0.7], device=device))
        return torch.tensor([1, 0], device=device)

    monkeypatch.setattr(sampler_module, "_fused_top_k_top_p_sample", sample)
    result = sampler_module.Sampler().forward_gpu(logits, data)
    assert result.tolist() == [0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sample_metadata_normalizes_zero_temperature_before_gpu_staging():
    data = InputData.__new__(InputData)
    data.seqs = [
        SimpleNamespace(temperature=0, top_k=20, top_p=0.95),
        SimpleNamespace(temperature=0.5, top_k=20, top_p=0.95),
    ]
    data.memory_manager = SimpleNamespace(
        dtype=torch.float32, vocab_size=32, build_repetition_penalty_mask=lambda seqs: None
    )
    data.prepare_sample()
    assert data.top_k.tolist() == [1, 20]
    assert data.temperature.tolist() == [1.0, 0.5]
