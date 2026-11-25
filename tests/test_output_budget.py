"""Output limits use actual prompt lengths."""

from types import SimpleNamespace

import pytest
import torch

from gllm.engine.llm import LLM
from gllm.layers import sampler as sampler_module
from gllm.runtime.id_allocator import IDAllocator
from gllm.runtime.input_data import InputData
from gllm.runtime.sequence import GenerationSequence


def make_engine(context=262144):
    engine = LLM.__new__(LLM)
    engine.model_max_length = context
    engine.model_path = "test"
    engine.generation_config = SimpleNamespace(
        temperature=None, top_p=None, top_k=None, repetition_penalty=None,
    )
    engine.finish_tokens = [99]
    engine.id_allocator = IDAllocator()
    return engine


@pytest.mark.parametrize("requested,configured,expected", [
    (None, 20, 20),
    (1, 20, 1),
    (5, 20, 5),
    (None, None, 1),
    (None, 1, 1),
])
def test_engine_top_k_sampling_defaults(requested, configured, expected):
    from gllm.layers.sampler import Sampler

    engine = make_engine()
    engine.generation_config.top_k = configured
    seq = engine.allocate_seq([1, 2], top_k=requested)
    assert seq.top_k == expected
    flags = Sampler._get_sampling_flags(SimpleNamespace(seqs=[seq]))
    assert flags["is_all_greedy"] == (expected == 1)


def test_engine_zero_temperature_overrides_model_sampling_defaults():
    from gllm.layers.sampler import Sampler

    engine = make_engine()
    engine.generation_config.temperature = 1.0
    engine.generation_config.top_k = 20
    engine.generation_config.top_p = 0.95
    seq = engine.allocate_seq([1, 2], temperature=0)
    assert seq.temperature == 0
    assert seq.top_k == 1
    assert Sampler._get_sampling_flags(SimpleNamespace(seqs=[seq]))["is_all_greedy"]


@pytest.mark.parametrize("prompt_len,requested,expected", [
    (180000, None, 82144),
    (262143, None, 1),
    (180000, 16384, 16384),
    (180000, 82144, 82144),
])
def test_engine_resolves_budget(prompt_len, requested, expected):
    engine = make_engine()
    tokens = [1] * prompt_len
    assert engine.check_seq_length(tokens, requested)
    seq = engine.allocate_seq(tokens, requested)
    assert seq.output_len == expected
    assert seq.requested_output_len == requested


@pytest.mark.parametrize("prompt_len,requested", [
    (262144, None), (262145, None), (262144, 1),
    (180000, 82145), (1, 0), (1, -1),
])
def test_invalid_budget_is_rejected_before_id_allocation(prompt_len, requested):
    engine = make_engine()
    tokens = [1] * prompt_len
    assert not engine.check_seq_length(tokens, requested)
    with pytest.raises(ValueError):
        engine.allocate_seq(tokens, requested)
    assert engine.id_allocator.get_num_used_ids() == 0


def test_default_budget_survives_cache_progress_and_preemption():
    engine = make_engine(context=8)
    seq = engine.allocate_seq([1, 2, 3, 4])
    seq.computed_token_num = 3  # Prefix hits must not enlarge the budget.
    assert seq.output_len == 4
    seq.append(10)
    seq.preempt()
    assert seq.raw_prompt_len == 4
    assert seq.prompt_len == 5
    assert seq.output_len == 4
    seq.computed_token_num = 5
    seq.token_ids.extend([11, 12])
    assert not seq.is_finish
    seq.append(13)
    assert seq.is_finish
    # The budget, not a finish token, ended it.
    assert seq.token_ids[-1] not in seq.finish_tokens
    assert len(seq.token_ids) - seq.raw_prompt_len == seq.output_len


def test_default_budget_still_stops_at_eos():
    seq = make_engine(context=8192).allocate_seq([1, 2])
    seq.computed_token_num = 2
    seq.append(99)
    assert seq.is_finish
    assert seq.token_ids[-1] in seq.finish_tokens


@pytest.mark.parametrize("requested,expected", [(None, 2), (1, 1), (2, 2)])
def test_disagg_budget_uses_expanded_image_prompt(monkeypatch, requested, expected):
    from gllm.disagg.lm_manager import DisaggCoordinator, DisaggEvents, _PendingSeq
    from gllm.layers.rotary_embedding import MRotaryEmbedding
    from gllm.runtime.model_runner import ModelRunner

    seq = make_engine(context=8).allocate_seq([1, 126, 2], requested)
    pending = _PendingSeq(seq, [SimpleNamespace(
        item_idx=0, modality="image",
        meta=SimpleNamespace(num_tokens=4, grid_thw=(1, 2, 2), content_hash="image"),
        pages=[0],
    )])
    coordinator = DisaggCoordinator.__new__(DisaggCoordinator)
    coordinator.image_token_id = 126
    coordinator.video_token_id = 127
    coordinator.mr = SimpleNamespace(model=SimpleNamespace(
        config=None, get_mm_placeholder_token_ids=lambda: [126, 127],
    ))
    monkeypatch.setattr(ModelRunner, "_splice_mm_pad_ids", lambda *args: args[0])
    monkeypatch.setattr(MRotaryEmbedding, "get_input_positions", lambda **kwargs: (None, 0))
    events = DisaggEvents()
    coordinator._admit_meta_complete(pending, events)
    assert seq.token_ids == [1, 126, 126, 126, 126, 2]
    assert seq.raw_prompt_len == 6
    assert seq.output_len == expected
    assert len(events.admits) == 1


@pytest.mark.parametrize("requested,image_tokens", [(None, 6), (None, 7), (3, 4)])
def test_disagg_rejects_budget_that_no_longer_fits(requested, image_tokens):
    from gllm.disagg.lm_manager import DisaggCoordinator, DisaggEvents, _PendingSeq

    seq = make_engine(context=8).allocate_seq([1, 126, 2], requested)
    pending = _PendingSeq(seq, [SimpleNamespace(meta=SimpleNamespace(num_tokens=image_tokens))])
    coordinator = DisaggCoordinator.__new__(DisaggCoordinator)
    coordinator.image_token_id = 126
    coordinator.video_token_id = 127
    events = DisaggEvents()
    with pytest.raises(ValueError):
        coordinator._admit_meta_complete(pending, events)
    assert not events.admits
    assert not pending.admitted


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
