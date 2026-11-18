"""DSpark serving behavior: backends, request defaults, and which batches
speculate versus decode without speculation."""

from types import SimpleNamespace

import pytest

from gllm.engine.llm import LLM
from gllm.runtime.config import EngineConfig
from gllm.runtime.id_allocator import IDAllocator


def _dspark_config(**overrides):
    values = dict(
        model_path="/tmp/target", speculative_backend="dspark",
        dspark_model_path="/tmp/draft", maxd=1, maxp=16,
        overlap_scheduling=False, enable_prefix_caching=False,
        disable_cuda_graph=True, piecewise_cuda_graph=False,
        attention_backend="flashinfer",
    )
    values.update(overrides)
    return EngineConfig(**values)


@pytest.mark.parametrize("backend", ["fa3", "flashinfer", "fa4"])
def test_dspark_accepts_explicit_validated_backends(backend):
    assert _dspark_config(attention_backend=backend).attention_backend == backend


def _llm(speculative_backend, temperature=1.0, top_k=20, repetition_penalty=None):
    llm = LLM.__new__(LLM)
    llm.config = SimpleNamespace(speculative_backend=speculative_backend)
    llm.generation_config = SimpleNamespace(
        temperature=temperature, top_p=0.95, top_k=top_k, repetition_penalty=repetition_penalty
    )
    llm.model_max_length = 1024
    llm.finish_tokens = [2]
    llm.id_allocator = IDAllocator(0, 99)
    return llm


def test_dspark_keeps_checkpoint_sampling_defaults():
    seq = _llm("dspark", repetition_penalty=1.05).allocate_seq([1, 2, 3], output_len=8)
    assert (seq.temperature, seq.top_k, seq.repetition_penalty) == (1.0, 20, 1.05)


@pytest.mark.parametrize("seqs,expected", [
    ([dict(temperature=0, top_k=20)], True),
    ([dict(temperature=0, top_k=20), dict(temperature=0.7, top_k=20)], True),
    ([dict(temperature=1.0, top_k=-1)], True),
    ([dict(temperature=0, top_k=20, logprobs_enabled=True)], False),
])
def test_dspark_speculates_sampled_batches(seqs, expected):
    """Sampled rows use rejection sampling; only logprobs skip speculation."""
    from gllm.speculative.mtp import MtpMixin

    batch = [SimpleNamespace(**{"logprobs_enabled": False, **seq}) for seq in seqs]
    assert MtpMixin.mtp_sampling_compatible(batch) is expected
