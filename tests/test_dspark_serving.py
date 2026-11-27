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


def _confidence_seq(seq_id, position=1):
    class Sequence:
        pass

    seq = Sequence()
    seq.__dict__.update(
        seq_id=seq_id, token_ids=[4] * position + [5],
        raw_prompt_len=position, output_len=2, finish_tokens=[2],
        ignore_eos=False, _cache_epoch=0, temperature=0.8,
        top_k=20, top_p=0.9, repetition_penalty=1.0,
    )
    return seq


def _complete_confidence(observer, seqs, accepted=None):
    counts = accepted if accepted is not None else [0] * len(seqs)
    rows = observer.begin(
        seqs, [len(s.token_ids) - 1 for s in seqs],
        [s.token_ids[-1] for s in seqs], 3, sampled=True, sparse=True,
    )
    scores = [[0.2, 0.4, -0.1] for _ in seqs]
    observer.complete(rows, scores, counts, [[5, 6, 7] for _ in seqs], [9] * len(seqs))
    return rows, scores


def test_dspark_confidence_records_align_identity_positions_and_raw_acceptance():
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    seqs = [_confidence_seq(5, 1), _confidence_seq(2, 8), _confidence_seq(9, 3)]
    _, scores = _complete_confidence(observer, seqs, [0, 1, 3])
    records = observer.drain()
    scores[0][0] = 100
    assert [r.request_identity[1] for r in records] == [5, 2, 9]
    assert [r.anchor_pos for r in records] == [1, 8, 3]
    assert [r.accepted_drafts for r in records] == [0, 1, 3]
    assert [r.conditional_labels for r in records] == [(0, None, None), (1, 0, None), (1, 1, 1)]
    assert records[1].prefix_labels == (1, 0, 0)
    assert records[2].remaining_tokens == 1  # raw A is not the publication budget
    assert records[0].confidence_logits == (0.2, 0.4, -0.1)
    assert all(r.candidate_indices == (0, 1, 2) and r.round_id == 1 for r in records)
    assert all(r.completed_ns >= r.started_ns for r in records)


@pytest.mark.parametrize("change", ["epoch", "sampling", "gap", "release"])
def test_dspark_confidence_history_requires_same_epoch_sampling_and_continuity(change):
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    seq = _confidence_seq(3)
    _complete_confidence(observer, [seq], [1])
    first = observer.drain()[0]
    seq.token_ids.extend([5, 9])  # expected next anchor position: 1+1+1
    if change == "epoch":
        seq._cache_epoch += 1
    elif change == "sampling":
        seq.top_p = 0.8
    elif change == "gap":
        seq.token_ids.append(9)  # an intervening ordinary decode
    else:
        observer.forget(seq)
    _complete_confidence(observer, [seq])
    second = observer.drain()[0]
    assert second.previous_round_id is None
    assert second.request_identity == first.request_identity
    assert second.round_id == 2


def test_dspark_confidence_history_links_only_completed_rounds():
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    seq = _confidence_seq(3)
    first = observer.begin([seq], [1], [5], 3, sampled=True, sparse=True)
    # A later begin must not use the still-incomplete first row as history.
    second = observer.begin([seq], [2], [9], 3, sampled=True, sparse=True)
    assert second[0].previous_round_id is None
    observer.complete(first, [[0., 0., 0.]], [0], [[5, 6, 7]], [9])
    seq.token_ids.append(9)
    third = observer.begin([seq], [2], [9], 3, sampled=True, sparse=True)
    assert third[0].previous_round_id == 1


def test_dspark_confidence_reused_id_and_old_release_keep_new_request_independent():
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    old = _confidence_seq(4)
    _complete_confidence(observer, [old])
    previous = observer.drain()[0]
    observer.forget(old)
    new = _confidence_seq(4)
    _complete_confidence(observer, [new])
    current = observer.drain()[0]
    observer.forget(old)
    new.token_ids.append(9)
    _complete_confidence(observer, [new])
    following = observer.drain()[0]
    assert previous.request_identity != current.request_identity
    assert current.previous_round_id is None
    assert following.previous_round_id == current.round_id
    assert observer.active_requests == 1


@pytest.mark.parametrize("change", ["release", "epoch"])
def test_dspark_confidence_stale_completion_is_not_published(change):
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    seq = _confidence_seq(1)
    rows = observer.begin([seq], [1], [5], 3, sampled=True, sparse=True)
    if change == "release":
        observer.forget(seq)
    else:
        seq._cache_epoch += 1
    observer.complete(rows, [[0., 0., 0.]], [3], [[5, 6, 7]], [9])
    assert observer.drain() == []
    assert observer.stale_records == 1


def test_dspark_confidence_capacity_and_weak_ownership_are_bounded():
    import gc
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test", capacity=2)
    seq = _confidence_seq(1)
    for _ in range(3):
        _complete_confidence(observer, [seq])
        seq.token_ids.append(9)
    assert [row.round_id for row in observer.drain()] == [2, 3]
    assert observer.dropped_records == 1
    del seq
    gc.collect()
    assert observer.active_requests == 0


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_dspark_confidence_rejects_invalid_capacity(capacity):
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    with pytest.raises(ValueError, match="capacity"):
        DSparkConfidenceObserver("worker-test", capacity=capacity)


@pytest.mark.parametrize("unsupported", ["overlap", "graph", "piecewise", "async", "mixed", "gpu_anchor", "mtp"])
def test_dspark_confidence_requires_explicit_strict_eager(unsupported):
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    config = SimpleNamespace(overlap_scheduling=False, disable_cuda_graph=True, piecewise_cuda_graph=False)
    runner = SimpleNamespace(config=config, dspark_enabled=True, _dspark_drafter=object())
    DSparkConfidenceObserver.check_runner(runner)
    if unsupported == "overlap":
        config.overlap_scheduling = True
    elif unsupported == "graph":
        config.disable_cuda_graph = False
    elif unsupported == "piecewise":
        config.piecewise_cuda_graph = True
    elif unsupported == "async":
        runner._mtp_async_publish = True
    elif unsupported == "mtp":
        runner.dspark_enabled = False
    with pytest.raises(ValueError, match="strict synchronous Eager"):
        DSparkConfidenceObserver.check_runner(
            runner, mixed=unsupported == "mixed", gpu_anchor=unsupported == "gpu_anchor"
        )


def test_dspark_confidence_completion_is_once_and_nonfinite_is_not_history():
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    seq = _confidence_seq(1)
    rows = observer.begin([seq], [1], [5], 3, sampled=True, sparse=True)
    observer.complete(rows, [[float("nan"), 0., 0.]], [1], [[5, 6, 7]], [9])
    with pytest.raises(RuntimeError, match="twice"):
        observer.complete(rows, [[0., 0., 0.]], [0], [[5, 6, 7]], [9])
    assert not observer.drain()[0].finite
    assert observer.invalid_records == 1
    seq.token_ids.extend([5, 9])
    assert observer.begin([seq], [3], [9], 3, sampled=True, sparse=True)[0].previous_round_id is None


def test_dspark_confidence_runner_free_invalidates_history_without_changing_release():
    from gllm.runtime.model_runner import ModelRunner
    from gllm.runtime.sequence import GenerationSequence
    from gllm.speculative.dspark_confidence import DSparkConfidenceObserver

    observer = DSparkConfidenceObserver("worker-test")
    seq = GenerationSequence(1, [4, 5], [2], output_len=16,
                             temperature=0.8, top_k=20, top_p=0.9)
    _complete_confidence(observer, [seq])
    first = observer.drain()[0]
    freed = []
    runner = SimpleNamespace(
        _dspark_confidence_observer=observer, _mtp_relay={1: object()}, use_mm=False,
        memory_manager=SimpleNamespace(free=lambda item: freed.append(item)),
    )
    ModelRunner.free(runner, seq)
    assert freed == [seq] and not runner._mtp_relay
    assert observer.active_requests == 0
    seq.preempt()
    _complete_confidence(observer, [seq])
    second = observer.drain()[0]
    assert second.request_identity == first.request_identity
    assert second.cache_epoch == first.cache_epoch + 1
    assert second.previous_round_id is None and second.round_id == 2
