"""Speculative penalties preserve causal history and rejected-tail isolation."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from gllm.layers.repetition_penalty import apply_scaling_penalties
from gllm.runtime.memory_manager import MemoryManager
from gllm.runtime.model_runner import ModelRunner
from gllm.runtime.sequence import GenerationSequence
from gllm.speculative.repetition_penalty import (
    SpeculativeRepetitionPenalty, apply_speculative_penalties,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_penalty_allows_mtp_but_logprobs_still_fall_back():
    seqs = [SimpleNamespace(repetition_penalty=p) for p in (1.0, 1.05, 0.8)]
    assert ModelRunner.mtp_sampling_compatible(seqs)
    seqs[1].logprobs_enabled = True
    assert not ModelRunner.mtp_sampling_compatible(seqs)


def test_neutral_penalties_do_not_touch_manager():
    assert SpeculativeRepetitionPenalty.prepare(object(), [SimpleNamespace(repetition_penalty=1.0)]) is None


@cuda
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [1, 4, 8])
def test_verify_matches_ordinary_sampling_for_each_causal_prefix(dtype, width):
    batch, vocab = 4, 1301
    values = torch.tensor([1.05, 0.8, 1.0, 1.5], device='cuda', dtype=dtype)
    history = torch.ones(batch, vocab, device='cuda', dtype=dtype)
    history[:, [3, 7, 11]] = values[:, None]
    # Includes repeated and unseen tokens; a future token must not affect an
    # earlier row. Strided row layout also models a view into padded buffers.
    candidates = torch.tensor([[17, 17, 9, 23, 11, 23, 3, 29]] * batch, device='cuda')[:, :width]
    torch.manual_seed(17)
    storage = torch.randn(batch * width, vocab + 7, device='cuda', dtype=dtype)
    logits = storage[:, :vocab]
    logits[:, 0] = 0
    logits[:, 1] = -float('inf')
    original = logits.clone()
    expected = original.clone()
    for row in range(batch):
        mask = history[row:row+1].clone()
        for pos in range(width):
            mask[0, candidates[row, pos]] = values[row]
            apply_scaling_penalties(expected[row*width+pos:row*width+pos+1], mask)
    saved = history.clone()
    apply_speculative_penalties(logits, history, values, candidates)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(history, saved, rtol=0, atol=0)
    # Both argmax and sampling distributions must see the transformed logits.
    assert torch.equal(logits.argmax(-1), expected.argmax(-1))
    torch.testing.assert_close(logits.float().softmax(-1), expected.float().softmax(-1))


def _manager(vocab=64, capacity=2):
    manager = MemoryManager.__new__(MemoryManager)
    manager.dtype = torch.bfloat16
    manager.vocab_size = vocab
    manager.max_running_seqs = capacity
    manager._rep_pool = None
    manager._rep_free_slots = None
    return manager


@cuda
@pytest.mark.parametrize('accepted', [0, 1, 3])
def test_only_accepted_prefix_survives_async_placeholders_and_slot_reuse(accepted):
    manager = _manager()
    seq = GenerationSequence(1, [2, 7, 7], [], output_len=16, repetition_penalty=1.05)
    state = SpeculativeRepetitionPenalty.prepare(manager, [seq])
    candidates = torch.tensor([[13, 17, 17, 23]], device='cuda')
    count = torch.tensor([accepted], device='cuda')
    state.commit(candidates, count)
    # The successor can start before CPU finalization; placeholders must not
    # become vocabulary indices or cause accepted GPU tokens to be forgotten.
    seq.token_ids.extend([-1, -2, -3, -4])
    successor = SpeculativeRepetitionPenalty.prepare(manager, [seq])
    expected = torch.ones_like(successor.history)
    ids = [2, 7] + [13, 17, 17, 23][:accepted+1]
    expected[0, ids] = state.values[0]
    torch.testing.assert_close(successor.history, expected, rtol=0, atol=0)
    seq.token_ids[3:] = [13, 17, 17, 23][:accepted+1]
    compacted = SpeculativeRepetitionPenalty.prepare(manager, [seq])
    torch.testing.assert_close(compacted.history, expected, rtol=0, atol=0)
    manager.free_rep_slot(seq)
    # Force reuse of the old slot, as after request completion or preemption.
    manager._rep_free_slots = deque([int(state.slots[0].item())])
    replacement = GenerationSequence(2, [31], [], output_len=16, repetition_penalty=0.8)
    reused = SpeculativeRepetitionPenalty.prepare(manager, [replacement])
    expected.fill_(1)
    expected[0, 31] = reused.values[0]
    torch.testing.assert_close(reused.history, expected, rtol=0, atol=0)


@cuda
def test_pool_growth_remapping_and_neutral_rows():
    manager = _manager(capacity=1)
    a = GenerationSequence(1, [3], [], output_len=16, repetition_penalty=1.05)
    neutral = GenerationSequence(2, [7], [], output_len=16, repetition_penalty=1.0)
    state = SpeculativeRepetitionPenalty.prepare(manager, [a, neutral])
    old_pool = manager._rep_pool
    b = GenerationSequence(3, [11], [], output_len=16, repetition_penalty=0.8)
    SpeculativeRepetitionPenalty.prepare(manager, [b])  # Mixed prefill grows pool.
    assert manager._rep_pool.data_ptr() != old_pool.data_ptr()
    state.commit(torch.tensor([[13, 17], [19, 23]], device='cuda'), torch.tensor([1, 1], device='cuda'))
    reordered = SpeculativeRepetitionPenalty.prepare(manager, [b, neutral, a])
    assert reordered.history[2, 17] == state.values[0]
    assert reordered.history[0, 17] == 1
    assert torch.all(reordered.history[1] == 1)
    assert torch.all(manager._rep_pool[0] == 1)


@cuda
def test_draft_graph_updates_private_history_and_resets_between_chains():
    history = torch.ones(2, 64, device='cuda')
    history[0, 3] = 1.5
    values = torch.tensor([1.5, 1.0], device='cuda')
    tokens = torch.tensor([[7], [9]], device='cuda')
    source = torch.linspace(-4, 4, 64, device='cuda').repeat(2, 1)
    logits = source.clone()
    scratch = history.clone()
    def step():
        logits.copy_(source)
        apply_speculative_penalties(logits, scratch, values, tokens, update_history=True)
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    scratch.copy_(history)
    graph.replay()
    tokens.copy_(torch.tensor([[11], [13]], device='cuda'))
    graph.replay()
    expected = source.clone()
    mask = history.clone()
    mask[0, [7, 11]] = 1.5
    apply_scaling_penalties(expected, mask)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert history[0, 7] == 1  # Speculative draft tokens never enter committed pool.
    scratch.copy_(history)
    graph.replay()
    mask[0, 7] = 1
    expected.copy_(source)
    apply_scaling_penalties(expected, mask)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize('sampled', [False, True])
def test_eager_draft_uses_penalized_causal_distribution(sampled):
    source = torch.tensor([[6., 5., 4., 3.5]], device='cuda')
    history = torch.tensor([[2., 1., 1., 1.]], device='cuda')
    state = SimpleNamespace(history=history, values=torch.tensor([2.], device='cuda'))
    mtp = SimpleNamespace(
        forward=lambda inputs, hidden, token: hidden,
        logits_from_hidden=lambda hidden: source.clone(),
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(mtp=mtp), input_data=None, _mtp_penalties=state,
        prepare_input=lambda seqs: None, _prepare_attention_metadata=lambda inputs: None,
        _mtp_bcast_tp=lambda token: token,
        _mtp_probs_from_logits=lambda logits, seqs: logits.softmax(-1),
        _q_dense=ModelRunner._q_dense,
    )
    args = ([SimpleNamespace()], [[0]], [1], torch.zeros(1, 2, device='cuda'))
    if sampled:
        drafts, q = ModelRunner._draft_chain_eager_sampled(
            runner, *args, k=3, nd=1, gen=torch.Generator(device='cuda').manual_seed(7),
        )
        mask = history.clone()
        for pos, token in enumerate([1] + drafts[0][:-1]):
            mask[0, token] = 2
            expected = source.clone()
            apply_scaling_penalties(expected, mask)
            torch.testing.assert_close(q.dense[:, pos], expected.softmax(-1))
    else:
        drafts = ModelRunner._draft_chain_eager(runner, *args, k=3, nd=1)
        assert drafts == [[2, 3, 0]]
    torch.testing.assert_close(history, torch.tensor([[2., 1., 1., 1.]], device='cuda'))
