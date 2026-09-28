"""Sampled acceptance shares the GPU completion used by greedy overlap."""
from types import SimpleNamespace

import pytest
import torch

from gllm.speculative import mtp as mtp_module
from gllm.speculative.async_state import MtpAsyncCompletion
from gllm.speculative.mtp import MtpMixin, MtpQDist, MtpVerifyResult

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _runner(monkeypatch, *, asynchronous, sparse, tp_override=False):
    runner = MtpMixin()
    n, k, vocab, hidden = 3, 3, 8, 2
    seqs = [SimpleNamespace(
        seq_id=i, token_ids=[4, 5], computed_token_num=1,
        to_compute_token_num=1, to_compute_tokens=[5], temperature=1.0,
        top_k=2 if sparse else -1, repetition_penalty=1.05,
        ssm_block_table=[10*i+j for j in range(4)], recurrent_state_slot=10*i,
    ) for i in range(n)]
    runner.input_data = SimpleNamespace(num_decodes=n, seqs=seqs)
    runner.model = SimpleNamespace(mtp=object())
    runner.memory_manager = SimpleNamespace(
        vocab_size=vocab, ssm_segment=object(),
        pre_allocate_page_for_lengths=lambda *args: None,
    )
    host = torch.empty(n, dtype=torch.int64, pin_memory=True)
    runner._mtp_staging = SimpleNamespace(
        seed_hidden=torch.empty(n, hidden, device="cuda"),
        x1_host=host, x1_host_np=host.numpy(),
        x1_gpu=torch.empty(n, device="cuda", dtype=torch.int64),
        row_idx=torch.arange(n, device="cuda"),
        install_ctx_host=host.clone().pin_memory(),
    )
    runner._mtp_staging.install_ctx_host_np = runner._mtp_staging.install_ctx_host.numpy()
    runner._mtp_k = k
    runner.max_running_seqs = n
    runner._mtp_can_sample = True
    runner._mtp_sampling_seen = False
    runner._mtp_async_publish = asynchronous
    runner._mtp_async_state = None
    runner._mtp_prep_epoch = 0
    runner._mtp_relay = {}
    runner._mtp_rng = None
    runner._mtp_step = 0
    runner._mtp_draft_graph = True
    runner._draft_size_to_graph_sampled_sparse = {n: object()}
    runner._draft_size_to_graph_sampled = {n: object()}
    runner._record_mtp_metrics = lambda *args: None
    runner.sampler = SimpleNamespace(_structured=None)
    drafts = torch.ones(n, k, device="cuda", dtype=torch.int64)
    q = torch.zeros(n, k, vocab, device="cuda")
    q[:, :, 1] = 1
    # Accept zero, one and all drafts respectively; residual/bonus always 2.
    pred = torch.tensor([[2, 2, 2, 2], [1, 2, 2, 2], [1, 1, 1, 2]], device="cuda")
    logits = torch.full((n*4, vocab), -float("inf"), device="cuda")
    logits.scatter_(1, pred.reshape(-1, 1), 0)
    vhidden = torch.arange(n*4*hidden, device="cuda", dtype=torch.float32).reshape(n*4, hidden)

    def draft(*args, **kwargs):
        runner._drafts_gpu = drafts
        if sparse:
            vals, idx = q.topk(2, dim=-1)
            return None, MtpQDist(vals=vals, idx=idx, drawn=q[:, :, 1])
        return None, MtpQDist(dense=q)

    runner._draft_chain_graph = draft
    runner._mtp_verify_target = lambda **kwargs: MtpVerifyResult(logits.clone(), vhidden, None, [])
    runner._mtp_sample_params = lambda *args: (
        torch.ones(n, 1, device="cuda"), torch.full((n,), 2, device="cuda"),
        torch.ones(n, device="cuda"),
    )
    runner._mtp_probs_static = lambda logits, *args: logits.softmax(-1)
    runner._mtp_sparse_probs = lambda logits, *args: logits.softmax(-1).topk(2, dim=-1)
    runner.penalty_commits = []
    penalty = SimpleNamespace(
        verify=lambda *args: None,
        commit=lambda candidates, count: runner.penalty_commits.append((candidates.clone(), count.clone())),
    )
    monkeypatch.setattr(mtp_module.SpeculativeRepetitionPenalty, "prepare", lambda *args: penalty)
    monkeypatch.setattr(mtp_module, "get_tp_size", lambda: 2 if tp_override else 1)
    if tp_override:
        # Emulate rank zero overriding this rank's independently drawn result.
        decisions = torch.tensor([[2, 0, 1], [3, 3, 3]], device="cuda")
        runner._mtp_bcast_tp = lambda value: value.copy_(decisions)
    return runner, seqs, vhidden


@cuda
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("tp_override", [False, True])
def test_sampled_acceptance_matches_sync_without_reading_gpu_on_host(monkeypatch, sparse, tp_override):
    sync, _, _ = _runner(monkeypatch, asynchronous=False, sparse=sparse, tp_override=tp_override)
    x1 = [5, 5, 5]
    hidden = torch.zeros(3, 2, device="cuda")
    expected = sync._mtp_decode(hidden, x1)
    runner, seqs, vhidden = _runner(monkeypatch, asynchronous=True, sparse=sparse, tp_override=tp_override)
    original_cpu, original_list, original_item = torch.Tensor.cpu, torch.Tensor.tolist, torch.Tensor.item

    def reject_device_read(original):
        def wrapped(tensor, *args, **kwargs):
            if tensor.is_cuda:
                raise AssertionError("sampled acceptance read GPU state on the host")
            return original(tensor, *args, **kwargs)
        return wrapped

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", reject_device_read(original_cpu))
        guard.setattr(torch.Tensor, "tolist", reject_device_read(original_list))
        guard.setattr(torch.Tensor, "item", reject_device_read(original_item))
        completion = runner._mtp_decode(hidden, x1)
    assert isinstance(completion, MtpAsyncCompletion)
    valid, committed = completion.collect()
    assert committed == expected
    state = runner._mtp_async_state
    counts = torch.tensor(valid, device="cuda")
    torch.testing.assert_close(state.context_lens, counts+2)
    torch.testing.assert_close(state.resume_num_accepted, counts)
    expected_bonus = torch.tensor([sync._mtp_relay[s.seq_id][0] for s in seqs], device="cuda")
    torch.testing.assert_close(state.relay_tokens, expected_bonus)
    torch.testing.assert_close(state.relay_hidden, vhidden[torch.arange(3, device="cuda")*4+counts-1])
    torch.testing.assert_close(runner.penalty_commits[0][1], counts-1)
    # CPU GDN tables stay unchanged until drain; GPU resume columns carry the decision.
    assert seqs[1].ssm_block_table == [10, 11, 12, 13]


@cuda
def test_sampled_successor_uses_gpu_relay_before_predecessor_collection(monkeypatch):
    runner, seqs, _ = _runner(monkeypatch, asynchronous=True, sparse=True)
    first = runner._mtp_decode(torch.zeros(3, 2, device="cuda"), [5, 5, 5])
    state = runner._mtp_async_state
    # Simulate optimistic scheduler placeholders while the first output is pending.
    for seq in seqs:
        seq.token_ids.extend([-1]*4)
    second = runner._mtp_decode(state.relay_hidden, state.relay_tokens)
    assert all(state._busy)
    first_valid, first_tokens = first.collect()
    second_valid, second_tokens = second.collect()
    assert first_valid == second_valid == [1, 2, 4]
    assert [row[0] for row in first_tokens] == [5, 5, 5]
    assert [row[0] for row in second_tokens] == [2, 2, 2]
    assert state.context_lens.tolist() == [4, 6, 10]


def test_sampled_chain_requires_the_matching_graph_family():
    runner = MtpMixin()
    runner._mtp_can_sample = True
    runner._mtp_async_state = SimpleNamespace(can_remap=lambda ids: True)
    runner._mtp_gpu_prep_on = runner._mtp_draft_graph = runner._mtp_verify_graph = True
    runner._draft_size_to_graph = {4: object()}
    runner._draft_size_to_graph_sampled_sparse = {}
    runner._draft_size_to_graph_sampled = {}
    runner._verify_size_to_graph = {4: object()}
    seqs = [SimpleNamespace(seq_id=1, temperature=1.0, top_k=20)]
    assert not runner.mtp_async_can_chain(seqs)
    runner._draft_size_to_graph_sampled_sparse[4] = object()
    assert runner.mtp_async_can_chain(seqs)
    seqs[0].top_k = -1
    assert not runner.mtp_async_can_chain(seqs)
    runner._draft_size_to_graph_sampled[4] = object()
    assert runner.mtp_async_can_chain(seqs)
    runner._mtp_verify_graph = False
    assert not runner.mtp_async_can_chain(seqs)


@cuda
def test_parameter_uploads_do_not_overwrite_the_inflight_predecessor():
    runner = MtpMixin()
    runner.memory_manager = SimpleNamespace(vocab_size=128)
    runner.max_running_seqs = 2
    runner._sp_host_f = None
    runner._mtp_prep_epoch = 0
    seq = SimpleNamespace(temperature=0.8, top_k=20, top_p=0.95)
    # Allocate before the delay: allocation itself may synchronize.
    runner._mtp_sample_params([seq], torch.device("cuda"))
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        first = tuple(x.clone() for x in runner._mtp_sample_params([seq], torch.device("cuda")))
        runner._mtp_prep_epoch += 1
        seq.temperature, seq.top_k, seq.top_p = 1.2, 40, 0.7
        second = tuple(x.clone() for x in runner._mtp_sample_params([seq], torch.device("cuda")))
    stream.synchronize()
    assert first[0].item() == pytest.approx(0.8)
    assert first[1].item() == 20
    assert first[2].item() == pytest.approx(0.95)
    assert second[0].item() == pytest.approx(1.2)
    assert second[1].item() == 40
    assert second[2].item() == pytest.approx(0.7)
