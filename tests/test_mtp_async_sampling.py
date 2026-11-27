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
    runner.speculative_num_draft_tokens = k
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
    # One of the two tokens is the uncached x1; each row then caches x1 plus
    # its accepted drafts, i.e. ``valid`` tokens.
    torch.testing.assert_close(state.context_lens, counts+1)
    torch.testing.assert_close(state.resume_num_accepted, counts)
    assert completion.consumed_inputs() == x1
    # Each emitted row ends with the bonus, which is also the successor's x1.
    expected_bonus = torch.tensor([row[-1] for row in expected], device="cuda")
    assert [sync._mtp_relay[s.seq_id][0] for s in seqs] == (counts + 1).tolist()
    torch.testing.assert_close(state.relay_tokens, expected_bonus)
    torch.testing.assert_close(state.relay_hidden, vhidden[torch.arange(3, device="cuda")*4+counts-1])
    torch.testing.assert_close(runner.penalty_commits[0][1], counts-1)
    # CPU GDN tables stay unchanged until drain; GPU resume columns carry the decision.
    assert seqs[1].ssm_block_table == [10, 11, 12, 13]


@cuda
@pytest.mark.parametrize("chunk_seqs", [1, 2])
def test_dense_acceptance_is_unchanged_when_p_is_chunked(monkeypatch, chunk_seqs):
    x1, hidden = [5, 5, 5], torch.zeros(3, 2, device="cuda")
    whole, _, _ = _runner(monkeypatch, asynchronous=False, sparse=False)
    expected = whole._mtp_decode(hidden, x1)
    runner, _, _ = _runner(monkeypatch, asynchronous=False, sparse=False)
    runner._DENSE_P_CHUNK_BYTES = runner._TARGET_P_WORK_BYTES_PER_ELEMENT * 8 * 4 * chunk_seqs
    transform, rows = runner._mtp_target_probs_static, []
    runner._mtp_target_probs_static = lambda logits, *args, **kwargs: rows.append(len(logits)) or transform(logits, *args, **kwargs)
    assert runner._mtp_decode(hidden, x1) == expected
    # All 12 verify rows are transformed once, a whole sequence at a time.
    assert sum(rows) == 12 and max(rows) <= 4 * chunk_seqs and all(n % 4 == 0 for n in rows)


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
    assert first.consumed_inputs() == [5, 5, 5]
    # The first step's bonus is emitted there and consumed by the successor.
    assert [row[-1] for row in first_tokens] == [2, 2, 2]
    assert second.consumed_inputs() == [2, 2, 2]
    assert state.context_lens.tolist() == [3, 5, 9]


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


@cuda
@pytest.mark.parametrize("sparse_q", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_target_acceptance_helpers_replay_changed_inputs(sparse_q, dtype):
    """Synthetic acceptance/probability graph; no target backbone or weights."""
    runner = MtpMixin()
    n, k, vocab = 4, 2, 129
    runner._DENSE_P_CHUNK_BYTES = runner._TARGET_P_WORK_BYTES_PER_ELEMENT * vocab * (k + 1) * 2
    probabilities = torch.zeros(n, k + 1, vocab, device="cuda")
    probabilities[..., 0] = .2
    probabilities[..., 1] = .3
    probabilities[..., -1] = .5
    logits = probabilities.log().to(dtype).reshape(-1, vocab)
    temps = torch.ones(n * (k + 1), 1, device="cuda")
    ks = torch.full((n * (k + 1),), vocab, device="cuda")
    ps = torch.ones(n * (k + 1), device="cuda")
    drafts = torch.zeros(n, k, dtype=torch.int64, device="cuda")
    u = torch.tensor([[.9, .1], [.1, .9], [.1, .1], [.1, .9]], device="cuda")
    dense = torch.zeros(n, k, vocab, device="cuda")
    dense[..., 0], dense[..., 1] = .8, .2
    vals = torch.tensor([.8, .2], device="cuda").expand(n, k, 2).clone()
    idx = torch.tensor([0, 1], device="cuda").expand(n, k, 2).clone()
    drawn = torch.full((n, k), .8, device="cuda")
    q = MtpQDist(vals=vals, idx=idx, drawn=drawn) if sparse_q else MtpQDist(dense=dense)

    def run():
        q_d = runner._mtp_q_drawn(q, drafts)
        na, p_row = runner._mtp_dense_accept(logits, temps, ks, ps, drafts, q_d, u)
        q_row = runner._mtp_q_row(q, na.long().clamp(max=k - 1), vocab)
        residual = torch.where((na < k).unsqueeze(1),
                               (p_row - q_row).clamp_min(0), p_row)
        return na, p_row, residual

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    first = None
    last = None
    for stage in range(3):
        if stage == 1:
            probabilities[..., 0], probabilities[..., 1], probabilities[..., -1] = .6, .1, .3
            logits.copy_(probabilities.log().to(dtype).reshape(-1, vocab))
        if stage == 2:
            dense[..., 0], dense[..., 1] = .5, .5
            vals.fill_(.5)
            drawn.fill_(.5)
        expected = run()
        graph.replay()
        torch.cuda.synchronize()
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, atol=0., rtol=0.)
        assert actual[2][:, -1].gt(0).all()  # Target mass outside q survives.
        if first is None:
            first = tuple(x.clone() for x in actual)
        if stage == 1:
            assert not torch.equal(actual[1], first[1])
            last = actual[0].clone()
        if stage == 2:
            assert not torch.equal(actual[0], last)


@cuda
@pytest.mark.parametrize("sparse_q", [False, True])
def test_one_proposal_first_output_matches_full_target_distribution(sparse_q):
    """Synthetic one-proposal output distribution through actual GPU helpers."""
    import math

    runner = MtpMixin()
    n, vocab = 65536, 3
    torch.manual_seed(20261126 + int(sparse_q))
    p = torch.tensor([.2, .3, .5], device="cuda")
    q_row = torch.tensor([.8, .2, 0.], device="cuda").expand(n, vocab).contiguous()
    drafts = torch.multinomial(q_row, 1)
    q_d = q_row.gather(1, drafts)
    if sparse_q:
        q = MtpQDist(
            vals=q_row[:, :2].unsqueeze(1),
            idx=torch.tensor([0, 1], device="cuda").expand(n, 1, 2),
            drawn=q_d)
    else:
        q = MtpQDist(dense=q_row.unsqueeze(1))
    logits = p.log().expand(n * 2, vocab).contiguous()
    na, target = runner._mtp_dense_accept(
        logits, torch.ones(n * 2, 1, device="cuda"),
        torch.full((n * 2,), vocab, device="cuda"),
        torch.ones(n * 2, device="cuda"), drafts,
        runner._mtp_q_drawn(q, drafts), torch.rand(n, 1, device="cuda"))
    selected_q = runner._mtp_q_row(q, torch.zeros(n, dtype=torch.long, device="cuda"), vocab)
    residual = torch.where((na == 0).unsqueeze(1),
                           (target - selected_q).clamp_min(0), target)
    bonus = torch.multinomial(residual, 1).squeeze(1)
    first = torch.where(na > 0, drafts[:, 0], bonus)
    counts = torch.bincount(first, minlength=vocab).cpu()
    expected = p.cpu().double()
    error = (counts.double() / n - expected).abs()
    # Register both q representations, all three bins: family alpha 0.001.
    log_factor = math.log(2 * 6 / .001)
    bound = (2 * expected * (1 - expected) * log_factor / n).sqrt() + 2 * log_factor / (3 * n)
    assert bool((error <= bound).all()), (counts.tolist(), error.tolist(), bound.tolist())
    assert counts[2] > 0  # This token cannot be proposed by q.


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("temperature", [1.-5e-6, 1.+5e-6])
@pytest.mark.parametrize("context", ["solo", "sampled_peer", "greedy_peer"])
def test_target_temperature_gate_matches_real_ordinary_entry(
    monkeypatch, dtype, temperature, context,
):
    from gllm.layers import sampler as sampler_module

    seqs = [SimpleNamespace(temperature=temperature, top_k=4)]
    rows = [[0., -1., -2., -3.]]
    if context != "solo":
        seqs.append(SimpleNamespace(temperature=.6,
                                   top_k=1 if context == "greedy_peer" else 4))
        rows.append([-.5, .2, 1., -1.])
    logits = torch.tensor(rows, dtype=dtype, device="cuda")
    temps = torch.tensor([s.temperature for s in seqs], device="cuda")
    ks = torch.tensor([s.top_k for s in seqs], dtype=torch.int32, device="cuda")
    ps = torch.ones(len(seqs), device="cuda")
    ps[0] = logits[0].softmax(-1)[0].float()
    data = SimpleNamespace(seqs=seqs, temperature=temps, top_k=ks, top_p=ps,
                           needs_repetition_penalty=False)
    captured = []
    def sample(weights, count):
        assert count == 1
        captured.append(weights.clone())
        return weights.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(sampler_module, "top_k_top_p_sampling_from_probs", None)
    ordinary = sampler_module.Sampler()
    tokens = ordinary.forward_gpu(logits.clone(), data)
    expected = captured[0].clone()
    greedy = ks == 1
    expected[greedy] = 0
    expected[greedy, tokens[greedy]] = 1
    need_temperature = ordinary._get_sampling_flags(data)["need_temperature"]
    original = logits.clone()
    actual = MtpMixin._mtp_target_probs_static(
        logits, temps[:, None], ks, ps, need_temperature=need_temperature)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(logits, original, atol=0., rtol=0.)


@cuda
@pytest.mark.parametrize("peer_temperature", [1., .6])
def test_decode_passes_batch_temperature_gate_to_all_target_chunks(monkeypatch, peer_temperature):
    from gllm.layers.sampler import Sampler

    runner, seqs, _ = _runner(monkeypatch, asynchronous=False, sparse=True)
    seqs[0].temperature = 1.+5e-6
    seqs[1].temperature = peer_temperature
    runner._DENSE_P_CHUNK_BYTES = runner._TARGET_P_WORK_BYTES_PER_ELEMENT * 8 * 4
    expected = Sampler()._get_sampling_flags(SimpleNamespace(
        seqs=seqs, needs_repetition_penalty=False))["need_temperature"]
    transform, seen = runner._mtp_target_probs_static, []
    def observe(logits, *args, **kwargs):
        seen.append(kwargs["need_temperature"])
        return transform(logits, *args, **kwargs)
    monkeypatch.setattr(runner, "_mtp_target_probs_static", observe)
    runner._mtp_decode(torch.zeros(3, 2, device="cuda"), [5, 5, 5])
    assert seen == [expected] * 3
