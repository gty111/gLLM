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


def _target_reference(logits, temperature, k, p):
    """Enumerate probability groups directly, independently of sort/cumsum."""
    import math

    if k == 1:
        best = max(range(len(logits)), key=lambda i: logits[i])
        return [float(i == best) for i in range(len(logits))]
    maximum = max(logits)
    weights = [math.exp((x - maximum) / temperature) for x in logits]
    total = sum(weights)
    probabilities = [x / total for x in weights]
    limit = len(logits) if k == -1 else k
    selected = []
    for probability in probabilities:
        above = [x for x in probabilities if x > probability]
        selected.append(probability > 0 and len(above) < limit
                        and (p >= 1 or sum(above) < p))
    mass = sum(x for x, keep in zip(probabilities, selected) if keep)
    return [x / mass if keep else 0.
            for x, keep in zip(probabilities, selected)]



@pytest.mark.parametrize("temperature", [0.5, 1., 2.])
@pytest.mark.parametrize("k,p", [(2, .6), (2, 1.), (-1, .95), (1, .6)])
@pytest.mark.parametrize("values", [[0., -1., -2., -3.], [0.] * 129])
def test_speculative_target_matches_independent_reference(values, temperature, k, p):
    from gllm.speculative.mtp import MtpMixin

    logits = torch.tensor([values], dtype=torch.float64)
    original = logits.clone()
    actual = MtpMixin._mtp_target_probs_static(
        logits, torch.tensor([[temperature]], dtype=logits.dtype),
        torch.tensor([k]), torch.tensor([p], dtype=logits.dtype),
    )
    expected = torch.tensor([_target_reference(values, temperature, k, p)],
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual > 0, expected > 0)
    assert torch.equal(logits, original)


def test_speculative_target_retains_ties_beyond_proposal_window():
    from gllm.speculative.mtp import MtpMixin

    actual = MtpMixin._mtp_target_probs_static(
        torch.zeros(1, 129), torch.ones(1, 1),
        torch.tensor([20]), torch.tensor([.1]),
    )
    torch.testing.assert_close(actual, torch.full((1, 129), 1 / 129))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_proposal_probabilities_follow_token_ids_and_request_order(sparse, order):
    from gllm.speculative.mtp import MtpMixin, MtpQDist

    dense = torch.tensor([
        [[.7, .3, 0., 0., 0.], [0., 0., .4, .6, 0.]],
        [[0., 0., .8, 0., .2], [.1, .9, 0., 0., 0.]],
    ])[order]
    drafts = torch.tensor([[0, 3], [2, 1]])[order]
    positions = torch.tensor([1, 0])[order]
    drawn = dense.gather(2, drafts.unsqueeze(-1)).squeeze(-1)
    vals, idx = dense.topk(2, dim=-1)
    q = MtpQDist(dense=None if sparse else dense, vals=vals.flip(-1),
                 idx=idx.flip(-1), drawn=drawn)
    torch.testing.assert_close(MtpMixin._mtp_q_drawn(q, drafts), drawn)
    torch.testing.assert_close(MtpMixin._mtp_q_row(q, positions, 5),
                               dense[torch.arange(2), positions])


def test_sparse_proposal_residual_recovers_target_mass_outside_support():
    from gllm.speculative.mtp import MtpMixin, MtpQDist

    p = torch.tensor([.2, .3, .5], dtype=torch.float64)
    q = MtpQDist(vals=torch.tensor([[[.8, .2]]], dtype=p.dtype),
                 idx=torch.tensor([[[0, 1]]]), drawn=torch.tensor([[.8]]))
    q_row = MtpMixin._mtp_q_row(q, torch.tensor([0]), 3)[0]
    residual = (p - q_row).clamp_min(0)
    residual /= residual.sum()
    torch.testing.assert_close(residual, torch.tensor([0., 1 / 6, 5 / 6], dtype=p.dtype))
    # Enumerate the one-proposal output law, including rejection probability.
    accepted = torch.minimum(p, q_row)
    torch.testing.assert_close(accepted + (1 - accepted.sum()) * residual, p)


@pytest.mark.parametrize("context", ["solo", "sampled_peer", "greedy_peer"])
def test_target_temperature_boundary_matches_ordinary_entry(monkeypatch, context):
    from types import SimpleNamespace
    from gllm.speculative.mtp import MtpMixin

    seqs = [SimpleNamespace(temperature=1.+5e-6, top_k=4)]
    values = [[0., -1., -2., -3.]]
    if context != "solo":
        seqs.append(SimpleNamespace(temperature=.6,
                                   top_k=1 if context == "greedy_peer" else 4))
        values.append([-.5, .2, 1., -1.])
    logits = torch.tensor(values)
    temps = torch.tensor([s.temperature for s in seqs])
    ks = torch.tensor([s.top_k for s in seqs])
    ps = torch.ones(len(seqs))
    ps[0] = logits[0].softmax(-1)[0]
    data = SimpleNamespace(seqs=seqs, temperature=temps, top_k=ks, top_p=ps,
                           needs_repetition_penalty=False)
    captured = []
    def sample(weights, count):
        assert count == 1
        captured.append(weights.clone())
        return weights.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(sampler, "top_k_top_p_sampling_from_probs", None)
    ordinary = sampler.Sampler()
    tokens = ordinary.forward_gpu(logits.clone(), data)
    expected = captured[0]
    for i, k in enumerate(ks.tolist()):
        if k == 1:
            expected[i].zero_()
            expected[i, tokens[i]] = 1.
    actual = MtpMixin._mtp_target_probs_static(
        logits, temps[:, None], ks, ps,
        need_temperature=ordinary._get_sampling_flags(data)["need_temperature"],
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("chunk_requests", [1, 3])
@pytest.mark.parametrize("width", [1, 3])
def test_target_chunking_preserves_batch_temperature_gate(chunk_requests, width):
    from gllm.speculative.mtp import MtpMixin

    runner = MtpMixin()
    vocab = 4
    runner._DENSE_P_CHUNK_BYTES = (
        runner._TARGET_P_WORK_BYTES_PER_ELEMENT * vocab * width * chunk_requests
    )
    logits = torch.tensor([[0., -1., -2., -3.]] * 3)
    temps = torch.tensor([[1.+5e-6], [1.], [.6]])
    ks, ps = torch.full((3,), 4), torch.ones(3)
    expected = runner._mtp_target_probs_static(logits, temps, ks, ps,
                                               need_temperature=True)
    k = width - 1
    empty_or_drafts = torch.zeros((3, k), dtype=torch.int64)
    q = torch.ones((3, k))
    accepted, row = runner._mtp_dense_accept(
        logits.repeat_interleave(width, dim=0), temps.repeat_interleave(width, dim=0),
        ks.repeat_interleave(width), ps.repeat_interleave(width),
        empty_or_drafts, q, torch.ones_like(q), need_temperature=True,
    )
    torch.testing.assert_close(row, expected, atol=0, rtol=0)
    assert torch.equal(accepted, torch.zeros(3, dtype=torch.int64))


@pytest.mark.parametrize("sampled_peer", [False, True])
def test_target_bf16_mixed_greedy_scaling_tie_matches_ordinary(monkeypatch, sampled_peer):
    from types import SimpleNamespace
    from gllm.speculative.mtp import MtpMixin

    seqs = [SimpleNamespace(temperature=.6, top_k=1),
            SimpleNamespace(temperature=1., top_k=2 if sampled_peer else 1)]
    logits = torch.tensor([[1.21875, 1.2265625], [0., -1.]], dtype=torch.bfloat16)
    data = SimpleNamespace(seqs=seqs, needs_repetition_penalty=False,
                           temperature=torch.tensor([.6, 1.], dtype=torch.bfloat16),
                           top_k=torch.tensor([s.top_k for s in seqs]),
                           top_p=torch.ones(2, dtype=torch.bfloat16))
    monkeypatch.setattr(sampler, "top_k_top_p_sampling_from_probs", None)
    ordinary = sampler.Sampler()
    tokens = ordinary.forward_gpu(logits.clone(), data)
    flags = ordinary._get_sampling_flags(data)
    # Entirely greedy batches bypass temperature scaling in both paths.
    actual = MtpMixin._mtp_target_probs_static(
        logits, data.temperature[:, None], data.top_k, data.top_p,
        need_temperature=flags["need_temperature"] and not flags["is_all_greedy"])
    assert tokens[0].item() == (0 if sampled_peer else 1)
    assert actual[0].tolist() == ([1., 0.] if sampled_peer else [0., 1.])
    if not sampled_peer:
        assert torch.equal(actual.argmax(-1), tokens)
