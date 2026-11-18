from types import SimpleNamespace

import pytest
import torch

import gllm.runtime.model_runner as runner_module
from gllm.runtime.input_data import InputData
from gllm.runtime.model_runner import ModelRunner
from gllm.runtime.sequence import GenerationSequence
from gllm.speculative.staging import MtpStagingBuffers


class _MtpStub:
    def __init__(self):
        self.forward_calls = 0

    def forward(self, input_data, hidden, token):
        assert input_data.attention_prepared
        self.forward_calls += 1
        return hidden + token.to(hidden.dtype).unsqueeze(1)

    def logits_from_hidden(self, hidden):
        logits = torch.zeros((hidden.shape[0], 8), dtype=torch.float32)
        logits[:, 3] = 4.0
        return logits


def _runner_stub():
    mtp = _MtpStub()
    runner = SimpleNamespace(model=SimpleNamespace(mtp=mtp), input_data=None)

    def prepare_input(_seqs, *, num_decodes):
        assert num_decodes == len(_seqs)
        runner.input_data = SimpleNamespace(attention_prepared=False)

    def prepare_attention(input_data):
        input_data.attention_prepared = True

    runner.prepare_input = prepare_input
    runner._prepare_attention_metadata = prepare_attention
    runner._mtp_probs_from_logits = lambda logits, _seqs: torch.softmax(logits, dim=-1)
    runner._mtp_bcast_tp = lambda token: token
    runner._q_dense = ModelRunner._q_dense
    return runner, mtp


def _inputs():
    seqs = [SimpleNamespace(), SimpleNamespace()]
    # Token histories [1, 2] and [4, 5, 6]: everything before the uncached
    # last token (the draft chain's first input) is cached.
    ctx_lens = [1, 2]
    first_tokens = [2, 6]
    hidden = torch.zeros((2, 4), dtype=torch.float32)
    return seqs, ctx_lens, first_tokens, hidden


def test_eager_mtp_draft_prepares_attention_metadata_each_step():
    runner, mtp = _runner_stub()
    seqs, ctx_lens, first_tokens, hidden = _inputs()

    drafts = ModelRunner._draft_chain_eager(
        runner, seqs, ctx_lens, first_tokens, hidden, k=3, nd=2
    )

    assert mtp.forward_calls == 3
    assert drafts == [[3, 3, 3], [3, 3, 3]]


def test_sampled_eager_mtp_draft_prepares_attention_metadata_each_step():
    runner, mtp = _runner_stub()
    seqs, ctx_lens, first_tokens, hidden = _inputs()

    drafts, q_dist = ModelRunner._draft_chain_eager_sampled(
        runner,
        seqs,
        ctx_lens,
        first_tokens,
        hidden,
        k=3,
        nd=2,
        gen=torch.Generator().manual_seed(7),
    )

    assert mtp.forward_calls == 3
    assert len(drafts) == 2
    assert all(len(row) == 3 for row in drafts)
    assert q_dist.dense.shape == (2, 3, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mtp_staging_patches_host_and_device_boundary_tokens():
    staging = MtpStagingBuffers(
        capacity=4,
        max_tokens=8,
        hidden_size=16,
        hidden_dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    shifted = torch.zeros(8, dtype=torch.int64, device="cuda")
    gpu_values = torch.tensor([31, 47], dtype=torch.int64, device="cuda")

    staging.patch_shifted_tokens(
        shifted,
        host_pairs=[(1, 13)],
        gpu_src=gpu_values,
        gpu_pairs=[(5, 1)],
    )

    assert shifted.tolist() == [0, 13, 0, 0, 0, 47, 0, 0]

@pytest.mark.parametrize("new_first", [False, True])
@pytest.mark.parametrize("k", [1, 3, 7])
@pytest.mark.parametrize("uses_mrope", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="pinned MRoPE metadata requires CUDA")),
])
def test_eager_draft_keeps_new_request_in_decode_partition(
    make_mm_runner, monkeypatch, new_first, k, uses_mrope
):
    """Shifted positions must not reclassify a completed prompt as prefill."""
    monkeypatch.setattr(runner_module, "is_first_pp_rank", lambda: True)
    runner, _, _ = make_mm_runner(uses_mrope, False)
    runner.use_mm = True
    runner.model.mtp = _MtpStub()
    data = InputData.__new__(InputData)
    seen = []
    positions = []

    def materialize(seqs, plan):
        assert plan.num_decodes == len(seqs) == 2
        assert plan.query_lens == (1, 1)
        data.seqs = seqs
        data.num_decodes = plan.num_decodes
        data.num_prefills = 0
        data.attention_prepared = False
        seen.append(tuple(s.computed_token_num for s in seqs))

    data._materialize_cpu_metadata = materialize
    data.copy_to_input_buffer = lambda: None
    data.set_mrope_position = lambda value: positions.append(value.clone())
    runner.input_data = data

    def attention(current):
        assert current.forward_metadata_plan.num_decodes == 2
        current.attention_prepared = True

    runner._prepare_attention_metadata = attention
    fresh = GenerationSequence(1, [1, 2, 3], [], output_len=32)
    fresh.append(4)
    old = GenerationSequence(2, [1, 2, 3, 4], [], output_len=32)
    old.token_ids.extend([5, 6, 7])
    seqs = [fresh, old] if new_first else [old, fresh]
    ctx = [len(s) - 1 for s in seqs]
    prompts = [s.prompt_len for s in seqs]
    histories = [s.token_ids[:] for s in seqs]
    drafts = runner._draft_chain_eager(
        seqs, ctx, [s[-1] for s in seqs], torch.zeros((2, 4)), k=k, nd=2
    )
    assert drafts == [[3] * k, [3] * k]
    assert seen == [tuple(c + j - 1 for c in ctx) for j in range(k)]
    assert [s.prompt_len for s in seqs] == prompts
    assert [s.token_ids for s in seqs] == histories
    if uses_mrope:
        assert len(positions) == k
        for j, pos in enumerate(positions):
            assert pos.tolist() == [[c + j - 1 for c in ctx]] * 3


@pytest.mark.parametrize("count", [-1, 2])
def test_eager_draft_rejects_invalid_mm_decode_partition(make_mm_runner, count):
    runner, _, _ = make_mm_runner(False, False)
    seq = GenerationSequence(1, [1, 2, 3], [], output_len=8)
    with pytest.raises(ValueError, match="decode partition"):
        runner._mm_prepare_cpu([seq], num_decodes=count)


def test_eager_draft_does_not_override_prebuilt_partition():
    runner = ModelRunner.__new__(ModelRunner)
    with pytest.raises(ValueError, match="prebuilt input"):
        runner.prepare_input(input_data=SimpleNamespace(), num_decodes=1)


def test_eager_draft_default_mm_still_rejects_prefill_before_decode(make_mm_runner):
    runner, _, _ = make_mm_runner(False, False)
    fresh = GenerationSequence(1, [1, 2, 3], [], output_len=8)
    fresh.computed_token_num, fresh.to_compute_token_num = 2, 1
    old = GenerationSequence(2, [1, 2, 3], [], output_len=8)
    old.computed_token_num, old.to_compute_token_num = 4, 1
    with pytest.raises(AssertionError, match="decode seqs must precede"):
        runner._mm_prepare_cpu([fresh, old])
