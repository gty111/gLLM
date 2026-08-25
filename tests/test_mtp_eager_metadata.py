from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.model_runner import ModelRunner
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

    def prepare_input(_seqs):
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
    orig_tokens = [[1, 2], [4, 5, 6]]
    first_tokens = [2, 6]
    hidden = torch.zeros((2, 4), dtype=torch.float32)
    return seqs, orig_tokens, first_tokens, hidden


def test_eager_mtp_draft_prepares_attention_metadata_each_step():
    runner, mtp = _runner_stub()
    seqs, orig_tokens, first_tokens, hidden = _inputs()

    drafts = ModelRunner._draft_chain_eager(
        runner, seqs, orig_tokens, first_tokens, hidden, k=3, nd=2
    )

    assert mtp.forward_calls == 3
    assert drafts == [[3, 3, 3], [3, 3, 3]]


def test_sampled_eager_mtp_draft_prepares_attention_metadata_each_step():
    runner, mtp = _runner_stub()
    seqs, orig_tokens, first_tokens, hidden = _inputs()

    drafts, q_dist = ModelRunner._draft_chain_eager_sampled(
        runner,
        seqs,
        orig_tokens,
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
