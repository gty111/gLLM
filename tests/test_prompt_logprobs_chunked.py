from types import SimpleNamespace

import torch

import gllm.runtime.model_runner as model_runner_module
from gllm.runtime.model_runner import ModelRunner


class _Head:
    def __init__(self, weight):
        self.weight = weight
        self.rows = []

    def __call__(self, hidden):
        self.rows.append(hidden.shape[0])
        return hidden @ self.weight


def _seq(seq_id, token_ids, top):
    return SimpleNamespace(
        seq_id=seq_id,
        token_ids=list(token_ids),
        raw_prompt_len=len(token_ids),
        computed_token_num=0,
        computed_prompt=False,
        prompt_logprobs_enabled=True,
        num_prompt_logprobs=top,
        prompt_logprobs_data=None,
    )


def test_prompt_logprobs_project_bounded_rows_and_match_one_shot(monkeypatch):
    torch.manual_seed(0)
    vocab, hidden_size = 32, 8
    head = _Head(torch.randn(hidden_size, vocab))
    # 4 rows x 32 vocab x 4 bytes per LM-head call.
    monkeypatch.setattr(model_runner_module, "_PROMPT_LOGPROBS_STEP_BYTES", 4 * vocab * 4)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(logits_from_hidden=head)
    runner.model_loader = SimpleNamespace(vocab_size=vocab)
    seqs = [_seq(1, torch.randint(vocab, (11,)).tolist(), 3), _seq(2, torch.randint(vocab, (6,)).tolist(), 0)]
    runner.input_data = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 11, 17]))
    hidden = torch.randn(17, hidden_size)

    runner._compute_prompt_logprobs(seqs, hidden)

    assert max(head.rows) <= 4
    assert sum(head.rows) == 10 + 5
    for seq, start in ((seqs[0], 0), (seqs[1], 11)):
        n = seq.raw_prompt_len - 1
        full = torch.log_softmax((hidden[start:start + n] @ head.weight).float(), dim=-1)
        assert seq.prompt_logprobs_data[0] is None
        for j in range(n):
            token, value, top_ids, top_vals = seq.prompt_logprobs_data[j + 1]
            assert token == seq.token_ids[j + 1]
            assert abs(value - full[j, token].item()) < 1e-6
            vals, ids = torch.topk(full[j], seq.num_prompt_logprobs) if seq.num_prompt_logprobs else ([], [])
            assert top_ids == list(map(int, ids))
            assert top_vals == [float(v) for v in vals]
    assert set(runner._last_prompt_logprobs) == {1, 2}


def test_next_prefill_chunk_shares_prompt_logprobs_with_the_running_chunk():
    from gllm.runtime.sequence import GenerationSequence
    from gllm.scheduling.scheduler import copy_for_next_prefill_chunk

    seq = GenerationSequence(7, list(range(10)), [], 4, False, 1.0, 1.0, -1, 1.0, None, False, 0, True, 2)
    seq.to_compute_token_num = 4
    nxt = copy_for_next_prefill_chunk(seq)
    # The scheduled chunk runs after the copy is taken and fills its rows.
    seq.prompt_logprobs_data[1] = (1, -0.5, [1], [-0.5])
    assert nxt.prompt_logprobs_data is seq.prompt_logprobs_data
    assert len(nxt.prompt_logprobs_data) == 10
    assert nxt.token_ids is not seq.token_ids

    plain = GenerationSequence(8, list(range(10)), [], 4, False, 1.0, 1.0, -1, 1.0, None, False, 0, False, 0)
    assert copy_for_next_prefill_chunk(plain).prompt_logprobs_data is None
