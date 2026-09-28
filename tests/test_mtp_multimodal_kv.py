"""MTP refresh must consume shifted visual features, including chunk tails."""
from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.model_runner import EmbeddingInfo
from gllm.runtime.sequence import GenerationSequence
from gllm.speculative import mtp


def image_seq(sid=2):
    seq = GenerationSequence(sid, [10, 126, 126, 126, 20, 127, 127, 30], [],
                             output_len=16, mm_contents={'image': ['unused']})
    return seq


def install_visual(runner, seq):
    mask = torch.tensor(seq.token_ids) >= 126
    # Extra columns simulate deepstack; only the base visual embedding belongs
    # in the MTP embedding input, while the target hidden already includes it.
    features = torch.arange(40, dtype=torch.float32).reshape(5, 8) + 1000
    runner.embedding_cache[seq.seq_id] = EmbeddingInfo(
        multimodal_embeddings=features, is_multimodal_cpu=mask,
        prompt_positions=torch.arange(8), mrope_position_delta=0,
    )
    return mask, features


@pytest.mark.parametrize('start,end', [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (7, 8)])
@torch.inference_mode()
def test_shifted_visual_span_and_final_chunk_release(make_mm_runner, start, end):
    runner, weight, _ = make_mm_runner(False, True)
    runner.mtp_enabled = True
    runner.model.mtp = SimpleNamespace(supports_visual_inputs=True)
    seq = image_seq()
    mask, features = install_visual(runner, seq)
    seq.computed_token_num, seq.to_compute_token_num = start, end - start
    runner.mm_prepare_inputs([seq])
    chunk_start, chunk_end, indices, values = runner._mtp_prefill_visual[seq.seq_id]
    assert (chunk_start, chunk_end) == (start, end)
    full = weight[torch.tensor(seq.token_ids)].clone()
    full[mask] = features[:, :4]
    shifted_end = min(end + 1, seq.prompt_len)
    actual = weight[torch.tensor(seq.token_ids[start + 1:shifted_end], dtype=torch.long)].clone()
    if indices.numel():
        actual[indices] = values
    torch.testing.assert_close(actual, full[start + 1:shifted_end])
    if end == seq.prompt_len:
        assert runner.embedding_cache[seq.seq_id].multimodal_embeddings is None
        assert runner.embedding_cache[seq.seq_id].is_multimodal_cpu is None


@torch.inference_mode()
def test_mixed_refresh_keeps_text_and_image_rows_aligned(make_mm_runner, monkeypatch):
    monkeypatch.setattr(mtp, 'is_last_pp_rank', lambda: True)
    monkeypatch.setattr(mtp, 'is_first_pp_rank', lambda: True)
    monkeypatch.setattr(mtp, 'is_dp_attn', lambda: False)
    runner, weight, _ = make_mm_runner(False, True)
    runner.mtp_enabled = True
    seen = []

    def forward(data, hidden, tokens, *, visual_inputs=None):
        embeddings = weight[tokens].clone()
        if visual_inputs is not None:
            indices, values = visual_inputs
            embeddings.index_copy_(0, indices, values)
        seen.append(embeddings)

    runner.model.mtp = SimpleNamespace(supports_visual_inputs=True, forward=forward)
    media = image_seq()
    mask, visual = install_visual(runner, media)
    media.computed_token_num, media.to_compute_token_num = 2, 3
    text = GenerationSequence(3, [40, 41, 42], [], output_len=16)
    text.to_compute_token_num = 3
    runner.mm_prepare_inputs([media, text])
    data = SimpleNamespace(seqs=[media, text], tokens=torch.tensor([126, 126, 20, 40, 41, 42]),
                           query_start_loc_cpu=torch.tensor([0, 3, 6]),
                           forward_metadata_plan=SimpleNamespace(num_tokens=6))

    def patch(tokens, patches, gpu_next, gpu_pairs):
        for index, value in patches:
            tokens[index] = value
        for index, row in gpu_pairs:
            tokens[index] = gpu_next[row]

    runner._mtp_staging = SimpleNamespace(refresh_tokens=torch.empty(6, dtype=torch.long),
                                           capacity=2, patch_shifted_tokens=patch)
    runner._prepare_attention_metadata = lambda _: None
    runner._mtp_sync_kv(data, torch.zeros(6, 4), tail_seqs=[media, text],
                        tail_next_tokens=torch.tensor([50, 51]))
    expected_image = torch.stack([visual[2, :4], weight[20], visual[3, :4]])
    expected_text = weight[torch.tensor([41, 42, 51])]
    torch.testing.assert_close(seen[0], torch.cat([expected_image, expected_text]))
    assert runner._mtp_prefill_visual == {}
    # A following text/decode batch must not inherit the previous image patch.
    text.computed_token_num = text.prompt_len
    assert runner._mtp_shifted_visual_inputs(SimpleNamespace(seqs=[text])) is None


@torch.inference_mode()
def test_visual_patch_uses_current_batch_offsets(make_mm_runner):
    runner, _, _ = make_mm_runner(False, True)
    media = image_seq()
    media.computed_token_num, media.to_compute_token_num = 2, 2
    _, features = install_visual(runner, media)
    chunk = runner._mtp_visual_chunk(media, runner.embedding_cache[media.seq_id], 1)
    runner._mtp_prefill_visual = {media.seq_id: chunk}
    decode = GenerationSequence(1, [1, 2], [], output_len=8)
    decode.computed_token_num = decode.prompt_len
    data = SimpleNamespace(seqs=[decode, media], query_start_loc_cpu=torch.tensor([0, 4, 6]),
                           tokens=torch.zeros(6, dtype=torch.long))
    indices, values = runner._mtp_shifted_visual_inputs(data)
    assert indices.tolist() == [4]
    torch.testing.assert_close(values, features[2:3, :4])
    media.computed_token_num = 3
    with pytest.raises(RuntimeError, match='scheduled prefill span'):
        runner._mtp_shifted_visual_inputs(data)


def test_qwen_head_uses_visual_embeddings_before_norm(monkeypatch):
    from gllm.models import qwen3_5
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    head = SimpleNamespace(_embed=lambda ids: weight[ids],
                           pre_fc_norm_embedding=lambda e: e * 2,
                           pre_fc_norm_hidden=lambda h: h * 3,
                           fc=lambda eh: eh, mtp_block=lambda data, x, residual: (x, None),
                           norm=None)
    monkeypatch.setattr(qwen3_5, 'maybe_fused_norm', lambda x, *args: (x, None))
    hidden = torch.ones(3, 4)
    visual = torch.full((1, 4), 99.0)
    out = qwen3_5.Qwen3_5MTP.forward(head, None, hidden, torch.tensor([1, 2, 3]),
                                    visual_inputs=(torch.tensor([1]), visual))
    expected = weight[torch.tensor([1, 2, 3])].clone()
    expected[1] = visual[0]
    torch.testing.assert_close(out, torch.cat([expected * 2, hidden * 3], dim=-1))
