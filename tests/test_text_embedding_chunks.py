from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.model_runner import EmbeddingInfo, ModelRunner
from gllm.runtime.sequence import GenerationSequence


def make_runner(uses_mrope, tuple_output):
    runner = ModelRunner.__new__(ModelRunner)
    runner.uses_mrope = uses_mrope
    runner.hidden_size = 4
    runner.input_hidden_states = torch.empty((32, 4))
    runner.embedding_cache = {}
    runner.disagg_embeds = {}
    weight = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
    calls = []

    def embed(ids, media, mask):
        assert media is None
        calls.append(ids.clone())
        value = torch.nn.functional.embedding(ids.masked_fill(mask, 0), weight)
        return (value, None) if tuple_output else value

    runner.model = SimpleNamespace(
        get_mm_placeholder_token_ids=lambda: [126, 127],
        embed_input_ids=embed,
    )
    return runner, weight, calls


@pytest.mark.parametrize('uses_mrope', [False, True])
@pytest.mark.parametrize('tuple_output', [False, True])
@torch.inference_mode()
def test_chunks_match_full_embedding_and_keep_no_prompt_tensor(uses_mrope, tuple_output):
    if uses_mrope and not torch.cuda.is_available():
        pytest.skip('MRoPE staging requires CUDA pinned memory')
    runner, weight, calls = make_runner(uses_mrope, tuple_output)
    tokens = [10, 11, 126, 13, 14, 15, 16, 17, 18, 19]
    seq = GenerationSequence(1, tokens, [], output_len=8)
    reference_ids = torch.tensor(tokens).masked_fill(torch.tensor(tokens) >= 126, 0)
    reference = torch.nn.functional.embedding(reference_ids, weight)
    for start, end in [(0, 3), (3, 7), (7, 10)]:
        seq.computed_token_num = start
        seq.to_compute_token_num = end - start
        ctx = runner._mm_prepare_cpu([seq])
        assert ctx['prefill_works'][0]['input_ids_cpu'].tolist() == tokens[start:end]
        output = runner._mm_prepare_gpu(ctx)
        torch.testing.assert_close(output, reference[start:end])
        positions = torch.arange(start, end)
        if uses_mrope:
            positions = positions.expand(3, -1)
        torch.testing.assert_close(ctx['mrope_positions'], positions)
        info = runner.embedding_cache[seq.seq_id]
        assert info.embedding is None
        assert info.prompt_positions is None
        assert info.deepstack_embedding is None
        assert info.mrope_position_delta == 0
    assert [len(x) for x in calls] == [3, 4, 3]

    # A fresh request can begin after a prefix-cache hit without embedding
    # the cached prefix. Preemption must also permit recomputing old spans.
    runner.embedding_cache.clear()
    seq.computed_token_num = 7
    seq.to_compute_token_num = 3
    output, _ = runner.mm_prepare_inputs([seq])
    torch.testing.assert_close(output, reference[7:])
    seq.token_ids.append(20)
    seq.computed_token_num = seq.prompt_len
    seq.to_compute_token_num = 1
    decode = runner._mm_prepare_cpu([seq])
    assert decode['prefill_works'] == []
    expected = torch.tensor([10])
    if uses_mrope:
        expected = expected.expand(3, -1)
    torch.testing.assert_close(decode['mrope_positions'], expected)
    seq.preempt()
    seq.to_compute_token_num = 2
    output, _ = runner.mm_prepare_inputs([seq])
    torch.testing.assert_close(output, reference[:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA pinned memory')
@torch.inference_mode()
def test_mixed_decode_text_and_cached_image_keep_positions_and_deepstack():
    runner, weight, calls = make_runner(True, True)
    decode = GenerationSequence(1, [10, 11, 12], [])
    decode.prompt_len = 2
    decode.computed_token_num = 2
    decode.to_compute_token_num = 1
    runner.embedding_cache[1] = EmbeddingInfo(mrope_position_delta=5)
    text = GenerationSequence(2, [20, 21, 22, 23, 24], [])
    text.computed_token_num = 2
    text.to_compute_token_num = 2
    media = GenerationSequence(3, [30, 31, 32, 33], [], mm_contents={'image': ['unused'], 'video': []})
    media.computed_token_num = 1
    media.to_compute_token_num = 2
    image_embeddings = weight[30:34].clone()
    deepstack = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    runner.embedding_cache[3] = EmbeddingInfo(
        embedding=image_embeddings,
        prompt_positions=torch.arange(40, 44).expand(3, -1),
        mrope_position_delta=40,
        deepstack_embedding=deepstack,
    )
    cleared, written = [], []
    runner.model._clear_deepstack_input_embeds = lambda n: cleared.append(n)
    runner.model._set_deepstack_input_embeds = lambda chunk, offset: written.append((chunk.clone(), offset))
    output, positions = runner.mm_prepare_inputs([decode, text, media])
    assert output.shape == (5, 4)  # First row is the existing decode placeholder.
    torch.testing.assert_close(output[1:3], weight[22:24])
    torch.testing.assert_close(output[3:], image_embeddings[1:3])
    torch.testing.assert_close(positions, torch.tensor([7, 2, 3, 41, 42]).expand(3, -1))
    assert len(calls) == 1 and calls[0].tolist() == [22, 23]
    assert runner.embedding_cache[3].embedding is image_embeddings
    assert cleared == [5]
    assert len(written) == 1 and written[0][1] == 3
    torch.testing.assert_close(written[0][0], deepstack[:, 1:3])
