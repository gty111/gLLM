import pytest
import torch

from gllm.models.utils import _merge_multimodal_embeddings


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'))])
@torch.inference_mode()
def test_merge_preserves_rows_dtype_and_nested_embedding_order(device):
    backing = torch.arange(12 * 16, dtype=torch.float32, device=device).reshape(12, 16)
    embeddings = backing[:, ::2]  # A non-contiguous destination.
    mask = torch.tensor([False, True, True, False, False, True, False, False, True, False, False, False], device=device)
    first = torch.full((2, 8), -3, dtype=torch.float64, device=device)
    second = torch.full((1, 1, 8), -7, dtype=torch.float64, device=device)
    third = torch.full((1, 8), -9, dtype=torch.float64, device=device)
    expected = embeddings.clone()
    expected[mask] = torch.cat([first, second.flatten(0, -2), third]).float()
    result = _merge_multimodal_embeddings(embeddings, mask, [first, (second, third)])
    assert result is embeddings
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(backing[:, 1::2], torch.arange(12 * 16, device=device).reshape(12, 16)[:, 1::2].float())


@torch.inference_mode()
def test_merge_empty_and_mismatched_placeholders():
    embeddings = torch.ones(5, 4)
    empty_mask = torch.zeros(5, dtype=torch.bool)
    _merge_multimodal_embeddings(embeddings, empty_mask, torch.empty(0, 4))
    torch.testing.assert_close(embeddings, torch.ones(5, 4))
    with pytest.raises(ValueError, match='2 placeholders'):
        _merge_multimodal_embeddings(embeddings, torch.tensor([True, False, True, False, False]), torch.ones(1, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@torch.inference_mode()
def test_long_prompt_merge_workspace_is_not_proportional_to_hidden_elements():
    # Production failed allocating 146176 * 5120 * 8 bytes of scan workspace.
    tokens, hidden = 146176, 5120
    embeddings = torch.zeros((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    mask = torch.zeros(tokens, device='cuda', dtype=torch.bool)
    mask[1024:1280] = True
    visual = torch.ones((256, hidden), device='cuda', dtype=embeddings.dtype)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _merge_multimodal_embeddings(embeddings, mask, visual)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    assert extra < 16 * 1024 * 1024, f'merge allocated {extra} temporary bytes'
    assert bool((embeddings[1024:1280] == 1).all())
    assert bool((embeddings[:1024] == 0).all())
    assert bool((embeddings[-1] == 0).all())
