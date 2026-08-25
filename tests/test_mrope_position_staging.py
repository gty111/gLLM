import pytest
import torch

from gllm.runtime.model_runner import _concat_mrope_positions_pinned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_concat_mrope_positions_is_pinned_and_exact():
    first = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
    second = torch.tensor([[7], [8], [9]], dtype=torch.int64)

    actual = _concat_mrope_positions_pinned([first, second])
    expected = torch.concat([first, second], dim=1)

    assert actual.is_pinned()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_concat_empty_mrope_positions_preserves_layout():
    actual = _concat_mrope_positions_pinned([])

    assert actual.shape == (3, 0)
    assert actual.dtype is torch.int64
