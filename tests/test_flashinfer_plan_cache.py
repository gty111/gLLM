"""CPU regressions for planning against mutable inference-mode metadata."""

import pytest
import torch

from gllm.layers.ops import flash_attn_compat as compat


@pytest.fixture
def plan(monkeypatch):
    class Wrapper:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, **kwargs):
            self.q_lengths = kwargs["qo_indptr"].tolist()
            self.k_lengths = kwargs["kv_indptr"].tolist()

    monkeypatch.setattr(compat, "BatchPrefillWithRaggedKVCacheWrapper", Wrapper)
    monkeypatch.setattr(compat, "_workspace", lambda device: None)
    q = torch.zeros(4, 2, 64)

    @torch.inference_mode()
    def build(q_lengths, k_lengths):
        return compat._planned_wrapper(q, q, q, q_lengths, k_lengths, False, None)

    return build


@torch.inference_mode()
def test_shared_inference_metadata_is_replanned_after_mutation(plan):
    lengths = torch.tensor([0, 2, 4], dtype=torch.int32)
    first = plan(lengths, lengths)
    assert plan(lengths, lengths) is not first
    lengths[1] = 1
    second = plan(lengths, lengths)
    assert second is not first
    assert second.q_lengths == [0, 1, 4]
    assert second.k_lengths == [0, 1, 4]


@torch.inference_mode()
def test_distinct_inference_metadata_is_replanned_after_mutation(plan):
    q_lengths = torch.tensor([0, 2, 4], dtype=torch.int32)
    k_lengths = torch.tensor([0, 2, 4], dtype=torch.int32)
    first = plan(q_lengths, k_lengths)
    assert plan(q_lengths, k_lengths) is not first
    q_lengths[1] = 1
    k_lengths[1] = 3
    updated = plan(q_lengths, k_lengths)
    assert updated.q_lengths == [0, 1, 4]
    assert updated.k_lengths == [0, 3, 4]
