from types import SimpleNamespace

import torch

from gllm.multimodal.mixin import EmbeddingInfo
from gllm.speculative.mtp import MtpMixin


def test_verify_dummy_embedding_cache_preserves_live_entries(monkeypatch):
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: zeros(
        *args, **{**kwargs, "device": "cpu"}
    ))
    live = EmbeddingInfo(mrope_position_delta=torch.tensor([7]))
    runner = SimpleNamespace(
        memory_manager=SimpleNamespace(
            dummy_page=0, ssm_segment=None, recurrent_segment=None,
        ),
        _mtp_k=3, page_size=16, use_mm=True, uses_mrope=True,
        embedding_cache={1_000_000: live},
    )

    seqs = MtpMixin._create_dummy_verify_seqs(runner, nd=2, qlen=4)

    assert len(seqs) == 2
    assert runner.embedding_cache[1_000_000] is live
    stub = runner.embedding_cache[1_000_001]
    assert isinstance(stub, EmbeddingInfo)
    assert stub.mrope_position_delta.tolist() == [0]
    assert runner._dummy_verify_cache_ids == [1_000_001]
    MtpMixin._drop_dummy_verify_cache(runner)
    assert runner.embedding_cache == {1_000_000: live}
