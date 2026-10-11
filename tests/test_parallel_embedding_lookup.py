"""Resident BF16/FP8 embedding shards and replay with changing row IDs."""

import json

import pytest
import torch
from safetensors.torch import save_file

from gllm.models.weight_utils import ShardedEmbeddingLookup


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_parallel_table_partition(tmp_path, monkeypatch, fp8, rank):
    import gllm.distributed.parallel_state as parallel

    monkeypatch.setattr(parallel, "get_tp_size", lambda: 3)
    monkeypatch.setattr(parallel, "get_tp_rank", lambda: rank)
    monkeypatch.setattr(parallel, "tensor_model_parallel_all_reduce", lambda x: x)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(dtype)
    keys = [f"table.shard_{i}.weight" for i in range(2)]
    tensors = {keys[0]: table[:4], keys[1]: table[4:]}
    if fp8:
        tensors["table.weight_scale"] = torch.tensor([0.25])
    save_file(tensors, str(tmp_path / "parts.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "parts.safetensors")})
    )
    source = ShardedEmbeddingLookup(tmp_path, keys)
    resident = source.load_parallel(device="cpu")
    ids = torch.tensor([[6, 0, 3], [4, 2, 6]])
    expected = source(ids)
    valid = (ids >= rank * 3) & (ids < min(rank * 3 + 3, 7))
    torch.testing.assert_close(
        resident(ids), expected * valid.unsqueeze(-1), rtol=0, atol=0
    )
    assert resident.weight.dtype == dtype
    assert resident.weight.shape == (3, 4)
    source.close()
    torch.testing.assert_close(
        resident(ids), expected * valid.unsqueeze(-1), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8", [False, True])
def test_lookup_graph_replays_changed_ids(fp8):
    from gllm.models.weight_utils import ParallelEmbeddingLookup

    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    table = torch.arange(32, device="cuda", dtype=torch.float32).reshape(8, 4).to(dtype)
    lookup = ParallelEmbeddingLookup(
        table, 0, 8, 1, torch.tensor([0.5], device="cuda") if fp8 else None
    )
    ids = torch.tensor([0, 1, 2], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lookup(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = lookup(ids)
    ids.copy_(torch.tensor([7, 2, 5], device="cuda"))
    graph.replay()
    torch.testing.assert_close(result, lookup(ids), rtol=0, atol=0)
