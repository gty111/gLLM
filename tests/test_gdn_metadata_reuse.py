"""Per-forward caches must expire when query boundaries change."""

import torch

from gllm.layers.ops.fla.index import prepare_chunk_indices, prepare_chunk_offsets
from gllm.models.qwen3_5 import _partition_query_start_loc
from gllm.runtime.input_data import InputData


def test_forward_metadata_reuses_partition_and_expires_with_new_batch():
    data = InputData.__new__(InputData)
    data.query_start_loc = torch.tensor([0, 1, 66, 196], dtype=torch.int32)
    data.query_start_loc_cpu = data.query_start_loc.clone()
    boundaries = data.get_query_start_loc()
    assert data.get_query_start_loc() is boundaries
    suffix = _partition_query_start_loc(boundaries, data.query_start_loc_cpu, 1)
    assert _partition_query_start_loc(boundaries, data.query_start_loc_cpu, 1) is suffix
    indices = prepare_chunk_indices(suffix, 64)
    offsets = prepare_chunk_offsets(suffix, 64)
    assert prepare_chunk_indices(suffix, 64) is indices
    assert prepare_chunk_offsets(suffix, 64) is offsets
    torch.testing.assert_close(
        indices,
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(offsets, torch.tensor([0, 2, 5], dtype=torch.int32))
    data.query_start_loc.copy_(torch.tensor([0, 1, 2, 3]))
    data.query_start_loc_cpu = data.query_start_loc.clone()
    fresh = data.get_query_start_loc()
    assert fresh is not boundaries
    suffix = _partition_query_start_loc(fresh, data.query_start_loc_cpu, 1)
    torch.testing.assert_close(
        prepare_chunk_offsets(suffix, 64), torch.tensor([0, 1, 2], dtype=torch.int32)
    )


def test_unmanaged_mutable_boundaries_do_not_use_forward_cache():
    boundaries = torch.tensor([0, 65], dtype=torch.int32)
    torch.testing.assert_close(
        prepare_chunk_offsets(boundaries, 64), torch.tensor([0, 2], dtype=torch.int32)
    )
    boundaries[1] = 1
    torch.testing.assert_close(
        prepare_chunk_offsets(boundaries, 64), torch.tensor([0, 1], dtype=torch.int32)
    )
