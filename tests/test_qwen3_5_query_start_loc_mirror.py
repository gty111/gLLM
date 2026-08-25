import pytest
import torch

from gllm.layers.ops.fla import index as fla_index
from gllm.models.qwen3_5 import (
    _apply_strided_attention_output_gate,
    _partition_query_start_loc,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partition_preserves_cpu_mirror_and_avoids_gpu_fallback(monkeypatch):
    cpu = torch.tensor([0, 4, 10, 80], dtype=torch.int64)
    cuda = cpu.cuda()

    partition = _partition_query_start_loc(cuda, cpu, 1)
    assert partition.tolist() == [0, 6, 76]
    assert partition._cpu_view.tolist() == [0, 6, 76]

    def fail_prepare_lens(*args, **kwargs):
        raise AssertionError("GPU fallback must not run when a CPU mirror exists")

    monkeypatch.setattr(fla_index, "prepare_lens", fail_prepare_lens)
    indices = fla_index.prepare_chunk_indices(partition, 16)
    offsets = fla_index.prepare_chunk_offsets(partition, 16)
    assert indices.cpu().tolist() == [
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
    ]
    assert offsets.cpu().tolist() == [0, 1, 6]

    # Repeated layer calls with the same lengths must reuse device metadata.
    assert fla_index.prepare_chunk_indices(partition, 16).data_ptr() == indices.data_ptr()
    assert fla_index.prepare_chunk_offsets(partition, 16).data_ptr() == offsets.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partition_without_cpu_metadata_keeps_previous_behavior():
    query_start_loc = torch.tensor([0, 4, 10, 80], device="cuda")
    partition = _partition_query_start_loc(query_start_loc, None, 1)

    assert partition.tolist() == [0, 6, 76]
    assert not hasattr(partition, "_cpu_view")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_strided_attention_output_gate_is_bitwise_exact():
    torch.manual_seed(29)
    tokens, heads, head_dim = 128, 24, 256
    q_size = heads * head_dim
    kv_size = 4 * head_dim
    qkv = torch.randn(
        tokens,
        2 * q_size + 2 * kv_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_gate = qkv[:, : 2 * q_size].view(tokens, heads, 2 * head_dim)
    _, gate = torch.chunk(q_gate, 2, dim=-1)
    attn_out = torch.randn(
        tokens, q_size, device="cuda", dtype=torch.bfloat16
    )

    expected = attn_out * torch.sigmoid(gate.reshape(tokens, q_size))
    actual = _apply_strided_attention_output_gate(attn_out, gate)

    assert not gate.is_contiguous()
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
