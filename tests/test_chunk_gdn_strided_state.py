import pytest
import torch

from gllm.layers.ops.fla import chunk_gated_delta_rule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gdn_updates_arena_strided_state_in_place():
    torch.manual_seed(1)
    dtype = torch.bfloat16
    nseq, heads, key_dim, value_dim, seq_len = 2, 2, 16, 16, 64
    total = nseq * seq_len
    q = torch.randn(1, total, heads, key_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, total, heads, value_dim, device="cuda", dtype=dtype)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total, heads, device="cuda", dtype=dtype)
    )
    beta = torch.sigmoid(torch.randn_like(g))
    cu_seqlens = torch.tensor([0, seq_len, total], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int32)

    compact = torch.zeros(
        5, heads, value_dim, key_dim, device="cuda", dtype=dtype
    )
    compact[1].normal_()
    compact[3].normal_()
    padded_stride = heads * value_dim * key_dim + 97
    backing = torch.zeros(5 * padded_stride, device="cuda", dtype=dtype)
    arena_state = torch.as_strided(
        backing,
        compact.shape,
        (padded_stride, value_dim * key_dim, key_dim, 1),
    )
    arena_state.copy_(compact)
    reference_state = compact.clone()

    reference_out = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=reference_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )[0]
    arena_out = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=arena_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )[0]

    torch.cuda.synchronize()
    assert torch.equal(reference_out, arena_out)
    assert torch.equal(
        reference_state[state_indices.long()], arena_state[state_indices.long()]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gdn_reads_strided_value_projection_bitwise_exact():
    torch.manual_seed(17)
    dtype = torch.bfloat16
    nseq, heads, key_dim, value_dim, seq_len = 2, 2, 16, 16, 64
    total = nseq * seq_len
    q = torch.randn(1, total, heads, key_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    compact_v = torch.randn(
        1, total, heads, value_dim, device="cuda", dtype=dtype
    )
    # Mirror V as the trailing slice of a fused [Q, K, V, Z] projection: each
    # token row has extra columns while heads and value columns remain packed.
    row_width = heads * value_dim + 37
    fused = torch.empty(1, total, row_width, device="cuda", dtype=dtype)
    strided_v = fused[..., -heads * value_dim :].view(
        1, total, heads, value_dim
    )
    strided_v.copy_(compact_v)
    assert not strided_v.is_contiguous()

    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total, heads, device="cuda", dtype=dtype)
    )
    beta = torch.sigmoid(torch.randn_like(g))
    cu_seqlens = torch.tensor(
        [0, seq_len, total], device="cuda", dtype=torch.int32
    )
    state_indices = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    reference_state = torch.randn(
        5, heads, value_dim, key_dim, device="cuda", dtype=dtype
    )
    strided_state = reference_state.clone()

    reference_out = chunk_gated_delta_rule(
        q,
        k,
        compact_v,
        g,
        beta,
        initial_state=reference_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )[0]
    strided_out = chunk_gated_delta_rule(
        q,
        k,
        strided_v,
        g,
        beta,
        initial_state=strided_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )[0]

    torch.cuda.synchronize()
    assert torch.equal(reference_out, strided_out)
    assert torch.equal(
        reference_state[state_indices.long()],
        strided_state[state_indices.long()],
    )
