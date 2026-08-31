import torch
import pytest

from gllm.layers.attention.deepseek_v4.ops import (
    apply_rope_inplace,
    compressed_indices,
    fp8_fake_quantize_inplace,
    precompute_rope_frequencies,
    sparse_attention_fused,
    sparse_attention_reference,
    window_indices,
)


def test_deepseek_v4_rope_matches_official_complex_formula():
    torch.manual_seed(29)
    frequencies = precompute_rope_frequencies(
        64,
        16,
        original_sequence_length=65536,
        base=160000.0,
        factor=16.0,
        beta_fast=32,
        beta_slow=1,
    )
    x = torch.randn(2, 16, 3, 64, dtype=torch.bfloat16)
    expected_complex = torch.view_as_complex(
        x.float().unflatten(-1, (-1, 2))
    )
    expected = torch.view_as_real(
        expected_complex * frequencies.view(1, 16, 1, 32)
    ).flatten(-2).to(torch.bfloat16)

    actual = x.clone()
    apply_rope_inplace(actual, frequencies)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_fp8_qat_uses_e8m0_scales_per_64():
    x = torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16)
    x[0, :8] = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x[1] = torch.linspace(-12, 12, 64, device="cuda", dtype=torch.bfloat16)

    grouped = x.float().view(2, 1, 64)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    expected = (
        (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        * scale
    ).reshape_as(x).to(torch.bfloat16)

    actual = x.clone()
    fp8_fake_quantize_inplace(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_fp8_qat_handles_prefix_slice_bitwise():
    torch.manual_seed(37)
    backing = torch.randn(3, 5, 512, device="cuda", dtype=torch.bfloat16)
    actual = backing[..., :448]
    grouped = actual.float().view(3, 5, 7, 64)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    expected = (
        (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        * scale
    ).reshape_as(actual).to(torch.bfloat16)

    fp8_fake_quantize_inplace(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_window_indices_prefill_and_decode_match_reference_layout():
    prefill = window_indices(4, 1, 6, 0)[0]
    torch.testing.assert_close(
        prefill,
        torch.tensor(
            [
                [0, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, 2, -1],
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
            ],
            dtype=torch.int32,
        ),
    )
    # start_pos=6 means circular slot 2 was just written: oldest -> newest.
    torch.testing.assert_close(
        window_indices(4, 1, 1, 6)[0],
        torch.tensor([[3, 0, 1, 2]], dtype=torch.int32),
    )


def test_compressed_indices_are_causal():
    actual = compressed_indices(4, 1, 12, 0, 12)[0]
    assert actual.shape == (12, 3)
    for token in range(12):
        count = (token + 1) // 4
        torch.testing.assert_close(
            actual[token, :count],
            torch.arange(12, 12 + count, dtype=torch.int32),
        )
        assert torch.all(actual[token, count:] == -1)


def test_sparse_attention_sink_matches_scalar_reference():
    torch.manual_seed(31)
    q = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    kv = torch.randn(2, 7, 8, dtype=torch.bfloat16)
    indices = torch.tensor(
        [
            [[0, -1, -1], [0, 1, -1], [0, 2, 1]],
            [[3, 1, -1], [4, 2, 0], [6, -1, -1]],
        ],
        dtype=torch.int32,
    )
    sinks = torch.randn(4, dtype=torch.float32)
    scale = 8**-0.5
    actual = sparse_attention_reference(q, kv, indices, sinks, scale).float()

    expected = torch.empty_like(actual)
    for batch in range(2):
        for token in range(3):
            valid = indices[batch, token][indices[batch, token] >= 0].long()
            values = kv[batch, valid].float()
            for head in range(4):
                logits = q[batch, token, head].float() @ values.T * scale
                logits_with_sink = torch.cat([logits, sinks[head : head + 1]])
                probs = logits_with_sink.softmax(0)[:-1]
                expected[batch, token, head] = probs @ values
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sglang_sparse_prefill_matches_reference_with_tp_head_padding():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SGLang FlashMLA sparse prefill test targets SM100")
    pytest.importorskip("sgl_kernel")

    torch.manual_seed(33)
    batch, sequence, heads, dim, kv_length = 2, 3, 16, 512, 160
    query = torch.randn(
        batch, sequence, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(
        batch, kv_length, dim, device="cuda", dtype=torch.bfloat16
    )
    # Deliberately leave a gap of -1 sentinels between the SWA and compressed
    # regions.  This is the layout produced by the correctness oracle and
    # verifies that the fused wrapper must not use a simple prefix length.
    indices = torch.full(
        (batch, sequence, 137), -1, device="cuda", dtype=torch.int32
    )
    for row in range(sequence):
        indices[:, row, : row + 5] = torch.arange(
            row + 5, device="cuda", dtype=torch.int32
        )
        indices[:, row, 128:137] = torch.arange(
            32, 41, device="cuda", dtype=torch.int32
        )
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    scale = dim**-0.5

    actual = sparse_attention_fused(query, kv, indices, sinks, scale)
    expected = sparse_attention_reference(query, kv, indices, sinks, scale)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_dsv4_decode_matches_sparse_reference():
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("FlashInfer DeepSeek-V4 sparse MLA requires Blackwell")
    pytest.importorskip("flashinfer")
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    torch.manual_seed(37)
    batch, heads, dim, window = 1, 64, 512, 128
    query = (
        torch.randn(
            batch, 1, heads, dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    ).contiguous()
    kv = (
        torch.randn(window, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ).contiguous()
    swa_cache = kv.view(window, 1, 1, dim)
    compressed_cache = torch.zeros(
        1, 1, 1, dim, device="cuda", dtype=torch.bfloat16
    )
    indices = torch.arange(window, device="cuda", dtype=torch.int32).view(1, -1)
    lengths = torch.tensor([window], device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    actual = trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=swa_cache,
        workspace_buffer=workspace,
        sparse_indices=indices,
        compressed_kv_cache=compressed_cache,
        sparse_topk_lens=lengths,
        seq_lens=lengths,
        bmm1_scale=dim**-0.5,
        sinks=sinks,
    )
    expected = sparse_attention_reference(
        query,
        kv.view(1, window, dim),
        indices.view(1, 1, window),
        sinks,
        dim**-0.5,
    )
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=2e-4)
