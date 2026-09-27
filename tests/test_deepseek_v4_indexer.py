import pytest
import torch

from gllm.layers.attention.deepseek_v4.indexer import (
    causal_indexer_topk,
    indexer_scores,
    mxfp4_fake_quantize,
    normalized_hadamard,
    prepare_indexer_query_and_weights,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_normalized_hadamard_matches_official_dependency():
    fast_hadamard_transform = pytest.importorskip("fast_hadamard_transform")
    torch.manual_seed(53)
    x = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    expected = fast_hadamard_transform.hadamard_transform(
        x, scale=128**-0.5
    )
    torch.testing.assert_close(normalized_hadamard(x), expected, rtol=0, atol=0)


def test_normalized_hadamard_rejects_non_bfloat16_input():
    with pytest.raises(TypeError, match="must be bfloat16"):
        normalized_hadamard(torch.randn(2, 128))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_fake_quantize_uses_e2m1_levels_per_32():
    x = torch.zeros(1, 32, dtype=torch.float32, device="cuda")
    x[0, :8] = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda"
    )
    torch.testing.assert_close(mxfp4_fake_quantize(x), x, rtol=0, atol=0)
    # Amax=12 selects scale=2, so the same E2M1 levels are doubled.
    torch.testing.assert_close(mxfp4_fake_quantize(x * 2), x * 2, rtol=0, atol=0)


def test_index_scores_and_causal_topk_match_explicit_formula():
    torch.manual_seed(59)
    q = torch.randn(1, 8, 4, 16, dtype=torch.bfloat16)
    kv = torch.randn(1, 2, 16, dtype=torch.bfloat16)
    weights = torch.randn(1, 8, 4, dtype=torch.bfloat16)
    scores = indexer_scores(q, kv, weights)
    expected = torch.einsum("bshd,btd->bsht", q, kv)
    expected = (expected.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    assert scores.dtype == torch.bfloat16

    selected = causal_indexer_topk(
        scores, compress_ratio=4, start_pos=0, topk=2, offset=128
    )
    assert torch.all(selected[:, :3] == -1)
    assert torch.all(selected[:, 3, 1:] == -1)
    assert torch.all((selected[:, 7] >= 128) & (selected[:, 7] < 130))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_indexer_query_accepts_tp_local_heads_with_global_scaling():
    torch.manual_seed(61)
    query = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, 2, 16, device="cuda", dtype=torch.bfloat16)
    actual_query, actual_weights = prepare_indexer_query_and_weights(
        query, weights, num_heads=64
    )
    expected_query = mxfp4_fake_quantize(normalized_hadamard(query))
    expected_weights = weights * (128**-0.5 * 64**-0.5)
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


# --- fused MXFP4 QAT kernel ----------------------------------------------
#
# ``mxfp4_fake_quantize`` dispatches to a Triton kernel on CUDA. It sits inside
# the indexer's top-k selection, so a drifted code would silently change which
# compressed positions attention reads -- and never raise.


def _reference_mxfp4(x, group_size=32):
    """Independent MXFP4 E2M1/E8M0 oracle, kept in the test on purpose.

    ``mxfp4_fake_quantize`` is a single Triton kernel in production. Spelling
    the spec out here in plain PyTorch means the oracle cannot drift along with
    the implementation it checks -- and it documents exactly what the E2M1
    ladder is.
    """
    shape = x.shape
    grouped = x.float().reshape(-1, shape[-1] // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * (2.0**-126))
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    scaled = grouped / scale

    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int64)
    for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += magnitude > midpoint
    # E2M1 magnitudes are [0, .5, 1, 1.5, 2, 3, 4, 6].
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device
    )
    magnitude_q = levels[code]
    quantized = magnitude_q * torch.sign(scaled) * scale
    return quantized.reshape(shape).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(1, 1, 16, 128), (4, 1, 16, 128),
                                   (2, 7, 16, 128), (3, 5, 8, 256), (1, 1, 1, 64)])
@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e3])
def test_fused_mxfp4_is_bit_exact(shape, magnitude):
    torch.manual_seed(sum(shape))
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * magnitude
    assert torch.equal(mxfp4_fake_quantize(x), _reference_mxfp4(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "name",
    ["zeros", "denormal", "negative", "at_midpoints", "at_codepoints"],
)
def test_fused_mxfp4_edge_values(name):
    """Zero, denormals and exact ladder boundaries are where rounding splits."""
    shape = (2, 1, 4, 128)
    if name == "zeros":
        x = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    elif name == "denormal":
        x = torch.full(shape, 1e-38, device="cuda", dtype=torch.bfloat16)
    elif name == "negative":
        x = -torch.rand(shape, device="cuda", dtype=torch.bfloat16)
    elif name == "at_midpoints":
        ladder = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.0],
                              device="cuda", dtype=torch.bfloat16)
        x = ladder.repeat(2 * 1 * 4 * 16).reshape(shape)
    else:
        codes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                             device="cuda", dtype=torch.bfloat16)
        x = codes.repeat(2 * 1 * 4 * 16).reshape(shape)
    assert torch.equal(mxfp4_fake_quantize(x), _reference_mxfp4(x))


