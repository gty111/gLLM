import pytest
import torch

from gllm.layers.ops.fla.l2norm import l2norm_fwd


# Ragged serving hands these kernels arbitrary token counts.  ``T``/``R`` are
# deliberately RUNTIME arguments (a ``tl.constexpr`` there cost one Triton
# compile per distinct count), so the awkward lengths below -- not multiples of
# ``BT=16``, primes, 1, and values straddling a block boundary -- are what keep
# the masked paths honest.
RAGGED_TOKENS = [1, 2, 3, 7, 13, 15, 16, 17, 31, 33, 64, 127, 257, 1021, 4204]

# The kernels reduce in fp32 and round once to bf16, in a different order than
# torch's reference reduction, so a handful of elements land one bf16 ULP
# (~2**-8 near 1.0) apart.  Bitwise equality is asserted where it IS the
# contract -- between the strided and compact kernels -- not against torch.
_BF16_ULP = 2.0 ** -8


def _assert_l2_normalized(actual: torch.Tensor, x: torch.Tensor) -> None:
    ref = x.float()
    expected = ref / torch.sqrt((ref * ref).sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=2 * _BF16_ULP,
        atol=2 * _BF16_ULP,
    )


@pytest.mark.parametrize("tokens", RAGGED_TOKENS)
def test_l2norm_4d_strided_matches_compact_bitwise(tokens: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    fused = torch.randn(
        (1, tokens, 80, 128), device="cuda", dtype=torch.bfloat16
    )
    x = fused[:, :, :16, :]
    assert x.stride(-1) == 1
    if x.is_contiguous():
        # Degenerate shapes (a single token) leave the head slice contiguous,
        # so ``l2norm_fwd`` never selects the strided kernel and there is
        # nothing to compare.  The compact kernels cover this row count.
        pytest.skip("head slice is contiguous at this shape")

    expected = l2norm_fwd(x.contiguous())
    actual = l2norm_fwd(x)

    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_l2norm_4d_strided_cuda_graph_safe():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    fused = torch.randn((1, 64, 80, 128), device="cuda", dtype=torch.bfloat16)
    x = fused[:, :, :16, :]
    expected = l2norm_fwd(x.contiguous())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = l2norm_fwd(x)
    graph.replay()

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("tokens", RAGGED_TOKENS)
@pytest.mark.parametrize("dim", [64, 128, 512])
def test_l2norm_compact_matches_reference(tokens: int, dim: int):
    """The ``make_block_ptr`` kernel over every ragged row count."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(tokens * 1000 + dim)
    x = torch.randn((tokens, dim), device="cuda", dtype=torch.bfloat16)

    _assert_l2_normalized(l2norm_fwd(x), x)


@pytest.mark.parametrize("tokens", [1, 17, 1021, 4204])
def test_l2norm_wide_feature_matches_reference(tokens: int):
    """Feature dim > 512 takes the one-row-per-program kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(tokens)
    x = torch.randn((tokens, 1024), device="cuda", dtype=torch.bfloat16)

    _assert_l2_normalized(l2norm_fwd(x), x)


def test_l2norm_ragged_lengths_share_one_specialization():
    """``T``/``R`` must stay runtime args, or ragged serving re-JITs per shape.

    A ``tl.constexpr`` row count made every new token count a fresh Triton
    compile (~160 ms each, ~640 ms for the first GDN layer of a cold shape).
    Triton still specializes integers on divisibility-by-16 and ``== 1``, so
    the bound here is a small constant, not one.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from gllm.layers.ops.fla import l2norm as l2norm_mod

    def num_specializations(kernel) -> int:
        caches = getattr(kernel, "device_caches", None)
        if not caches:
            pytest.skip("triton build does not expose device_caches")
        total = 0
        for entry in caches.values():
            total += len(entry[0] if isinstance(entry, tuple) else entry)
        return total

    strided = l2norm_mod.l2norm_fwd_4d_strided_kernel
    before = num_specializations(strided)
    for tokens in (101, 203, 307, 409, 512, 613, 719, 829):
        fused = torch.randn(
            (1, tokens, 80, 128), device="cuda", dtype=torch.bfloat16
        )
        l2norm_fwd(fused[:, :, :16, :])
    added = num_specializations(strided) - before

    # 8 distinct row counts; only the divisibility classes may compile.
    assert added <= 2, f"{added} new specializations for 8 ragged lengths"
