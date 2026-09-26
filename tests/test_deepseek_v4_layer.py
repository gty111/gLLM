"""DeepSeek-V4 paged-attention internals that do not depend on the deleted
token-at-a-time numerical oracle: the per-forward decode-index memo, the
masked scatter commit, and the fused RoPE kernel.
"""

from types import SimpleNamespace

import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_index_cache_is_invalidated_per_forward():
    """The per-forward memo must never serve one batch's indices to another.

    The decode path caches index tensors that depend on the batch's positions.
    They are keyed on the metadata object, which is rebuilt every forward; a
    stale hit would be a silent wrong answer rather than a crash.
    """
    from gllm.layers.attention.deepseek_v4.layer import DeepseekV4Attention

    first = SimpleNamespace(a=1)
    holder = SimpleNamespace(metadata=first)

    cache = DeepseekV4Attention._decode_cache(holder)
    cache["probe"] = "first-batch"
    assert DeepseekV4Attention._decode_cache(holder) is cache, "same forward reuses"

    # A new forward installs a new metadata object; the memo must reset.
    holder.metadata = SimpleNamespace(a=2)
    fresh = DeepseekV4Attention._decode_cache(holder)
    assert fresh is not cache
    assert "probe" not in fresh

    # Identity, not equality: a distinct object that compares equal is still a
    # new forward.
    holder.metadata = SimpleNamespace(a=2)
    assert DeepseekV4Attention._decode_cache(holder) is not fresh


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "pages,rows,width,batch", [(8, 4, 576, 1), (16, 8, 576, 5), (4, 2, 64, 3)]
)
def test_scatter_rows_where_matches_read_modify_write(pages, rows, width, batch):
    """The masked commit must be identical to the gather/where/scatter it replaces."""
    from gllm.layers.ops.deepseek_v4.scatter import scatter_rows_where

    torch.manual_seed(pages + batch)
    fused = torch.randn(pages, rows, width, device="cuda", dtype=torch.bfloat16)
    reference = fused.clone()
    page_ids = torch.randint(0, pages, (batch,), device="cuda", dtype=torch.int64)
    row_ids = torch.randint(0, rows, (batch,), device="cuda", dtype=torch.int64)
    src = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(batch, device="cuda") > 0.5

    scatter_rows_where(fused, page_ids, row_ids, src, mask)
    old = reference[page_ids, row_ids]
    reference[page_ids, row_ids] = torch.where(mask.unsqueeze(1), src, old)

    assert torch.equal(fused, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("value", [False, True])
def test_scatter_rows_where_uniform_mask(value):
    """An all-false mask must leave the cache untouched; all-true writes every row."""
    from gllm.layers.ops.deepseek_v4.scatter import scatter_rows_where

    cache = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    before = cache.clone()
    page_ids = torch.arange(4, device="cuda", dtype=torch.int64)
    row_ids = torch.zeros(4, device="cuda", dtype=torch.int64)
    src = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((4,), value, device="cuda", dtype=torch.bool)

    scatter_rows_where(cache, page_ids, row_ids, src, mask)
    if value:
        assert torch.equal(cache[page_ids, row_ids], src)
    else:
        assert torch.equal(cache, before)


def _rope_reference(x, frequencies, *, inverse=False):
    """Plain-PyTorch RoPE oracle, kept here so it cannot drift with the kernel."""
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if complex_x.ndim == 3:
        if frequencies.ndim == 2:
            frequencies = frequencies.view(1, complex_x.size(1), complex_x.size(-1))
    else:
        if frequencies.ndim == 2:
            frequencies = frequencies.view(
                1, complex_x.size(1), 1, complex_x.size(-1)
            )
        elif frequencies.ndim == 3:
            frequencies = frequencies.unsqueeze(-2)
    if inverse:
        frequencies = frequencies.conj()
    return torch.view_as_real(complex_x * frequencies).flatten(-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,freq_shape",
    [
        ((4, 1, 16, 128), (4, 1, 32)),   # decode q / indexer: per-row frequencies
        ((2, 7, 8, 128), (7, 32)),       # prefill: one row per position
        ((3, 1, 576), (3, 1, 32)),       # kv latent, 3D
        ((1, 1, 4, 128), (1, 1, 32)),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_fused_rope_is_bit_exact(shape, freq_shape, inverse):
    """RoPE feeds attention scores directly; a drifted rotation is silent."""
    from gllm.layers.attention.deepseek_v4.ops import apply_rope_inplace

    torch.manual_seed(sum(shape))
    rope_dim = 64
    full = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    got, want = full.clone(), full.clone()
    frequencies = torch.polar(
        torch.ones(*freq_shape, device="cuda"),
        torch.randn(*freq_shape, device="cuda"),
    )

    # Every call site rotates a trailing slice of a wider tensor, so the rows
    # the kernel touches are strided.
    apply_rope_inplace(got[..., -rope_dim:], frequencies, inverse=inverse)
    want[..., -rope_dim:] = _rope_reference(
        want[..., -rope_dim:], frequencies, inverse=inverse
    )
    assert torch.equal(got, want)
