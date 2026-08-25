import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashInfer RoPE requires CUDA",
)


@torch.inference_mode()
def test_rotary_cache_stays_fp32_with_bf16_model_dtype():
    from gllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        rope = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=16,
            base=10_000,
            is_neox_style=True,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    assert rope.cos_sin_cache.dtype == torch.float32

    positions = torch.tensor([0, 3, 9], device="cuda", dtype=torch.int64)
    query = torch.randn(3, 2 * 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    query_reference = query.clone().view(3, 2, 64)
    key_reference = key.clone().view(3, 1, 64)
    cache = rope.cos_sin_cache[positions]
    cos, sin = cache.chunk(2, dim=-1)
    query_reference = apply_rotary_emb(
        query_reference, cos[:, None, :], sin[:, None, :]
    ).to(query.dtype)
    key_reference = apply_rotary_emb(
        key_reference, cos[:, None, :], sin[:, None, :]
    ).to(key.dtype)

    rope(positions, query, key)

    torch.testing.assert_close(
        query, query_reference.flatten(1), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        key, key_reference.flatten(1), rtol=0.02, atol=0.02
    )


@torch.inference_mode()
def test_partial_neox_rope_single_cat_is_bitwise_exact():
    from gllm.layers.rotary_embedding import apply_rotary_emb

    torch.manual_seed(41)
    x = torch.randn(128, 24, 256, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(128, 1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(128, 1, 32, device="cuda", dtype=torch.bfloat16)

    x_rot = x[..., :64]
    x_pass = x[..., 64:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    reference = torch.cat((torch.cat((o1, o2), dim=-1), x_pass), dim=-1)

    actual = apply_rotary_emb(x, cos, sin, interleaved=False)
    assert torch.equal(actual, reference)


@torch.inference_mode()
def _build_interleaved_rope():
    from gllm.layers.rotary_embedding import MRotaryEmbedding

    return MRotaryEmbedding(
        head_size=256,
        rotary_dim=64,
        max_position_embeddings=512,
        base=10_000,
        is_neox_style=True,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )


@torch.inference_mode()
def test_interleaved_mrope_combined_gather_is_bitwise_exact():
    """The one-shot axis gather must equal the clone-and-patch reference."""
    from gllm.layers.rotary_embedding import apply_interleaved_rope

    torch.manual_seed(43)
    rope = _build_interleaved_rope()
    positions = torch.randint(0, 512, (3, 128), device="cuda")

    cache = rope.cos_sin_cache[positions]
    cos, sin = cache.chunk(2, dim=-1)
    expected_cos = apply_interleaved_rope(cos, rope.mrope_section)
    expected_sin = apply_interleaved_rope(sin, rope.mrope_section)

    axis = rope.interleaved_axis.view(1, -1, 1).expand(128, -1, 1)
    combined = cache.permute(1, 2, 0).gather(2, axis).squeeze(2)
    actual_cos, actual_sin = combined.chunk(2, dim=-1)

    assert torch.equal(actual_cos, expected_cos)
    assert torch.equal(actual_sin, expected_sin)


@torch.inference_mode()
def test_interleaved_mrope_rotates_in_place_without_losing_accuracy():
    """The in-place kernel replaced a gather + elementwise + ``cat`` tail.

    It is NOT bitwise-equal to that tail: the kernel keeps the fp32 cosine and
    sine from the cache and rounds once, while the PyTorch path rounded them to
    bf16 first.  So assert what actually matters -- the result stays within one
    bf16 ULP of an fp32 reference, and is no further from it than the path it
    replaced -- plus that the untouched dims and the input buffers are handled
    as the in-place contract requires.
    """
    from gllm.layers.rotary_embedding import (
        apply_interleaved_rope,
        apply_rotary_emb_dispatch,
    )

    torch.manual_seed(43)
    rope = _build_interleaved_rope()
    positions = torch.randint(0, 512, (3, 128), device="cuda")
    query = torch.randn(128, 24 * 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(128, 4 * 256, device="cuda", dtype=torch.bfloat16)
    query_in, key_in = query.clone(), key.clone()

    cache = rope.cos_sin_cache[positions]
    cos, sin = cache.chunk(2, dim=-1)
    cos = apply_interleaved_rope(cos, rope.mrope_section)
    sin = apply_interleaved_rope(sin, rope.mrope_section)

    # fp32 reference, rounded once at the end.
    ref_q = apply_rotary_emb_dispatch(
        query_in.float().view(128, 24, 256), cos, sin, True
    ).reshape_as(query)
    ref_k = apply_rotary_emb_dispatch(
        key_in.float().view(128, 4, 256), cos, sin, True
    ).reshape_as(key)
    # The bf16 path this change replaced.
    old_q = apply_rotary_emb_dispatch(
        query_in.view(128, 24, 256), cos, sin, True
    ).reshape_as(query)
    old_k = apply_rotary_emb_dispatch(
        key_in.view(128, 4, 256), cos, sin, True
    ).reshape_as(key)

    actual_query, actual_key = rope(positions, query, key)

    # ``triton_mrope`` rotates in place and hands back the same storage.
    assert actual_query.data_ptr() == query.data_ptr()
    assert actual_key.data_ptr() == key.data_ptr()

    bf16_ulp = 2.0**-8
    for actual, old, ref, x_in in (
        (actual_query, old_q, ref_q, query_in),
        (actual_key, old_k, ref_k, key_in),
    ):
        new_err = (actual.float() - ref.float()).abs()
        old_err = (old.float() - ref.float()).abs()
        assert new_err.max() <= 4 * bf16_ulp * ref.float().abs().max()
        assert new_err.mean() <= old_err.mean()
        # Dims past ``rotary_dim`` must come through untouched.
        heads = actual.shape[-1] // 256
        untouched = actual.view(128, heads, 256)[..., 64:]
        assert torch.equal(untouched, x_in.view(128, heads, 256)[..., 64:])
