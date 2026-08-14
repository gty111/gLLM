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
