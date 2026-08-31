"""The packed FP8 window cache must be a pure storage change.

DeepSeek-V4 rounds the NoPE half of every cached KV row through E4M3 with a
power-of-two per-64-group scale before it is stored, so keeping the codes
instead of the dequantized BF16 cannot lose information. These tests pin that
down, because the moment it stops being exact the cache stops being free.
"""

import pytest
import torch

from gllm.layers.attention.deepseek_v4.kv_fp8 import (
    gather_raw_fp8,
    pack_raw_fp8,
    raw_row_bytes,
    supports_packed_layout,
)
from gllm.layers.attention.deepseek_v4.ops import fp8_fake_quantize_inplace


HEAD_DIM = 512
ROPE_DIM = 64


def test_packed_row_matches_the_ds_mla_width():
    # 448 NoPE bytes + 64 RoPE BF16 (128 B) + 8 scale bytes.
    assert raw_row_bytes(HEAD_DIM, ROPE_DIM) == 584
    assert raw_row_bytes(HEAD_DIM, ROPE_DIM) < HEAD_DIM * 2
    assert supports_packed_layout(HEAD_DIM, ROPE_DIM)
    # Geometries without whole 64-wide NoPE groups cannot be packed.
    assert not supports_packed_layout(100, 64)
    assert not supports_packed_layout(64, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 37, 512])
def test_round_trip_is_bitwise_identical_to_the_bf16_bank(tokens):
    torch.manual_seed(3 + tokens)
    page_size, pages = 64, 16
    kv = torch.randn(tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    # Exactly what the projection hands the cache today.
    fp8_fake_quantize_inplace(kv[:, : HEAD_DIM - ROPE_DIM], group_size=64)

    cache = torch.zeros(
        pages, page_size, raw_row_bytes(HEAD_DIM, ROPE_DIM),
        dtype=torch.uint8, device="cuda",
    )
    slots = torch.randperm(pages * page_size, device="cuda")[:tokens]
    offsets = slots.to(torch.int64) * cache.shape[-1]
    pack_raw_fp8(kv, cache, offsets, rope_dim=ROPE_DIM)
    back = gather_raw_fp8(
        cache, offsets, head_dim=HEAD_DIM, rope_dim=ROPE_DIM
    )
    assert torch.equal(back, kv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_round_trip_survives_degenerate_groups():
    """All-zero, denormal-small, and saturating groups all round-trip."""
    torch.manual_seed(11)
    page_size, pages, tokens = 32, 4, 8
    kv = torch.randn(tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    kv[0, :64] = 0.0
    kv[1, 64:128] = 1e-6
    kv[2, 128:192] = 5.0e4          # saturates E4M3 before scaling
    kv[3, : HEAD_DIM - ROPE_DIM] = torch.linspace(
        -500, 500, HEAD_DIM - ROPE_DIM, device="cuda"
    ).bfloat16()
    kv[4, :64] = -0.0
    fp8_fake_quantize_inplace(kv[:, : HEAD_DIM - ROPE_DIM], group_size=64)

    cache = torch.zeros(
        pages, page_size, raw_row_bytes(HEAD_DIM, ROPE_DIM),
        dtype=torch.uint8, device="cuda",
    )
    slots = torch.arange(tokens, device="cuda")
    offsets = slots.to(torch.int64) * cache.shape[-1]
    pack_raw_fp8(kv, cache, offsets, rope_dim=ROPE_DIM)
    back = gather_raw_fp8(
        cache, offsets, head_dim=HEAD_DIM, rope_dim=ROPE_DIM
    )
    assert torch.equal(back, kv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rope_half_is_stored_verbatim():
    """RoPE dims are positional and never quantized; they must survive exactly."""
    torch.manual_seed(5)
    page_size, pages, tokens = 16, 2, 5
    kv = torch.randn(tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 8
    fp8_fake_quantize_inplace(kv[:, : HEAD_DIM - ROPE_DIM], group_size=64)
    cache = torch.zeros(
        pages, page_size, raw_row_bytes(HEAD_DIM, ROPE_DIM),
        dtype=torch.uint8, device="cuda",
    )
    slots = torch.arange(tokens, device="cuda")
    offsets = slots.to(torch.int64) * cache.shape[-1]
    pack_raw_fp8(kv, cache, offsets, rope_dim=ROPE_DIM)
    back = gather_raw_fp8(
        cache, offsets, head_dim=HEAD_DIM, rope_dim=ROPE_DIM
    )
    assert torch.equal(back[:, HEAD_DIM - ROPE_DIM :], kv[:, HEAD_DIM - ROPE_DIM :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_config_falls_back_to_bf16_for_unpackable_geometry():
    from gllm.runtime.memory_manager import DeepseekV4StateCacheConfig

    packed = DeepseekV4StateCacheConfig([4], head_dim=512, qk_rope_head_dim=64)
    assert packed.window_fp8
    assert packed.window_row_width == 584
    assert packed.window_row_dtype is torch.uint8

    plain = DeepseekV4StateCacheConfig([4], head_dim=16, qk_rope_head_dim=64)
    assert not plain.window_fp8
    assert plain.window_row_width == 16
    assert plain.window_row_dtype is torch.bfloat16
