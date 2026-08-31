from types import SimpleNamespace

import pytest
import torch

from gllm.layers.ops.triton_decode_attention import decode_attention_fwd
from gllm.runtime.cache_arena import CacheArena
from gllm.runtime.memory_manager import (
    DeepseekV4KVCacheConfig,
    DeepseekV4StateCacheConfig,
    DeepseekV4StateSegment,
    MemoryManager,
    SSMSegment,
    SSMCacheConfig,
    Segment,
)


def _arena_for(layout, slots=12):
    arena = CacheArena(
        torch.empty(slots * layout.entry_bytes, dtype=torch.uint8),
        physical_page_bytes=layout.entry_bytes,
    )
    return arena, arena.register_cache(layout)


def test_explicit_kv_segment_uses_registered_cache_only():
    manager = MemoryManager(
        0.9,
        num_layers=3,
        dtype=torch.bfloat16,
        page_size=4,
        kv_head_num=2,
        kv_head_dim=8,
        vocab_size=32,
    )
    layout = manager._kv_cache_layout()
    arena, cache = _arena_for(layout)
    segment = Segment(3, 4, 2, 8, False, cache)

    assert tuple(tensor.name for tensor in layout.tensors) == ("key", "value")
    assert segment.k_cache[1].shape == (12, 4, 2, 8)
    assert segment.v_cache[1].shape == (12, 4, 2, 8)
    assert segment.k_cache[1].untyped_storage().data_ptr() == (
        arena.backing.untyped_storage().data_ptr()
    )
    page = segment.allocate()
    segment.k_cache[1][page].fill_(2)
    segment.v_cache[1][page].fill_(3)
    assert torch.all(cache.tensor("key")[page, 1] == 2)
    assert torch.all(cache.tensor("value")[page, 1] == 3)
    segment.free(page)


def test_mla_and_index_banks_share_one_registered_page_lifetime():
    manager = MemoryManager(
        0.9,
        num_layers=2,
        dtype=torch.bfloat16,
        page_size=64,
        kv_head_num=1,
        kv_head_dim=576,
        vocab_size=32,
        use_mla=True,
        index_head_dim=128,
        qk_rope_head_dim=64,
        mla_cache_fp8=True,
    )
    layout = manager._kv_cache_layout()
    arena, cache = _arena_for(layout, slots=4)
    segment = Segment(
        2,
        64,
        1,
        576,
        True,
        cache,
        index_head_dim=128,
        qk_rope_head_dim=64,
        mla_cache_fp8=True,
    )

    assert tuple(tensor.name for tensor in layout.tensors) == (
        "mla",
        "index_key",
        "index_key_fp8",
    )
    assert segment.kv_cache[0].shape == (4, 64, 1, 656)
    assert segment.index_k_cache[0].shape == (4, 64, 128)
    assert segment.index_k_fp8_cache[0].shape == (4, 64 * 132)
    assert segment.index_fp8_bytes == 132
    assert manager.get_sizeof_KV_per_page() == layout.entry_bytes
    assert all(
        tensor.untyped_storage().data_ptr()
        == arena.backing.untyped_storage().data_ptr()
        for tensor in (
            segment.kv_cache[0],
            segment.index_k_cache[0],
            segment.index_k_fp8_cache[0],
        )
    )


def test_ssm_segment_consumes_registered_working_and_snapshot_layouts():
    cfg = SSMCacheConfig(
        num_layers=2,
        conv_dim=12,
        conv_kernel=4,
        num_v_heads=2,
        head_v_dim=4,
        head_k_dim=4,
        dtype=torch.float32,
        conv_state_dtype=torch.bfloat16,
    )
    manager = MemoryManager(
        0.9,
        num_layers=2,
        dtype=torch.bfloat16,
        page_size=4,
        kv_head_num=2,
        kv_head_dim=8,
        vocab_size=32,
        ssm_cache_config=cfg,
    )
    kv_layout = manager._kv_cache_layout()
    arena = CacheArena(
        torch.empty(24 * kv_layout.entry_bytes, dtype=torch.uint8),
        physical_page_bytes=kv_layout.entry_bytes,
    )
    arena.register_cache(kv_layout)
    working = arena.register_cache(manager._ssm_cache_layout("ssm_state"))
    snapshot = arena.register_cache(manager._ssm_cache_layout("ssm_snapshot"))
    segment = SSMSegment(cfg, state_cache=working, snapshot_cache=snapshot)

    assert segment.conv_state.shape == (2, working.num_slots, 12, 3)
    assert segment.temporal_state.shape == (2, working.num_slots, 2, 4, 4)
    blocks = segment.allocate_block_table(2)
    assert blocks is not None
    segment.free_block_table(blocks)


def test_dsv4_state_config_packs_variable_compressor_shapes():
    cfg = DeepseekV4StateCacheConfig(
        compress_ratios=[0, 4, 128], head_dim=16, index_head_dim=8
    )

    assert cfg.state_layout(1) == (0, 8 * 32, (8, 32))
    assert cfg.state_layout(1, indexer=True) == (
        8 * 32,
        8 * 32 + 8 * 16,
        (8, 16),
    )
    assert cfg.state_layout(2) == (
        8 * 32 + 8 * 16,
        cfg.state_numel,
        (128, 16),
    )
    with pytest.raises(ValueError, match="no main compressor"):
        cfg.state_layout(0)
    with pytest.raises(ValueError, match="no indexer compressor"):
        cfg.state_layout(2, indexer=True)

    normalized = SimpleNamespace(
        layer_types=["swa", "c4", "c128", "c4"],
        compress_rates={"c4": 4, "c128": 128},
        head_dim=16,
        index_head_dim=8,
        qk_rope_head_dim=8,
        sliding_window=8,
    )
    local = DeepseekV4StateCacheConfig.from_model_config(
        normalized, start_layer=1, end_layer=3
    )
    assert local.compress_ratios == [4, 128]


def test_dsv4_state_segment_is_request_owned_and_resets_reused_slots():
    cfg = DeepseekV4StateCacheConfig(
        compress_ratios=[4, 128], head_dim=16, index_head_dim=8
    )
    manager = MemoryManager(
        0.9,
        num_layers=2,
        dtype=torch.bfloat16,
        page_size=4,
        kv_head_num=2,
        kv_head_dim=8,
        vocab_size=32,
        dsv4_state_cache_config=cfg,
    )
    layout = manager._dsv4_state_cache_layout()
    arena, cache = _arena_for(layout, slots=5)
    segment = DeepseekV4StateSegment(cfg, state_cache=cache)

    first = segment.allocate_block()
    second = segment.allocate_block()
    assert first is not None and second is not None and first != second
    first_state = segment.state_view(first, 0)
    first_state.kv.fill_(3)
    first_state.score.fill_(5)
    assert torch.all(segment.state_view(first, 0).kv == 3)
    assert torch.all(segment.state_view(second, 0).kv == 0)
    assert torch.all(torch.isneginf(segment.state_view(second, 0).score))

    slots = torch.tensor([first, second], dtype=torch.int32)
    gathered = segment.gather_states(slots, 0, indexer=True)
    gathered.kv[0].fill_(7)
    gathered.score[0].fill_(11)
    segment.store_states(slots, 0, gathered, indexer=True)
    assert torch.all(segment.state_view(first, 0, indexer=True).kv == 7)
    assert torch.all(segment.state_view(first, 0, indexer=True).score == 11)
    assert torch.all(segment.state_view(second, 0, indexer=True).kv == 0)

    segment.free_block(first)
    reused = segment.allocate_block()
    assert reused == first
    reset = segment.state_view(reused, 0)
    assert torch.all(reset.kv == 0)
    assert torch.all(torch.isneginf(reset.score))


def test_dsv4_paged_banks_share_page_lifetime_and_map_compressed_rows():
    cfg = DeepseekV4KVCacheConfig(
        compress_ratios=[0, 4, 128, 4], head_dim=16, index_head_dim=8
    )
    manager = MemoryManager(
        0.9,
        num_layers=4,
        dtype=torch.bfloat16,
        page_size=64,
        kv_head_num=1,
        kv_head_dim=16,
        vocab_size=32,
        use_mla=True,
        dsv4_kv_cache_config=cfg,
    )
    layout = manager._kv_cache_layout()
    arena, cache = _arena_for(layout, slots=3)
    segment = Segment(
        4,
        64,
        1,
        16,
        True,
        cache,
        dsv4_kv_cache_config=cfg,
    )

    # The sliding window lives in the per-request ring, so the paged banks
    # hold compressed state only.
    assert tuple(tensor.name for tensor in layout.tensors) == (
        "dsv4_c4",
        "dsv4_c4_index",
        "dsv4_c128",
    )
    assert segment.dsv4_compressed_cache[1].shape == (3, 16, 16)
    assert segment.dsv4_index_cache[3].shape == (3, 16, 8)
    assert segment.dsv4_compressed_cache[2].shape == (3, 1, 16)
    assert segment.dsv4_compressed_cache[0] is None
    assert cfg.compressed_page_position(15, 64, 4) == (0, 15)
    assert cfg.compressed_page_position(16, 64, 4) == (1, 0)
    assert cfg.compressed_page_position(0, 64, 128) == (1, 0)
    assert cfg.compressed_page_position(1, 64, 128) == (3, 0)

    page = segment.allocate()
    page2 = segment.allocate()
    segment.dsv4_compressed_cache[1][page].fill_(2)
    segment.dsv4_index_cache[1][page].fill_(3)
    assert all(
        tensor.untyped_storage().data_ptr()
        == arena.backing.untyped_storage().data_ptr()
        for tensor in (
            segment.dsv4_compressed_cache[1],
            segment.dsv4_index_cache[1],
        )
    )
    page_table = [page, page2]
    compressed_values = torch.stack(
        [torch.full((16,), 5, dtype=torch.bfloat16),
         torch.full((16,), 7, dtype=torch.bfloat16)]
    )
    segment.store_dsv4_compressed(
        1, page_table, [15, 16], compressed_values
    )
    torch.testing.assert_close(
        segment.gather_dsv4_compressed(1, page_table, [15, 16]),
        compressed_values,
        rtol=0,
        atol=0,
    )
    segment.free(page)
    segment.free(page2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_mla_decode_honors_arena_block_stride():
    torch.manual_seed(0)
    batch, heads, dim, page_size, num_blocks, num_splits = 3, 8, 32, 16, 12, 4
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    key_backing = torch.randn(
        num_blocks, 2, page_size, 1, dim, device="cuda", dtype=torch.bfloat16
    )
    value_backing = torch.randn_like(key_backing)
    key_strided = key_backing[:, 0]
    value_strided = value_backing[:, 0]
    block_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([61, 55, 48], device="cuda", dtype=torch.int32)

    def run(key, value):
        output = torch.empty_like(q)
        lse = torch.empty(batch, heads, device="cuda", dtype=torch.bfloat16)
        logits = torch.empty(
            batch,
            heads,
            num_splits,
            dim + 1,
            device="cuda",
            dtype=torch.float32,
        )
        decode_attention_fwd(
            q,
            key,
            value,
            output,
            lse,
            block_table,
            seq_lens,
            logits,
            num_splits,
            dim**-0.5,
            page_size,
        )
        return output, lse

    contiguous = run(key_strided.contiguous(), value_strided.contiguous())
    strided = run(key_strided, value_strided)
    torch.cuda.synchronize()
    assert torch.equal(contiguous[0], strided[0])
    assert torch.equal(contiguous[1], strided[1])
