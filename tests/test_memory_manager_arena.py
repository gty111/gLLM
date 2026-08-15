import pytest
import torch

from gllm.layers.ops.triton_decode_attention import decode_attention_fwd
from gllm.runtime.cache_arena import CacheArena
from gllm.runtime.memory_manager import MemoryManager, SSMSegment, SSMCacheConfig, Segment


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
