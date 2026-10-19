from types import SimpleNamespace

import pytest
import torch

from gllm.layers.ops.triton_decode_attention import decode_attention_fwd
from gllm.runtime.cache_arena import CacheArena, select_physical_page_bytes
from gllm.runtime.memory_manager import (
    DSparkKVCacheConfig,
    DSparkRequestCache,
    DeepseekV4KVCacheConfig,
    DeepseekV4StateCacheConfig,
    DeepseekV4StateSegment,
    MemoryManager,
    PrefixSegment,
    SSMSegment,
    SSMCacheConfig,
    Segment,
)

from gllm.runtime.sequence import GenerationSequence


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


def test_dspark_kv_layout_uses_draft_geometry():
    cfg = DSparkKVCacheConfig(num_layers=5, kv_head_num=8, kv_head_dim=128)
    manager = MemoryManager(
        0.9, 16, torch.float32, 16, 4, 256, 32,
        dspark_kv_cache_config=cfg,
    )
    layout = manager._dspark_kv_cache_layout()

    assert layout.name == "dspark_kv_cache"
    assert tuple(bank.name for bank in layout.tensors) == ("key", "value")
    assert all(bank.shape == (5, 16, 8, 128) for bank in layout.tensors)
    assert all(bank.dtype == torch.bfloat16 for bank in layout.tensors)
    assert layout.entry_bytes == 2 * 5 * 16 * 8 * 128 * 2


def test_dspark_segment_shares_arena_without_aliasing_live_target_pages():
    cfg = DSparkKVCacheConfig(num_layers=2, kv_head_num=2, kv_head_dim=8)
    manager = MemoryManager(
        0.9, 1, torch.bfloat16, 4, 1, 8, 32,
        dspark_kv_cache_config=cfg,
    )
    target_layout = manager._kv_cache_layout()
    arena, target_cache = _arena_for(target_layout, slots=24)
    target = Segment(1, 4, 1, 8, False, target_cache)
    draft_cache = arena.register_cache(manager._dspark_kv_cache_layout())
    draft = Segment(2, 4, 2, 8, False, draft_cache)

    target_page = target.allocate()
    draft_page = draft.allocate()
    assert target_page is not None and draft_page is not None
    target.k_cache[0][target_page].fill_(11)
    target.v_cache[0][target_page].fill_(13)
    for layer in range(2):
        draft.k_cache[layer][draft_page].fill_(layer + 2)
        draft.v_cache[layer][draft_page].fill_(layer + 5)
        assert draft.k_cache[layer].untyped_storage().data_ptr() == (
            arena.backing.untyped_storage().data_ptr()
        )
        assert torch.all(draft_cache.tensor("key")[draft_page, layer] == layer + 2)
        assert torch.all(draft_cache.tensor("value")[draft_page, layer] == layer + 5)
    assert torch.all(target.k_cache[0][target_page] == 11)
    assert torch.all(target.v_cache[0][target_page] == 13)
    assert draft.k_cache[0].stride(0) == (
        draft_cache.cache_type.pages_per_slot * arena.physical_page_bytes // 2
    )
    assert draft.k_cache[0].stride(0) > draft.k_cache[0][0].numel()
    draft.free(draft_page)
    target.free(target_page)
    assert target.get_num_free_pages() == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
def test_memory_manager_registers_dspark_cache_only_when_configured(monkeypatch, enabled):
    cfg = DSparkKVCacheConfig(num_layers=2, kv_head_num=2, kv_head_dim=8)
    manager = MemoryManager(
        1.0, 1, torch.bfloat16, 4, 1, 8, 32,
        dspark_kv_cache_config=cfg if enabled else None,
    )
    layouts = manager._cache_layouts()
    grain = select_physical_page_bytes(layouts.values())
    kv_bytes = layouts["kv_cache"].entry_bytes
    kv_stride = ((kv_bytes + grain - 1) // grain) * grain
    budget = 24 * kv_stride
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (budget, budget))
    manager.init()

    assert manager.cache_arena.backing.numel() == budget
    assert manager.cache_arena.physical_page_bytes == grain
    assert manager.cache_arena.allocator.cache_type_names == tuple(layouts)
    assert manager.segment.get_num_free_pages() == 24
    if not enabled:
        assert manager.dspark_segment is None
        assert manager.cache_arena.allocator.cache_type_names == ("kv_cache",)
        return
    draft = manager.dspark_segment
    cache = manager.cache_arena.cache("dspark_kv_cache")
    assert draft.k_cache[0].shape == (cache.num_slots, 4, 2, 8)
    assert draft.k_cache[0].untyped_storage().data_ptr() == (
        manager.cache_arena.backing.untyped_storage().data_ptr()
    )
    assert not cache.cache_type.live_slots


def test_dspark_layout_participates_in_physical_grain_selection():
    manager = MemoryManager(
        1.0, 16, torch.bfloat16, 16, 4, 256, 32,
        dspark_kv_cache_config=DSparkKVCacheConfig(5, 8, 128),
    )
    layouts = manager._cache_layouts()
    assert tuple(layouts) == ("kv_cache", "dspark_kv_cache")
    assert layouts["kv_cache"].entry_bytes == 1048576
    assert layouts["dspark_kv_cache"].entry_bytes == 327680
    assert select_physical_page_bytes(layouts.values()) == 65536


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_init_uses_exact_target_and_draft_slot_strides(monkeypatch):
    manager = MemoryManager(
        1.0, 16, torch.bfloat16, 16, 4, 256, 32,
        dspark_kv_cache_config=DSparkKVCacheConfig(5, 8, 128),
    )
    budget = 24 * 1048576
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (budget, budget))
    manager.init(reserve_dummy_page=True)
    arena = manager.cache_arena
    target = arena.cache("kv_cache")
    draft = arena.cache("dspark_kv_cache")
    assert arena.physical_page_bytes == 65536
    assert target.cache_type.pages_per_slot == 16
    assert draft.cache_type.pages_per_slot == 5
    assert target.slot_bytes == 1048576
    assert draft.slot_bytes == 327680
    assert target.num_slots == 24
    assert draft.num_slots == budget // 327680
    assert manager.segment.k_cache[0].stride(0) * 2 == target.slot_bytes
    assert manager.dspark_segment.k_cache[0].stride(0) * 2 == draft.slot_bytes
    target_page = manager.segment.allocate()
    draft_page = manager.dspark_segment.allocate()
    for layer in range(16):
        manager.segment.k_cache[layer][target_page].fill_(11)
        manager.segment.v_cache[layer][target_page].fill_(13)
    for layer in range(5):
        manager.dspark_segment.k_cache[layer][draft_page].fill_(17)
        manager.dspark_segment.v_cache[layer][draft_page].fill_(19)
    assert torch.all(target.tensor("key")[target_page] == 11)
    assert torch.all(target.tensor("value")[target_page] == 13)
    assert torch.all(draft.tensor("key")[draft_page] == 17)
    assert torch.all(draft.tensor("value")[draft_page] == 19)
    assert arena.allocator.num_used_physical_pages == 16 + 16 + 5
    manager.dspark_segment.free(draft_page)
    manager.segment.free(target_page)
    assert arena.allocator.num_used_physical_pages == 16
    assert manager.segment.get_num_free_pages() == 23


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_history_attention_reads_arena_stride():
    torch.manual_seed(0)
    cfg = DSparkKVCacheConfig(num_layers=5, kv_head_num=8, kv_head_dim=128)
    manager = MemoryManager(
        0.9, 1, torch.bfloat16, 16, 1, 128, 32,
        dspark_kv_cache_config=cfg,
    )
    target_layout = manager._kv_cache_layout()
    draft_layout = manager._dspark_kv_cache_layout()
    physical_pages = 4 * (
        (draft_layout.entry_bytes + target_layout.entry_bytes - 1)
        // target_layout.entry_bytes
    )
    arena = CacheArena(
        torch.empty(
            physical_pages * target_layout.entry_bytes,
            dtype=torch.uint8,
            device="cuda",
        ),
        physical_page_bytes=target_layout.entry_bytes,
    )
    cache = arena.register_cache(draft_layout)
    draft = Segment(5, 16, 8, 128, False, cache)
    pages = arena.allocator.allocate(cache.name, 3)
    assert pages is not None
    # Reverse two allocated slots to exercise the logical page table.
    page_ids = [pages[2], pages[0]]
    key, value = draft.k_cache[2], draft.v_cache[2]
    for page in page_ids:
        key[page].normal_()
        value[page].normal_()
    q = torch.randn(7, 32, 128, device="cuda", dtype=torch.bfloat16)
    block_table = torch.tensor(
        [page_ids] * 7, device="cuda", dtype=torch.int32
    )
    seq_lens = torch.full((7,), 21, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    lse = torch.empty(7, 32, device="cuda", dtype=torch.float32)
    logits = torch.empty(7, 32, 4, 129, device="cuda", dtype=torch.float32)
    decode_attention_fwd(
        q, key, value, output, lse, block_table, seq_lens, logits,
        4, 128**-0.5, 16,
    )

    # A short dense oracle is test-only; serving reads the strided pages directly.
    history_k = key[page_ids].reshape(-1, 8, 128)[:21].repeat_interleave(4, dim=1)
    history_v = value[page_ids].reshape(-1, 8, 128)[:21].repeat_interleave(4, dim=1)
    scores = torch.einsum("qhd,shd->qhs", q.float(), history_k.float()) * 128**-0.5
    expected = torch.einsum(
        "qhs,shd->qhd", scores.softmax(-1), history_v.float()
    )
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=0.02, rtol=0.02)
    arena.allocator.free(cache.name, pages)


def _dspark_request_manager(pages=24):
    cfg = DSparkKVCacheConfig(num_layers=2, kv_head_num=2, kv_head_dim=8)
    state = SSMCacheConfig(
        num_layers=1, conv_dim=8, conv_kernel=2,
        num_v_heads=1, head_v_dim=4, head_k_dim=4,
    )
    mm = MemoryManager(
        0.9, 1, torch.bfloat16, 4, 1, 8, 32,
        dspark_kv_cache_config=cfg, ssm_cache_config=state,
    )
    arena, target = _arena_for(mm._kv_cache_layout(), slots=pages)
    mm.cache_arena = arena
    mm.segment = Segment(1, 4, 1, 8, False, target)
    mm.dspark_segment = Segment(
        2, 4, 2, 8, False, arena.register_cache(mm._dspark_kv_cache_layout())
    )
    mm.ssm_segment = SSMSegment(
        state,
        state_cache=arena.register_cache(mm._ssm_cache_layout("ssm_state")),
        snapshot_cache=arena.register_cache(mm._ssm_cache_layout("ssm_snapshot")),
    )
    return mm


def _dspark_sequence(seq_id=1):
    return GenerationSequence(seq_id, [1], [], output_len=32)


def test_dspark_reservation_separates_capacity_from_valid_context_and_resets():
    mm = _dspark_request_manager()
    seq, cache = _dspark_sequence(), DSparkRequestCache()
    allocator = mm.cache_arena.allocator
    owners = list(allocator._owners)
    assert mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=4)
    assert len(seq.page_table) == len(cache.page_table) == 1
    assert cache.context_length == 0
    for layer in range(2):
        mm.dspark_segment.k_cache[layer][cache.page_table[0]].fill_(layer + 1)
        mm.dspark_segment.v_cache[layer][cache.page_table[0]].fill_(layer + 2)
    mm.commit_dspark_context(cache, 3)
    assert mm.reserve_dspark_pages(seq, cache, target_length=5, draft_length=5)
    assert len(seq.page_table) == len(cache.page_table) == 2
    assert cache.context_length == 3
    reserved = list(allocator._owners)
    assert mm.reserve_dspark_pages(seq, cache, target_length=5, draft_length=5)
    assert allocator._owners == reserved
    with pytest.raises(ValueError):
        mm.commit_dspark_context(cache, 9)
    with pytest.raises(ValueError):
        mm.commit_dspark_context(cache, 2)
    assert cache.context_length == 3
    mm.free_dspark_cache(seq, cache)
    assert not seq.page_table and not cache.page_table
    assert cache.context_length == 0 and seq.recurrent_state_slot is None
    assert allocator._owners == owners
    mm.free_dspark_cache(seq, cache)
    assert mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=4)
    assert cache.context_length == 0
    mm.free_dspark_cache(seq, cache)
    assert allocator._owners == owners



@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_gpu_page_table_reuses_storage_and_uploads_only_new_ids(monkeypatch):
    cache = DSparkRequestCache()
    device = torch.device("cuda", torch.cuda.current_device())
    copies = []
    copy = torch.Tensor.copy_

    def track_copy(destination, source, *args, **kwargs):
        if destination.is_cuda:
            copies.append((
                source.device.type, source.numel(),
                source.is_pinned() if not source.is_cuda else False,
                kwargs.get("non_blocking", False),
            ))
        return copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", track_copy)
    assert cache.get_device_page_table(device).shape == (1, 0)
    cache.get_device_page_table(device)
    assert copies == []
    cache.page_table.extend([7, 2, 9])
    first = cache.get_device_page_table(device)
    staging = cache._page_table_staging
    assert first.dtype == torch.int32 and first.tolist() == [[7, 2, 9]]
    assert cache.get_device_page_table(device).data_ptr() == first.data_ptr()
    assert cache._page_table_staging is staging
    assert copies == [("cpu", 3, True, True)]

    cache.page_table.append(4)
    grown = cache.get_device_page_table(device)
    assert grown.data_ptr() != first.data_ptr()
    assert first.tolist() == [[7, 2, 9]]
    assert grown.tolist() == [[7, 2, 9, 4]]
    cache.page_table.append(6)
    assert cache.get_device_page_table(device).data_ptr() == grown.data_ptr()
    assert cache.get_device_page_table(device).tolist() == [[7, 2, 9, 4, 6]]
    assert [entry for entry in copies if entry[0] == "cpu"] == [
        ("cpu", 3, True, True), ("cpu", 1, True, True), ("cpu", 1, True, True),
    ]
    assert [entry for entry in copies if entry[0] == "cuda"] == [
        ("cuda", 3, False, True),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_gpu_page_table_keeps_inflight_staging_immutable(monkeypatch):
    # Hold completion reporting pending so lifetime checks do not depend on
    # whether a tiny transfer happens to finish before the next Python call.
    real_event = torch.cuda.Event
    events = []

    class PendingEvent:
        def __init__(self):
            self.event = real_event()
            self.completed = False
            events.append(self)

        def record(self, stream):
            self.event.record(stream)

        def query(self):
            return self.completed and self.event.query()

    monkeypatch.setattr(torch.cuda, "Event", PendingEvent)
    cache = DSparkRequestCache(page_table=[7, 2])
    device = torch.device("cuda", torch.cuda.current_device())
    first = cache.get_device_page_table(device)
    old_staging = cache._page_table_staging
    cache.page_table.append(9)
    second = cache.get_device_page_table(device)
    active_staging = cache._page_table_staging
    assert any(buffer is old_staging for _, buffer in cache._pending_page_table_staging)
    assert old_staging.tolist() == [7, 2]
    assert active_staging[2].item() == 9

    cache.page_table.append(4)
    third = cache.get_device_page_table(device)
    assert cache._page_table_staging is active_staging
    assert old_staging.tolist() == [7, 2]
    assert active_staging[2:4].tolist() == [9, 4]
    assert first.tolist() == [[7, 2]]
    assert second.tolist() == [[7, 2, 9]]
    assert third.tolist() == [[7, 2, 9, 4]]
    torch.cuda.current_stream(device).synchronize()
    for event in events:
        event.completed = True
    cache.page_table.append(6)
    assert cache.get_device_page_table(device).tolist() == [[7, 2, 9, 4, 6]]
    assert not cache._pending_page_table_staging


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_gpu_page_table_resets_before_request_reuse():
    mm = _dspark_request_manager()
    seq, cache = _dspark_sequence(), DSparkRequestCache()
    device = torch.device("cuda", torch.cuda.current_device())
    assert mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=4)
    old = cache.get_device_page_table(device)
    assert old.tolist() == [cache.page_table]
    mm.commit_dspark_context(cache, 3)
    # Poison the old mirror so this also catches reuse with the same page
    # count/IDs, which a length-only refresh would incorrectly skip.
    old.fill_(-1)
    torch.cuda.current_stream(device).synchronize()
    mm.free_dspark_cache(seq, cache)
    assert cache.page_table == [] and cache.context_length == 0
    assert cache._device_page_table is None
    assert cache._page_table_staging is None and cache._num_uploaded_pages == 0
    assert cache._page_table_copy_done is None
    assert cache._pending_page_table_staging == []
    mm.free_dspark_cache(seq, cache)
    assert mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=4)
    restored = cache.get_device_page_table(device)
    assert restored.data_ptr() != old.data_ptr()
    assert restored.tolist() == [cache.page_table]
    assert cache.context_length == 0
    mm.free_dspark_cache(seq, cache)

def test_dspark_requests_have_independent_pages_and_valid_lengths():
    mm = _dspark_request_manager()
    seq1, seq2 = _dspark_sequence(1), _dspark_sequence(2)
    first, second = DSparkRequestCache(), DSparkRequestCache()
    assert mm.reserve_dspark_pages(seq1, first, target_length=4, draft_length=4)
    assert mm.reserve_dspark_pages(seq2, second, target_length=4, draft_length=4)
    assert set(first.page_table).isdisjoint(second.page_table)
    mm.commit_dspark_context(first, 3)
    assert second.context_length == 0
    mm.free_dspark_cache(seq1, first)
    assert len(second.page_table) == 1 and seq2.recurrent_state_slot is not None
    mm.free_dspark_cache(seq2, second)


@pytest.mark.parametrize("failure,existing", [
    ("ssm_state", False),
    ("dspark_kv_cache", False),
    ("kv_cache", False),
    ("dspark_kv_cache", True),
    ("kv_cache", True),
])
def test_dspark_failed_reservation_preserves_existing_ownership(
    monkeypatch, failure, existing
):
    mm = _dspark_request_manager()
    seq, cache = _dspark_sequence(), DSparkRequestCache()
    if existing:
        assert mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=4)
        mm.commit_dspark_context(cache, 3)
    allocator = mm.cache_arena.allocator
    owners = list(allocator._owners)
    target, draft = list(seq.page_table), list(cache.page_table)
    state = seq.recurrent_state_slot
    original = allocator.allocate

    def fail_one_type(name, count=1, **kwargs):
        if name == failure and count:
            return None
        return original(name, count, **kwargs)

    monkeypatch.setattr(allocator, "allocate", fail_one_type)
    assert not mm.reserve_dspark_pages(seq, cache, target_length=8, draft_length=8)
    assert allocator._owners == owners
    assert seq.page_table == target and cache.page_table == draft
    assert cache.context_length == (3 if existing else 0)
    assert seq.recurrent_state_slot == state


def test_dspark_real_exhaustion_rolls_back_state_and_draft_claims():
    mm = _dspark_request_manager(pages=9)
    seq, cache = _dspark_sequence(), DSparkRequestCache()
    allocator = mm.cache_arena.allocator
    owners = list(allocator._owners)
    # After dummy and working state, one four-page draft slot fits, but four
    # target pages do not. Each type alone appears to have enough capacity.
    assert not mm.reserve_dspark_pages(seq, cache, target_length=16, draft_length=4)
    assert allocator._owners == owners
    assert not seq.page_table and not cache.page_table
    assert seq.recurrent_state_slot is None


def test_dspark_cannot_reserve_below_valid_context():
    mm = _dspark_request_manager()
    seq, cache = _dspark_sequence(), DSparkRequestCache()
    assert mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=4)
    mm.commit_dspark_context(cache, 3)
    owners = list(mm.cache_arena.allocator._owners)
    with pytest.raises(ValueError):
        mm.reserve_dspark_pages(seq, cache, target_length=4, draft_length=2)
    assert mm.cache_arena.allocator._owners == owners
    mm.free_dspark_cache(seq, cache)


def test_dspark_rejects_prefix_cache_before_allocating_arena():
    mm = MemoryManager(
        0.9, 1, torch.bfloat16, 4, 1, 8, 32,
        dspark_kv_cache_config=DSparkKVCacheConfig(2, 2, 8),
    )
    with pytest.raises(ValueError, match="does not support prefix caching"):
        mm.init(segment_cls=PrefixSegment)


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
