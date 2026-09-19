"""Retention order must preserve a completed prefix across other requests."""
import random
from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.cache_arena import CacheArenaAllocator, CacheArena
from gllm.runtime.memory_manager import MemoryManager, PrefixMemoryManager, PrefixSegment
from gllm.runtime.sequence import GenerationSequence


def allocator(pages=12):
    arena = CacheArenaAllocator(pages)
    arena.register_type("kv", 1, 1)
    arena.register_type("state", 3, 1, prefer_high=True)
    return arena


def cache_and_free(arena, pages):
    arena.mark_cached("kv", reversed(pages))
    arena.free("kv", pages)


def test_clean_pages_before_recent_prefix_and_tail_before_head():
    arena = allocator(12)
    pages = arena.allocate("kv", 4)
    cache_and_free(arena, pages)
    assert arena.allocate("kv", 8) == list(range(4, 12))
    assert arena.allocate("kv", 4) == [3, 2, 1, 0]


def test_older_request_before_newer_request():
    arena = allocator(6)
    older = arena.allocate("kv", 3)
    newer = arena.allocate("kv", 3)
    cache_and_free(arena, older)
    cache_and_free(arena, newer)
    assert arena.allocate("kv", 6) == [2, 1, 0, 5, 4, 3]


def test_prefix_hit_refreshes_age_without_invalidating_contents():
    arena = allocator(6)
    evictions = []
    arena.set_evictor("kv", evictions.append)
    older = arena.allocate("kv", 3)
    newer = arena.allocate("kv", 3)
    cache_and_free(arena, older)
    cache_and_free(arena, newer)
    evictions.clear()
    for page in older:
        assert arena.allocate("kv", slot=page, retain=True) == [page]
    assert evictions == []
    cache_and_free(arena, older)
    assert arena.allocate("kv", 6) == [5, 4, 3, 2, 1, 0]


def test_aligned_state_prefers_clean_extent_over_high_cached_extent():
    arena = allocator(12)
    for page in [9, 10, 11]:
        arena.allocate("kv", slot=page)
    cache_and_free(arena, [9, 10, 11])
    assert arena.allocate("state") == [2]
    assert arena.allocate("state") == [1]
    assert arena.allocate("state") == [0]
    assert arena.allocate("state") == [3]


def test_aligned_state_reclaims_oldest_extent_and_clears_stale_priorities():
    arena = allocator(12)
    pages = arena.allocate("kv", 12)
    # The oldest extent has a low address; this overrides prefer_high.
    for start in range(0, 12, 3):
        cache_and_free(arena, pages[start:start + 3])
    assert arena.allocate("state") == [0]
    arena.free("state", [0])
    assert arena.allocate("kv", 3) == [0, 1, 2]


def test_pinned_pages_are_not_reclaimed_and_failed_batch_is_atomic():
    arena = allocator(6)
    pages = arena.allocate("kv", 6)
    cache_and_free(arena, pages[1:])
    # Only the second aligned extent is free; do not partially allocate it.
    assert arena.allocate("state", 2) is None
    assert arena.allocate("state") == [1]
    assert arena.allocate("kv", 2) == [2, 1]
    assert arena.allocate("kv") is None
    assert arena.allocate("kv", slot=0, retain=True) == [0]


def test_late_registered_type_observes_retained_cache():
    arena = allocator(6)
    pages = arena.allocate("kv", 6)
    cache_and_free(arena, pages[3:])
    arena.free("kv", pages[:3])
    arena.register_type("late", 3, 1, prefer_high=True)
    assert arena.allocate("late") == [0]


def test_repeated_exact_slot_hits_bound_stale_heap_entries():
    arena = allocator(6)
    for _ in range(1000):
        arena.allocate("kv", slot=0, retain=True)
        cache_and_free(arena, [0])
    for typ in arena._types.values():
        assert len(typ.free_heap) <= 4 * typ.num_slots + 64
    assert arena.allocate("kv", 6) == [1, 2, 3, 4, 5, 0]


def prefix_manager(pages=128):
    base = MemoryManager(0.85, num_layers=1, dtype=torch.bfloat16, page_size=16,
                         kv_head_num=1, kv_head_dim=8, vocab_size=1024)
    layout = base._kv_cache_layout()
    arena = CacheArena(torch.empty(pages * layout.entry_bytes, dtype=torch.uint8),
                       physical_page_bytes=layout.entry_bytes)
    cache = arena.register_cache(layout)
    segment = PrefixSegment(1, 16, 1, 8, False, cache)
    arena.allocator.set_evictor(cache.name, segment.evict_arena_slot)
    manager = PrefixMemoryManager.__new__(PrefixMemoryManager)
    manager.segment = segment
    manager.page_size = 16
    manager.ssm_segment = manager.dsv4_state_segment = None
    manager.num_allocated_pages = manager.num_hit_pages = 0
    return manager, arena, cache


def populate(segment, seq):
    seq.page_table = [segment.allocate(seq, end)
                      for end in range(16, len(seq.token_ids) + 1, 16)]


def test_parallel_decode_does_not_destroy_recently_released_prefix():
    manager, arena, cache = prefix_manager()
    seg = manager.segment
    long = GenerationSequence(1, list(range(512)), [], output_len=16)
    populate(seg, long)
    short = GenerationSequence(2, [999] * 16, [], output_len=128)
    populate(seg, short)
    seg.free_many(long.page_table)
    # Previously reused page0, losing the entire512-token prefix.
    next_decode_page = seg.allocate()
    assert next_decode_page not in long.page_table
    continuation = GenerationSequence(3, long.token_ids + [777], [], output_len=16)
    manager.pre_allocate_computed_page([continuation])
    assert continuation.computed_token_num == 512


def test_shared_pages_only_become_reclaimable_after_last_reference():
    manager, arena, cache = prefix_manager(pages=4)
    seg = manager.segment
    a = GenerationSequence(1, list(range(64)), [], output_len=16)
    populate(seg, a)
    b = GenerationSequence(2, a.token_ids + [99], [], output_len=16)
    manager.pre_allocate_computed_page([b])
    seg.free_many(a.page_table)
    assert arena.allocator.allocate(cache.name) is None
    seg.free_many(b.page_table)
    assert [seg.allocate() for _ in range(4)] == list(reversed(a.page_table))


def test_batch_retention_is_atomic_on_conflict_or_invalid_slot():
    arena = allocator(12)
    arena.allocate("state", slot=2)
    before = (arena._owners[:], arena.num_used_physical_pages)
    assert arena.retain_many("kv", [0, 1, 6]) is None
    assert (arena._owners, arena.num_used_physical_pages) == before
    with pytest.raises(IndexError):
        arena.retain_many("kv", [0, 1, 12])
    assert (arena._owners, arena.num_used_physical_pages) == before
    assert arena.retain_many("kv", [0, 1, 0]) == [0, 1]
    assert arena.num_used_physical_pages == before[1] + 2


@pytest.mark.parametrize("seed", range(8))
def test_batch_retention_matches_scalar_with_fragmentation_and_live_pages(seed):
    rng = random.Random(seed)
    pages = list(range(24))
    rng.shuffle(pages)
    released = pages[:18]
    hits = pages[::2] + pages[1:4]
    scalar, batch = allocator(24), allocator(24)
    events = [[], []]
    for arena, recorded in zip((scalar, batch), events):
        arena.allocate("kv", 24)
        cache_and_free(arena, released)
        for name in ("kv", "state"):
            arena.set_evictor(name, lambda slot, n=name, out=recorded: out.append((n, slot)))
    for page in hits:
        scalar.allocate("kv", slot=page, retain=True)
    batch.retain_many("kv", hits)
    assert batch._owners == scalar._owners
    assert batch._cached_pages == scalar._cached_pages
    assert batch.num_used_physical_pages == scalar.num_used_physical_pages
    # Batch eviction calls each stale view once even if several KV pages overlap.
    assert set(events[0]) == set(events[1])
    assert not any(name == "kv" for name, _ in events[1])
    for name in scalar._types:
        a, b = scalar.cache_type(name), batch.cache_type(name)
        assert a.occupied_pages == b.occupied_pages
        assert a.free_slots == b.free_slots
        assert a.live_slots == b.live_slots
        assert a.retained_slots == b.retained_slots
    for arena in (scalar, batch):
        cache_and_free(arena, list(dict.fromkeys(hits)))
    count = len(scalar.cache_type("kv").free_slots)
    assert scalar.allocate("kv", count) == batch.allocate("kv", count)


def test_batch_retention_coalesces_contiguous_candidate_updates():
    arena = allocator(120)
    pages = arena.allocate("kv", 96)
    cache_and_free(arena, pages)
    updates = []
    original = arena._update_candidates

    def record(start, end, delta):
        updates.append((start, end, delta))
        original(start, end, delta)

    arena._update_candidates = record
    arena.retain_many("kv", reversed(pages))
    assert updates == [(0, 96, 1)]


def test_batch_retention_invalidates_other_retained_views_without_callback():
    arena = allocator(12)
    arena.allocate("state", slot=0)
    arena.mark_cached("state", [0])
    arena.free("state", [0])
    arena.retain_many("kv", [1])
    assert not arena.cache_type("state").retained_slots
    assert arena._cached_pages[:3] == [0, 0, 0]
    assert arena.cache_type("kv").cached_priority[0] == 0
    assert arena.cache_type("kv").cached_priority[2] == 0


def test_batch_retains_wide_slots_and_preserves_live_ownership():
    arena = allocator(12)
    arena.allocate("state", slot=1)
    arena.retain_many("state", [0, 1, 2, 0])
    assert arena.num_used_physical_pages == 9
    assert arena.cache_type("state").live_slots == {0, 1, 2}
    assert arena.cache_type("kv").free_slots == {9, 10, 11}


@pytest.mark.parametrize("miss_page", [0, 1, 3])
def test_batch_lookup_stops_at_first_miss_and_hashes_lazily(miss_page):
    manager, arena, cache = prefix_manager(8)
    seq = GenerationSequence(1, list(range(64)), [], output_len=16)
    populate(manager.segment, seq)
    manager.segment.free_many(seq.page_table)
    arena.allocator.allocate(cache.name, slot=seq.page_table[miss_page])
    continuation = GenerationSequence(2, seq.token_ids + [99], [], output_len=16)
    manager.pre_allocate_computed_page([continuation])
    assert continuation.page_table == seq.page_table[:miss_page]
    assert continuation.computed_token_num == miss_page * 16
    assert len(continuation._page_hashes) == miss_page + 1


def test_batch_lookup_uses_multimodal_hash_source_and_invalidates_old_hashes():
    manager, arena, cache = prefix_manager(8)
    seg = manager.segment
    seq = GenerationSequence(1, [7] * 64, [], output_len=16)
    seq.hash_token_ids = list(range(64))
    populate(seg, seq)
    seg.free_many(seq.page_table)
    continuation = GenerationSequence(2, seq.token_ids + [99], [], output_len=16)
    manager.pre_allocate_computed_page([continuation])
    assert continuation.page_table == []
    continuation.hash_token_ids = seq.hash_token_ids + [99]
    manager.pre_allocate_computed_page([continuation])
    assert continuation.page_table == seq.page_table
    assert continuation.computed_token_num == 64


def test_batch_lookup_canary_mismatch_does_not_pin_page():
    manager, arena, cache = prefix_manager(8)
    seq = GenerationSequence(1, list(range(64)), [], output_len=16)
    populate(manager.segment, seq)
    manager.segment.free_many(seq.page_table)
    manager.segment.page2canary[seq.page_table[1]] = (-1,)
    continuation = GenerationSequence(2, seq.token_ids + [99], [], output_len=16)
    manager.pre_allocate_computed_page([continuation])
    assert continuation.page_table == seq.page_table[:1]
    assert manager.segment.page_ref_num[seq.page_table[1]] == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_batch_full_hit_preserves_rollback_and_shared_references(hybrid):
    manager, arena, cache = prefix_manager(8)
    seq = GenerationSequence(1, list(range(64)), [], output_len=16)
    populate(manager.segment, seq)
    manager.segment.free_many(seq.page_table)
    restored = []
    if hybrid:
        manager.ssm_segment = object()
        manager._restore_ssm_working_state = lambda s: restored.append(s.computed_token_num)
    a = GenerationSequence(2, seq.token_ids[:], [], output_len=16)
    b = GenerationSequence(3, seq.token_ids[:], [], output_len=16)
    manager.pre_allocate_computed_page([a, b])
    assert a.page_table == b.page_table == seq.page_table
    assert a.computed_token_num == b.computed_token_num == (48 if hybrid else 63)
    assert manager.num_hit_pages == (6 if hybrid else 8)
    assert restored == ([48, 48] if hybrid else [])
    assert all(manager.segment.page_ref_num[p] == 2 for p in seq.page_table)
    manager.segment.free_many(a.page_table)
    assert arena.allocator.num_used_physical_pages == 4
    manager.segment.free_many(b.page_table)
    assert arena.allocator.num_used_physical_pages == 0


@pytest.mark.parametrize("last_snapshot_filled", [False, True])
def test_batch_lookup_pins_all_pages_before_restoring_filled_snapshot(
    monkeypatch, last_snapshot_filled
):
    monkeypatch.setattr("gllm.runtime.memory_manager.get_pp_size", lambda: 1)
    manager, arena, cache = prefix_manager(16)
    arena.register_cache_type("state", 3 * cache.layout.entry_bytes, prefer_high=True)
    arena.register_cache_type("snapshot", 3 * cache.layout.entry_bytes, prefer_high=True)
    seg = manager.segment
    seq = GenerationSequence(1, list(range(64)), [], output_len=16)
    populate(seg, seq)
    snapshots = arena.allocator.allocate("snapshot", 2)
    for page, slot, filled in zip(
        seq.page_table[2:], snapshots, [True, last_snapshot_filled]
    ):
        seg.page2ssm_snapshot[page] = slot
        seg.page2ssm_snapshot_valid[page] = filled
    seg.free_many(seq.page_table)
    copies = []
    manager.ssm_segment = SimpleNamespace(copy_state=lambda *args: copies.append(args))

    def allocate_working(continuation):
        assert all(seg.page_ref_num[p] == 1 for p in seq.page_table)
        assert all(p in arena.allocator.cache_type(cache.name).live_slots for p in seq.page_table)
        continuation.recurrent_state_slot = arena.allocator.allocate("state")[0]
        return True

    manager.allocate_recurrent_slot = allocate_working
    continuation = GenerationSequence(2, seq.token_ids + [99], [], output_len=16)
    manager.pre_allocate_computed_page([continuation])
    assert continuation.computed_token_num == (64 if last_snapshot_filled else 48)
    expected_slot = snapshots[1 if last_snapshot_filled else 0]
    assert copies == [("snapshot", expected_slot, "working", continuation.recurrent_state_slot)]
    assert continuation.page_table == seq.page_table


def test_unregistered_tail_is_used_before_cached_full_pages():
    manager, arena, cache = prefix_manager(pages=3)
    seg = manager.segment
    seq = GenerationSequence(1, list(range(33)), [], output_len=16)
    populate(seg, seq)
    tail = seg.allocate()
    seg.free_many(seq.page_table + [tail])
    assert seg.allocate() == tail
    assert seg.allocate() == seq.page_table[-1]
    assert seg.allocate() == seq.page_table[0]


def test_single_page_release_retains_valid_cache_priority():
    manager, arena, cache = prefix_manager(pages=3)
    seg = manager.segment
    seq = GenerationSequence(1, list(range(16)), [], output_len=16)
    populate(seg, seq)
    seg.free(seq.page_table[0])
    assert seg.allocate() != seq.page_table[0]


def test_snapshot_is_kept_until_its_cached_kv_page_is_actually_reused():
    manager, arena, cache = prefix_manager(pages=3)
    seg = manager.segment
    seq = GenerationSequence(1, list(range(16)), [], output_len=16)
    populate(seg, seq)
    page = seq.page_table[0]
    released = []
    seg.ssm_segment = SimpleNamespace(free_snapshot=released.append)
    seg.page2ssm_snapshot[page] = 7
    seg.page2ssm_snapshot_valid[page] = True
    seg._ssm_snapshot_lru[page] = None
    seg.free_many(seq.page_table)
    assert released == []
    assert [seg.allocate(), seg.allocate()] == [1, 2]
    assert released == []
    assert seg.allocate() == page
    assert released == [7]
    assert not seg.page2ssm_snapshot_valid[page]


def test_mark_cached_rejects_unowned_slots():
    arena = allocator()
    with pytest.raises(RuntimeError, match="unowned"):
        arena.mark_cached("kv", [0])


def test_partial_overwrite_invalidates_entire_retained_wide_view():
    arena = allocator(6)
    arena.allocate("state", slot=0)
    arena.mark_cached("state", [0])
    arena.free("state", [0])
    arena.allocate("kv", slot=0)
    # Overwriting one physical page invalidates the whole old state view;
    # the remaining bytes must not keep a phantom cache priority.
    assert arena._cached_pages[:3] == [0, 0, 0]
    assert not arena.cache_type("state").retained_slots
    assert arena.allocate("kv", 2) == [1, 2]


def test_retain_hit_avoids_per_page_cache_priority_rebuild(monkeypatch):
    arena = allocator(12)
    pages = arena.allocate("kv", 12)
    cache_and_free(arena, pages)
    updates = []
    monkeypatch.setattr(arena, "_update_cached_priorities",
                        lambda *args: updates.append(args))
    for page in pages:
        arena.allocate("kv", slot=page, retain=True)
    assert updates == []


def test_released_page_with_replaced_hash_mapping_becomes_uncached():
    manager, arena, cache = prefix_manager(pages=3)
    seg = manager.segment
    a = GenerationSequence(1, list(range(16)), [], output_len=16)
    populate(seg, a)
    seg.free_many(a.page_table)
    b = GenerationSequence(2, a.token_ids + [99], [], output_len=16)
    manager.pre_allocate_computed_page([b])
    # A recomputed copy may become the hash table's canonical entry while the
    # retained old copy remains pinned by another request.
    c = GenerationSequence(3, list(a.token_ids), [], output_len=16)
    populate(seg, c)
    seg.free_many(b.page_table)
    assert not arena.allocator.cache_type(cache.name).retained_slots
    assert seg.allocate() == a.page_table[0]
