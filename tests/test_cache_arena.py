import torch

from gllm.runtime.cache_arena import (
    ArenaSlotAllocator,
    CacheArena,
    CacheLayout,
    CacheTensorLayout,
)


def make_arena(num_pages=24, page_bytes=64):
    return CacheArena(
        torch.empty(num_pages * page_bytes, dtype=torch.uint8), page_bytes
    )


def test_registered_cache_types_reuse_the_same_physical_pages():
    arena = make_arena()
    arena.register_cache_type("kv", 64)
    state = arena.register_cache_type("state", 150, prefer_high=True)

    assert state.pages_per_slot == 3
    assert arena.allocator.allocate("state", 2) == [7, 6]
    assert arena.allocator.allocate("kv", 2) == [0, 1]
    assert arena.allocator.num_used_physical_pages == 8

    arena.allocator.free("state", [7])
    # The exact bytes formerly used by state are now ordinary KV candidates.
    allocated = arena.allocator.allocate("kv", 19)
    assert {21, 22, 23}.issubset(allocated)


def test_explicit_cache_retention_does_not_evict_its_own_contents():
    arena = make_arena(4)
    arena.register_cache_type("kv", 64)
    evicted = []
    arena.allocator.set_evictor("kv", evicted.append)

    slot = arena.allocator.allocate("kv")[0]
    assert evicted == [slot]
    arena.allocator.free("kv", [slot])

    # Explicit allocation means a prefix hit: the bytes were not overwritten
    # while unpinned, so metadata/content must be retained.
    assert arena.allocator.allocate("kv", slot=slot, retain=True) == [slot]
    assert evicted == [slot]


def test_overlapping_type_allocation_invalidates_stale_cache_metadata():
    arena = make_arena(6)
    arena.register_cache_type("kv", 64)
    arena.register_cache_type("state", 150)
    evicted = []
    arena.allocator.set_evictor("kv", evicted.append)

    pages = arena.allocator.allocate("kv", 3)
    arena.allocator.free("kv", pages)
    evicted.clear()
    assert arena.allocator.allocate("state", slot=0) == [0]
    assert evicted == [0, 1, 2]


def test_pressure_reclaims_evictable_cache_without_a_fixed_limit():
    arena = make_arena(6)
    arena.register_cache_type("working", 150, prefer_high=True)
    arena.register_cache_type("snapshot", 150, prefer_high=True)
    snapshots = arena.allocator.allocate("snapshot", 2)
    assert snapshots == [1, 0]

    def reclaim_oldest():
        if not snapshots:
            return False
        arena.allocator.free("snapshot", [snapshots.pop()])
        return True

    arena.allocator.set_reclaimer("snapshot", reclaim_oldest)
    # No working slot is initially free. Allocation drives reclamation instead
    # of relying on a pre-partition or snapshot-count watermark.
    assert arena.allocator.allocate("working", 1) == [0]
    assert snapshots == [1]


def test_entry_views_have_stable_type_stride_and_shared_storage():
    arena = make_arena(12)
    state = arena.register_cache_type("state", 150)
    view = arena.entry_view("state", torch.float32, (2, 3), entry_offset_bytes=8)

    assert view.shape == (state.num_slots, 2, 3)
    assert view.stride() == (48, 3, 1)
    assert (
        view.untyped_storage().data_ptr() == arena.backing.untyped_storage().data_ptr()
    )


def test_registered_cache_materializes_banks_with_one_slot_grid():
    layout = CacheLayout(
        "attention",
        (
            CacheTensorLayout("key", torch.bfloat16, (2, 4, 3)),
            CacheTensorLayout("value", torch.float32, (2, 4, 5)),
        ),
    )
    arena = make_arena(num_pages=20, page_bytes=layout.entry_bytes)
    cache = arena.register_cache(layout)

    assert cache.num_slots == 20
    assert cache.tensor("key").shape == (20, 2, 4, 3)
    assert cache.tensor("value").shape == (20, 2, 4, 5)
    assert cache.tensor("key").stride(0) * 2 == layout.entry_bytes
    assert cache.tensor("value").stride(0) * 4 == layout.entry_bytes
    assert cache.tensor("key").untyped_storage().data_ptr() == (
        cache.tensor("value").untyped_storage().data_ptr()
    )

    allocator = cache.slot_allocator()
    slot = allocator.allocate()
    cache.tensor("key")[slot].fill_(3)
    cache.tensor("value")[slot].fill_(7)
    assert torch.all(cache.tensor("key")[slot] == 3)
    assert torch.all(cache.tensor("value")[slot] == 7)


def test_id_allocator_facade_uses_logical_type_slots():
    arena = make_arena(9)
    arena.register_cache_type("kv", 64)
    arena.register_cache_type("state", 150)
    pages = ArenaSlotAllocator(arena, "kv")

    page = pages.allocate()
    assert not pages.is_free(page)
    pages.free(page)
    assert pages.is_free(page)


def test_id_allocator_facade_coalesces_batch_free():
    arena = make_arena(9)
    arena.register_cache_type("kv", 64)
    arena.register_cache_type("state", 150)
    pages = ArenaSlotAllocator(arena, "kv")
    allocated = [pages.allocate() for _ in range(6)]

    updates = []
    original = arena.allocator._update_candidates

    def record_update(start, end, delta):
        updates.append((start, end, delta))
        original(start, end, delta)

    arena.allocator._update_candidates = record_update
    pages.free_many(allocated)

    assert updates == [(0, 6, -1)]
    assert all(pages.is_free(page) for page in allocated)


def test_allocator_coalesces_batch_claim():
    arena = make_arena(9)
    arena.register_cache_type("kv", 64)
    arena.register_cache_type("state", 150)
    updates = []
    original = arena.allocator._update_candidates

    def record_update(start, end, delta):
        updates.append((start, end, delta))
        original(start, end, delta)

    arena.allocator._update_candidates = record_update
    assert arena.allocator.allocate("kv", 6) == list(range(6))
    assert updates == [(0, 6, 1)]


def test_free_rejects_invalid_cohort_before_releasing_any_live_slot():
    import pytest

    arena = make_arena(24).allocator
    arena.register_type("target", 4, 1)
    arena.register_type("draft", 5, 1)
    target = arena.allocate("target", 2)
    owners = list(arena._owners)
    for invalid in ([target[0], 2], [target[0], target[0]]):
        with pytest.raises(RuntimeError):
            arena.free("target", invalid)
        assert arena._owners == owners
        assert arena.cache_type("target").live_slots == set(target)
    arena.free("target", target)
    with pytest.raises(RuntimeError):
        arena.free("target", target)
    assert arena.num_used_physical_pages == 0


def test_exact_claim_distinguishes_own_slot_from_cross_type_overlap():
    import pytest

    arena = make_arena(24).allocator
    arena.register_type("target", 4, 1)
    arena.register_type("draft", 5, 1)
    assert arena.allocate("target", slot=0) == [0]
    owners = list(arena._owners)
    assert arena.allocate("target", slot=0, retain=True) == [0]
    with pytest.raises(RuntimeError):
        arena.allocate("target", slot=0)
    assert arena.allocate("draft", slot=0, retain=True) is None
    with pytest.raises(RuntimeError):
        arena.free("draft", [0])
    assert arena._owners == owners
    arena.free("target", [0])
    assert arena.allocate("draft", slot=0) == [0]


def test_cold_allocator_switches_to_retained_prefix_priorities():
    arena = make_arena(12).allocator
    arena.register_type("kv", 1, 1)
    arena.register_type("state", 3, 1, prefer_high=True)
    # Exercise cold allocation/release before the first prefix is retained.
    cold = arena.allocate("state", 2)
    arena.free("state", cold)
    arena.allocate("kv", slot=9)
    arena.mark_cached("kv", [9])
    arena.free("kv", [9])
    assert arena.allocate("state") == [2]
    # A late type must still observe ages even with no live physical pages.
    arena.free("state", [2])
    arena.register_type("late", 3, 1, prefer_high=True)
    assert arena.allocate("late") == [2]


def test_fragmented_slot_grids_match_physical_ownership_oracle():
    import random

    arena = make_arena(257, page_bytes=1).allocator
    for name, span in [("target", 16), ("draft", 5), ("state", 37)]:
        arena.register_type(name, span, 1, prefer_high=name == "state")
    rng = random.Random(59)
    for step in range(400):
        name = rng.choice(list(arena.cache_type_names))
        typ = arena.cache_type(name)
        live = sorted(typ.live_slots)
        if live and rng.random() < 0.5:
            slots = rng.sample(live, min(len(live), rng.randint(1, 3)))
            arena.free(name, slots)
        elif rng.random() < 0.5:
            arena.allocate(name, rng.randint(1, 3))
        else:
            arena.allocate(name, slot=rng.randrange(typ.num_slots), retain=True)
        expected = [None] * arena.num_physical_pages
        for typ in arena._types.values():
            for slot in typ.live_slots:
                start, end = arena._extent(typ, slot)
                assert all(owner is None for owner in expected[start:end])
                expected[start:end] = [(typ.name, slot)] * (end - start)
        assert arena._owners == expected
        assert arena.num_used_physical_pages == sum(x is not None for x in expected)
        for typ in arena._types.values():
            for slot in range(typ.num_slots):
                start, end = arena._extent(typ, slot)
                occupied = sum(x is not None for x in expected[start:end])
                assert typ.occupied_pages[slot] == occupied
                assert (slot in typ.free_slots) == (occupied == 0)


def test_large_coalesced_release_preserves_neighboring_live_owner():
    from gllm.runtime.cache_arena import CacheArenaAllocator

    arena = CacheArenaAllocator(120003)
    arena.register_type("wide", 40000, 1)
    arena.register_type("marker", 1, 1)
    wide = arena.allocate("wide", 3)
    assert arena.allocate("marker", slot=120000) == [120000]
    arena.free("wide", wide)
    assert arena._owners == [None] * 120000 + [("marker", 120000), None, None]
    assert arena.num_used_physical_pages == 1
    assert arena.num_free_slots("wide") == 3
    assert arena.num_free_slots("marker") == 120002


def test_cross_type_release_requeues_entries_skipped_while_owned():
    from gllm.runtime.cache_arena import CacheArenaAllocator

    arena = CacheArenaAllocator(32)
    arena.register_type("wide", 4, 1)
    arena.register_type("small", 1, 1)
    wide = arena.allocate("wide", 3)
    # The small-slot heap walks past entries covered by the wide allocation.
    assert arena.allocate("small", 3) == [12, 13, 14]
    arena.free("wide", wide)
    assert arena.allocate("small", 12) == list(range(12))


def test_empty_retention_does_not_change_clean_allocation_behavior():
    from gllm.runtime.cache_arena import CacheArenaAllocator

    arena = CacheArenaAllocator(64)
    arena.register_type("wide", 16, 1)
    arena.register_type("small", 1, 1)
    wide = arena.allocate("wide", 2)
    arena.mark_cached("wide", [])
    arena.free("wide", wide)
    assert arena.allocate("small", 32) == list(range(32))
