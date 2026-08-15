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
