"""Compare native ownership and prefix ages with the Python reference."""
import random

import pytest

pytest.importorskip("gllm._cache_arena")
from gllm.runtime.cache_arena import PythonCacheArenaAllocator
from gllm.runtime.native_cache_arena import NativeCacheArenaAllocator


@pytest.mark.parametrize("seed", range(20))
def test_native_matches_reference_with_prefixes_and_late_registration(seed):
    rng = random.Random(seed)
    reference = PythonCacheArenaAllocator(257)
    native = NativeCacheArenaAllocator(257)
    events = [[], []]
    for allocator, recorded in zip((reference, native), events):
        for name, span in (("kv", 16), ("draft", 5), ("state", 37), ("small", 1)):
            allocator.register_type(name, span, 1, prefer_high=name == "state")
            allocator.set_evictor(name, lambda s, n=name, out=recorded: out.append((n, s)))
    for step in range(1000):
        if step in (80, 280):
            for allocator, recorded in zip((reference, native), events):
                name = f"late{step}"
                allocator.register_type(name, 7, 1)
                allocator.set_evictor(name, lambda s, n=name, out=recorded: out.append((n, s)))
        name = rng.choice(reference.cache_type_names)
        group = reference.cache_type(name)
        live = sorted(group.live_slots)
        if live and rng.random() < 0.5:
            slots = rng.sample(live, min(len(live), rng.randint(1, 8)))
            if step > 150 and rng.random() < 0.3:
                reference.mark_cached(name, slots)
                native.mark_cached(name, slots)
            reference.free(name, slots)
            native.free(name, slots)
        elif rng.random() < 0.5:
            count = rng.randint(1, 8)
            assert reference.allocate(name, count) == native.allocate(name, count)
        elif rng.random() < 0.5:
            slots = [rng.randrange(group.num_slots) for _ in range(3)]
            assert reference.retain_many(name, slots) == native.retain_many(name, slots)
        else:
            slot = rng.randrange(group.num_slots)
            assert reference.allocate(name, slot=slot, retain=True) == native.allocate(name, slot=slot, retain=True)
        assert events[0] == events[1]
        assert reference._owners == native._owners
        assert reference._cached_pages == native._cached_pages
        assert reference.num_used_physical_pages == native.num_used_physical_pages
        for name in reference.cache_type_names:
            a, b = reference.cache_type(name), native.cache_type(name)
            for attr in ("live_slots", "free_slots", "retained_slots", "occupied_pages", "cached_priority"):
                assert getattr(a, attr) == getattr(b, attr), (step, name, attr)


def test_native_bad_batch_does_not_change_ownership_or_prefix_ages():
    allocator = NativeCacheArenaAllocator(32)
    allocator.register_type("kv", 2, 1)
    slots = allocator.allocate("kv", 3)
    before = allocator._owners
    for bad in ([slots[0], slots[0]], [slots[0], 15]):
        with pytest.raises(RuntimeError):
            allocator.free("kv", bad)
        assert allocator._owners == before
    with pytest.raises(IndexError):
        allocator.mark_cached("kv", [slots[0], 32])
    assert allocator._cache_clock == 0
    assert allocator._cached_pages == [0] * 32


@pytest.mark.parametrize("high", [False, True])
def test_native_bitset_boundaries_and_prefix_transition(high):
    reference = PythonCacheArenaAllocator(131)
    native = NativeCacheArenaAllocator(131)
    for allocator in (reference, native):
        allocator.register_type("kv", 1, 1, prefer_high=high)
    expected = reference.allocate("kv", 131)
    assert native.allocate("kv", 131) == expected
    assert native.allocate("kv") is None
    released = [0, 1, 63, 64, 65, 127, 128, 130]
    for allocator in (reference, native):
        allocator.free("kv", released)
    assert native.allocate("kv", len(released)) == reference.allocate("kv", len(released))
    for allocator in (reference, native):
        allocator.mark_cached("kv", [64, 130])
        allocator.free("kv", released)
    assert native.allocate("kv", len(released)) == reference.allocate("kv", len(released))
    assert native._owners == reference._owners
