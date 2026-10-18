"""Check Rust metadata against a small, page-by-page test oracle."""
import random

import pytest

from gllm.runtime.cache_arena import CacheArenaAllocator


@pytest.mark.parametrize("seed", range(20))
def test_native_ownership_prefix_ages_and_callbacks_match_physical_pages(seed):
    rng = random.Random(seed)
    native = CacheArenaAllocator(257)
    owners, ages = [None] * 257, [0] * 257
    clock = 0
    groups, retained = {}, {}
    actual_events, expected_events = [], []

    def register(name, span, high=False):
        native.register_type(name, span, 1, prefer_high=high)
        native.set_evictor(name, lambda slot, n=name: actual_events.append((n, slot)))
        groups[name] = (span, high)
        retained[name] = set()

    def extent(name, slot):
        span, _ = groups[name]
        return slot * span, (slot + 1) * span

    def claim(name, slots, preserve=False):
        protected = {(name, slot) for slot in slots} if preserve else set()
        for slot in slots:
            a, b = extent(name, slot)
            assert all(owner is None for owner in owners[a:b])
            owners[a:b] = [(name, slot)] * (b - a)
        seen = set()
        for slot in slots:
            a, b = extent(name, slot)
            for other, (span, _) in groups.items():
                for stale in range(a // span, min((b - 1) // span + 1, 257 // span)):
                    key = (other, stale)
                    if key in protected or key in seen:
                        continue
                    seen.add(key)
                    if stale in retained[other]:
                        retained[other].remove(stale)
                        x, y = extent(other, stale)
                        ages[x:y] = [0] * (y - x)
                    expected_events.append(key)

    for name, span in (("kv", 16), ("draft", 5), ("state", 37), ("small", 1)):
        register(name, span, high=name == "state")
    for step in range(1000):
        if step in (80, 280):
            register(f"late{step}", 7)
        name = rng.choice(tuple(groups))
        span, high = groups[name]
        live = [s for s in range(257 // span) if owners[s * span] == (name, s)]
        if live and rng.random() < 0.5:
            slots = rng.sample(live, min(len(live), rng.randint(1, 8)))
            if step > 150 and rng.random() < 0.3:
                native.mark_cached(name, slots)
                for slot in slots:
                    clock += 1
                    a, b = extent(name, slot)
                    ages[a:b] = [clock] * (b - a)
                    retained[name].add(slot)
            native.free(name, slots)
            for slot in slots:
                a, b = extent(name, slot)
                owners[a:b] = [None] * (b - a)
        elif rng.random() < 0.5:
            count = rng.randint(1, 8)
            free = [s for s in range(257 // span)
                    if all(owner is None for owner in owners[s * span:(s + 1) * span])]
            free.sort(key=lambda s: (max(ages[s * span:(s + 1) * span]), -s if high else s))
            expected = free[:count] if len(free) >= count else None
            assert native.allocate(name, count) == expected
            if expected:
                claim(name, expected)
        else:
            slots = ([rng.randrange(257 // span) for _ in range(3)]
                     if rng.random() < 0.5 else [rng.randrange(257 // span)])
            unique = list(dict.fromkeys(slots))
            fresh = [s for s in unique if s not in live]
            blocked = any(any(owner is not None for owner in owners[s * span:(s + 1) * span])
                          for s in fresh)
            expected = None if blocked else unique
            if len(slots) == 1:
                assert native.allocate(name, slot=slots[0], retain=True) == expected
            else:
                assert native.retain_many(name, slots) == expected
            if expected is not None:
                claim(name, fresh, preserve=True)
        assert actual_events == expected_events
        assert native._owners == owners
        assert native._cached_pages == ages
        assert native._cache_clock == clock
        assert native.num_used_physical_pages == sum(owner is not None for owner in owners)
        for name, (span, _) in groups.items():
            group = native.cache_type(name)
            occupied = [sum(owner is not None for owner in owners[s * span:(s + 1) * span])
                        for s in range(257 // span)]
            free = {s for s, count in enumerate(occupied) if count == 0}
            assert group.occupied_pages == occupied
            assert group.free_slots == free
            assert native.num_free_slots(name) == len(free)
            assert group.live_slots == {s for s in range(257 // span)
                                        if owners[s * span] == (name, s)}
            assert group.retained_slots == retained[name]
            priorities = group.cached_priority
            for slot in free:
                a, b = extent(name, slot)
                assert priorities[slot] == max(ages[a:b])


def test_native_bad_batch_does_not_change_ownership_or_prefix_ages():
    allocator = CacheArenaAllocator(32)
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
    allocator = CacheArenaAllocator(131)
    allocator.register_type("kv", 1, 1, prefer_high=high)
    assert allocator.allocate("kv", 131) == sorted(range(131), reverse=high)
    assert allocator.allocate("kv") is None
    released = [0, 1, 63, 64, 65, 127, 128, 130]
    allocator.free("kv", released)
    assert allocator.allocate("kv", len(released)) == sorted(released, reverse=high)
    allocator.mark_cached("kv", [64, 130])
    allocator.free("kv", released)
    # Uncached slots first, then retained slots from oldest to newest.
    expected = sorted(set(released) - {64, 130}, reverse=high) + [64, 130]
    assert allocator.allocate("kv", len(released)) == expected
    assert allocator.num_used_physical_pages == 131
