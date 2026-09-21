import pytest

from gllm.runtime.memory_manager import SSMSegment
from gllm.runtime.sequence import GenerationSequence
from test_prefix_cache_reuse import populate, prefix_manager


def snapshot_manager(num_pages=16, input_tokens=64):
    manager, arena, cache = prefix_manager(num_pages)
    for name in ("state", "snapshot"):
        arena.register_cache_type(name, 4 * cache.layout.entry_bytes, prefer_high=True)
    ssm = SSMSegment.__new__(SSMSegment)
    ssm.cache_arena = arena
    ssm.arena_type = "state"
    ssm.snapshot_arena_type = "snapshot"
    ssm.dummy_working_slot = -1
    values = {}
    ssm._reset_block = lambda slot: values.pop(slot, None)
    ssm.copy_state = lambda src_kind, src, dst_kind, dst: values.__setitem__(dst, values[src])
    manager.ssm_segment = manager.segment.ssm_segment = ssm
    manager._pending_ssm_restores = {}
    manager._ssm_restore_pins = {}
    seg = manager.segment
    arena.allocator.set_reclaimer(
        "snapshot", seg.reclaim_one_ssm_snapshot, seg.num_reclaimable_ssm_snapshots
    )
    seq = GenerationSequence(1, list(range(input_tokens)), [], output_len=16)
    populate(seg, seq)
    working = arena.allocator.allocate("state")[0]
    return manager, arena.allocator, seq, working, values


def save(seg, page, value, values):
    slot = seg.reserve_ssm_snapshot(page, 16)
    assert slot is not None
    values[slot] = value
    seg.publish_ssm_snapshot(page, slot)
    return slot


def test_full_pool_replaces_oldest_snapshot_and_keeps_latest_boundary_advancing():
    manager, allocator, seq, working, values = snapshot_manager()
    seg = manager.segment
    pages = seq.page_table
    old = save(seg, pages[0], 16, values)
    recent = save(seg, pages[1], 32, values)
    assert allocator.num_used_physical_pages == 16
    for step in range(2, 100):
        page = pages[step % len(pages)]
        slot = save(seg, page, (step + 1) * 16, values)
        assert values[slot] == (step + 1) * 16
        assert len(seg._ssm_snapshot_lru) == 2
        assert allocator.num_used_physical_pages == 16
        if step == 2:
            assert slot == old
            assert seg.page2ssm_snapshot[pages[0]] is None
            assert values[recent] == 32


def test_continuation_restores_latest_boundary_after_snapshot_pool_fills(monkeypatch):
    monkeypatch.setattr("gllm.runtime.memory_manager.get_pp_size", lambda: 1)
    manager, allocator, seq, working, values = snapshot_manager(48, 512)
    seg = manager.segment
    for index, page in enumerate(seq.page_table):
        save(seg, page, (index + 1) * 16, values)
    # Only three states fit alongside KV and working state, but the last
    # boundary must still advance to the end of the prompt.
    assert len(seg._ssm_snapshot_lru) == 3
    assert seg.page2ssm_snapshot[seq.page_table[0]] is None
    seg.free_many(seq.page_table)
    continuation = GenerationSequence(2, seq.token_ids + [99], [], output_len=16)

    def allocate_working(s):
        slots = allocator.allocate("state")
        assert slots is not None
        s.recurrent_state_slot = slots[0]
        return True

    manager.allocate_recurrent_slot = allocate_working
    manager.pre_allocate_computed_page([continuation])
    assert continuation.computed_token_num == 512
    assert values[continuation.recurrent_state_slot] == 512
    assert manager.num_hit_pages == 32


def test_pending_batch_targets_are_not_evicted_or_reported_reclaimable():
    manager, allocator, seq, working, values = snapshot_manager()
    seg = manager.segment
    a, b, c, _ = seq.page_table
    first = seg.reserve_ssm_snapshot(a, 16)
    second = seg.reserve_ssm_snapshot(b, 32)
    assert first != second
    assert seg.reserve_ssm_snapshot(c, 48) is None
    assert not seg.reclaim_one_ssm_snapshot()
    assert seg.get_memory_util() == 100.0
    assert allocator.num_available_slots("kv_cache") == 0
    seg.publish_ssm_snapshot(a, first)
    assert seg.get_memory_util() == 75.0
    replacement = seg.reserve_ssm_snapshot(c, 48)
    assert replacement == first
    assert seg.page2ssm_snapshot[b] == second
    # A stale post-enqueue notification must not publish a replacement tenant.
    seg.publish_ssm_snapshot(a, first)
    assert not seg.page2ssm_snapshot_valid[c]


def test_memory_util_counts_only_non_reclaimable_physical_pages():
    manager, allocator, seq, working, values = snapshot_manager()
    seg = manager.segment
    assert seg.get_memory_util() == 50.0
    a, b, _, _ = seq.page_table
    save(seg, a, 16, values)
    save(seg, b, 32, values)
    assert allocator.num_used_physical_pages == 16
    assert seg.get_memory_util() == 50.0
    seg.pin_ssm_snapshot(a)
    seg.pin_ssm_snapshot(a)
    assert seg.get_memory_util() == 75.0
    assert allocator.num_available_slots("kv_cache") == 4
    seg.unpin_ssm_snapshot(a)
    assert seg.get_memory_util() == 75.0
    seg.unpin_ssm_snapshot(a)
    assert seg.get_memory_util() == 50.0
    seg.free_many(seq.page_table)
    assert seg.get_memory_util() == 25.0


def test_existing_snapshot_write_target_is_protected_until_republished():
    manager, allocator, seq, working, values = snapshot_manager()
    seg = manager.segment
    a, b, c, _ = seq.page_table
    first = save(seg, a, 16, values)
    second = save(seg, b, 32, values)
    assert seg.reserve_ssm_snapshot(a, 16) == first
    assert not seg.page2ssm_snapshot_valid[a]
    assert seg.reserve_ssm_snapshot(c, 48) == second
    assert seg.page2ssm_snapshot[a] == first
    assert seg.num_reclaimable_ssm_snapshots() == 0
    seg.publish_ssm_snapshot(a, first)
    assert seg.num_reclaimable_ssm_snapshots() == 1


@pytest.mark.parametrize("pp_size", [1, 2])
def test_restore_source_survives_working_allocation_reclamation(monkeypatch, pp_size):
    monkeypatch.setattr("gllm.runtime.memory_manager.get_pp_size", lambda: pp_size)
    manager, allocator, seq, working, values = snapshot_manager()
    seg = manager.segment
    source = save(seg, seq.page_table[2], 48, values)
    other = save(seg, seq.page_table[0], 16, values)
    continuation = GenerationSequence(2, seq.token_ids + [99], [], output_len=16)
    continuation.page_table = seq.page_table[:3]
    continuation.computed_token_num = 48
    manager.num_hit_pages = 3

    def allocate_working(s):
        # With the source pinned, pressure can reclaim only the other state.
        assert seg.get_memory_util() == 75.0
        s.recurrent_state_slot = allocator.allocate("state")[0]
        return True

    manager.allocate_recurrent_slot = allocate_working
    manager._restore_ssm_working_state(continuation)
    assert values[continuation.recurrent_state_slot] == 48
    assert seg.page2ssm_snapshot[seq.page_table[2]] == source
    assert seg.page2ssm_snapshot[seq.page_table[0]] is None
    assert continuation.recurrent_state_slot == other
    if pp_size == 2:
        assert manager.consume_pending_ssm_restores() == {2: source}
        assert not seg.reclaim_one_ssm_snapshot()
        manager.free_recurrent_slot = lambda s: allocator.free("state", [s.recurrent_state_slot])
        manager.free_rep_slot = lambda s: None
        manager.free(continuation)
        assert manager._ssm_restore_pins == {}
    assert seg.reclaim_one_ssm_snapshot()


@pytest.mark.parametrize("raises", [False, True])
def test_failed_restore_releases_its_source_pin(monkeypatch, raises):
    monkeypatch.setattr("gllm.runtime.memory_manager.get_pp_size", lambda: 1)
    manager, allocator, seq, working, values = snapshot_manager()
    seg = manager.segment
    save(seg, seq.page_table[0], 16, values)
    seq.computed_token_num = 16
    manager.num_hit_pages = 1

    def allocate_working(s):
        assert not seg.reclaim_one_ssm_snapshot()
        if raises:
            raise RuntimeError("allocation failed")
        return False

    manager.allocate_recurrent_slot = allocate_working
    if raises:
        with pytest.raises(RuntimeError, match="allocation failed"):
            manager._restore_ssm_working_state(seq)
    else:
        manager._restore_ssm_working_state(seq)
        assert seq.computed_token_num == manager.num_hit_pages == 0
    assert not seg._ssm_snapshot_pins
    assert seg.reclaim_one_ssm_snapshot()
