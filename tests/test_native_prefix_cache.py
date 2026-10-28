"""Native metadata invariants, including failure atomicity and stale writers."""
import random

import pytest

from gllm._cache_arena import ArenaCore, PrefixCore, PrefixHashes
from gllm.runtime.memory_manager import _ensure_page_hash
from gllm.runtime.sequence import GenerationSequence


@pytest.mark.parametrize("page_size", [1, 16, 128])
def test_native_chain_matches_python_tuple_hash(page_size):
    rng = random.Random(42)
    tokens = [rng.randrange(-(2**62), 2**62) for _ in range(page_size * 7 + 1)]
    hashes = PrefixHashes(tokens)
    previous = 0x9E3779B97F4A7C15
    for i in range(7):
        previous = hash((previous, tuple(tokens[i * page_size:(i + 1) * page_size])))
        assert hashes.page_hash(tokens, page_size, i) == previous
    assert len(hashes) == 7
    assert hashes.page_hash(tokens, page_size, 0) == hash((0x9E3779B97F4A7C15, tuple(tokens[:page_size])))


def test_hash_chain_resets_on_source_or_page_size_change_and_extends_on_append():
    tokens = list(range(16))
    hashes = PrefixHashes(tokens)
    first = hashes.page_hash(tokens, 16, 0)
    tokens.extend(range(16, 32))
    assert hashes.page_hash(tokens, 16, 1) == hash((first, tuple(tokens[16:])))
    replacement = [99] * 32
    assert hashes.page_hash(replacement, 16, 0) == hash((0x9E3779B97F4A7C15, tuple(replacement[:16])))
    assert len(hashes) == 1
    assert hashes.page_hash(replacement, 8, 0) == hash((0x9E3779B97F4A7C15, tuple(replacement[:8])))
    assert len(hashes) == 1


def test_multimodal_source_replacement_invalidates_chain():
    seq = GenerationSequence(1, list(range(32)), [], output_len=1)
    text_hash = _ensure_page_hash(seq, 16, 1)
    seq.hash_token_ids = [1234] * 32
    image_hash = _ensure_page_hash(seq, 16, 1)
    assert image_hash != text_hash
    seq.hash_token_ids = [5678] * 32
    assert _ensure_page_hash(seq, 16, 1) != image_hash
    seq.hash_token_ids = None
    assert _ensure_page_hash(seq, 16, 1) == text_hash


def test_reference_release_is_atomic_on_duplicate_underflow():
    core = PrefixCore(3)
    core.retain([0, 1])
    with pytest.raises(RuntimeError, match="underflow"):
        core.release([0, 1, 1])
    assert core.refs() == [1, 1, 0]
    with pytest.raises(IndexError):
        core.retain([0, 3])
    assert core.refs() == [1, 1, 0]


def test_duplicate_hash_owner_survives_eviction_of_older_page():
    core = PrefixCore(2)
    core.register(0, 42, [1], True)
    core.register(1, 42, [1], True)
    core.evict(0)
    assert core.lookup(42, [1]) == 1
    core.retain([1])
    assert core.release([1]) == ([1], [1], [])


def test_zero_hash_is_a_valid_cache_key():
    core = PrefixCore(1)
    core.register(0, 0, [1], True)
    core.retain([0])
    assert core.release([0]) == ([0], [0], [])
    assert core.lookup(0, [1]) == 0
    core.evict(0)
    assert core.lookup(0, [1]) is None


def test_snapshot_pending_stale_publication_and_nested_pins():
    core = PrefixCore(3)
    core.set_snapshot(0, 4)
    assert core.reclaimable() == 0
    assert core.valid_snapshot(0) is None
    core.publish(0, 5)
    assert core.valid_snapshot(0) is None
    core.publish(0, 4)
    core.set_snapshot(1, 5)
    core.publish(1, 5)
    assert core.oldest_snapshot() == 0
    assert core.valid_snapshot(0) == 4
    assert core.oldest_snapshot() == 1
    core.pin(1)
    core.pin(1)
    assert core.oldest_snapshot() == 0
    with pytest.raises(RuntimeError, match="pinned"):
        core.set_snapshot(1, None)
    with pytest.raises(RuntimeError, match="pinned"):
        core.can_release_snapshot(1)
    core.unpin(1)
    assert core.pins() == {1: 1}
    core.unpin(1)
    assert core.lru() == [0, 1]
    assert core.prepare_snapshot(0) == 4
    assert core.lru() == [1]
    assert core.valid_snapshot(0) is None
    core.set_snapshot(0, 7)
    core.publish(0, 4)
    assert core.valid_snapshot(0) is None
    with pytest.raises(RuntimeError, match="underflow"):
        core.unpin(0)


def test_failed_arena_retain_does_not_increment_prefix_references():
    arena = ArenaCore(4)
    kv = arena.register(1, False)
    other = arena.register(2, False)
    core = PrefixCore(4)
    tokens = list(range(32))
    hashes = PrefixHashes(tokens)
    for page in range(2):
        core.register(page, hashes.page_hash(tokens, 16, page), tokens[:8], True)
    assert arena.allocate(other, 1) == [0]
    with pytest.raises(RuntimeError, match="occupied"):
        core.retain_scanned(arena, kv, tokens, hashes, 16, 2, tokens[:8])
    assert core.refs() == [0] * 4
    assert arena.counts(kv)[1] == 0


def test_failed_arena_release_preserves_reference_counts():
    arena = ArenaCore(2)
    kv = arena.register(1, False)
    core = PrefixCore(2)
    assert arena.allocate(kv, 1) == [0]
    core.retain([0, 1])
    with pytest.raises(RuntimeError, match="not owned"):
        core.release_arena(arena, kv, [0, 1])
    assert core.refs() == [1, 1]
    assert arena.counts(kv)[1] == 1


@pytest.mark.parametrize("operation", ["deepcopy", "pickle"])
def test_sequence_copy_preserves_native_hashes_without_sharing_mutable_state(operation):
    import copy
    import pickle

    seq = GenerationSequence(1, list(range(32)), [], output_len=1)
    expected = _ensure_page_hash(seq, 16, 1)
    cloned = copy.deepcopy(seq) if operation == "deepcopy" else pickle.loads(pickle.dumps(seq))
    assert cloned.token_ids is not seq.token_ids
    assert cloned._page_hashes is not seq._page_hashes
    assert cloned._page_hashes.matches_source(cloned.token_ids)
    assert len(cloned._page_hashes) == 2
    assert _ensure_page_hash(cloned, 16, 1) == expected
    cloned.token_ids.extend(range(32, 48))
    _ensure_page_hash(cloned, 16, 2)
    assert len(cloned._page_hashes) == 3
    assert len(seq._page_hashes) == 2


def test_snapshot_mirroring_is_validated_before_mutation():
    core = PrefixCore(3)
    core.mirror_snapshots([0, 1], [5, -1])
    assert core[0] == 5 and core[1] is None
    assert core.validity() == [False] * 3
    with pytest.raises(RuntimeError, match="length mismatch"):
        core.mirror_snapshots([0, 1], [8])
    with pytest.raises(IndexError):
        core.mirror_snapshots([0, 3], [8, 9])
    assert core[0] == 5


def test_intrusive_snapshot_lru_matches_ordered_reference():
    from collections import OrderedDict

    rng = random.Random(71)
    core = PrefixCore(32)
    lru = OrderedDict()
    pins = [0] * 32
    valid = [False] * 32
    for p in range(32):
        core.set_snapshot(p, p)
    for _ in range(3000):
        p = rng.randrange(32)
        operation = rng.randrange(4)
        if operation == 0:
            core.publish(p, p)
            valid[p] = True
            if pins[p] == 0:
                lru.pop(p, None)
                lru[p] = None
        elif operation == 1:
            core.pin(p)
            pins[p] += 1
            lru.pop(p, None)
        elif operation == 2 and pins[p]:
            core.unpin(p)
            pins[p] -= 1
            if pins[p] == 0 and valid[p]:
                lru.pop(p, None)
                lru[p] = None
        elif operation == 3:
            assert core.valid_snapshot(p) == (p if valid[p] else None)
            if valid[p] and pins[p] == 0:
                lru.pop(p, None)
                lru[p] = None
        assert core.lru() == list(lru)
        assert core.reclaimable() == len(lru)
        assert core.oldest_snapshot() == next(iter(lru), None)


def test_pp_payload_reads_native_snapshot_assignments_without_full_table_copy():
    from gllm.scheduling.distributed import DriverPayloadBuilder

    seq = GenerationSequence(1, list(range(48)), [], output_len=1)
    seq.page_table = [2, 7, 10]
    core = PrefixCore(128)
    core.mirror_snapshots([2, 7, 10], [4, -1, 9])
    builder = DriverPayloadBuilder()
    first = builder.build([seq], [], ssm_page2snap=core)
    assert first.updates[0].new_page_snap_slots == [4, -1, 9]
    unchanged = builder.build([seq], [], ssm_page2snap=core)
    assert unchanged.updates[0].new_page_snap_slots is None
    seq.page_table.append(15)
    core.mirror_snapshots([15], [11])
    appended = builder.build([seq], [], ssm_page2snap=core)
    assert appended.updates[0].new_page_ids == [15]
    assert appended.updates[0].new_page_snap_slots == [11]


@pytest.mark.parametrize("prompt_len", [15, 16, 19, 31, 32])
@pytest.mark.parametrize("width", [1, 4])
def test_multimodal_decode_hash_extends_only_finalized_boundary(prompt_len, width):
    seq = GenerationSequence(1, [7] * prompt_len, [], output_len=64)
    seq.hash_token_ids = [900 + i for i in range(prompt_len)]
    source = seq.hash_token_ids
    expected = source[:]
    for start in range(0, 40, width):
        row = list(range(100 + start, 100 + start + width))
        seq.token_ids.extend(row)
        expected.extend(row)
        # An overlapped successor may already have placeholders in token_ids.
        seq.token_ids.extend([-1] * 4)
        for page_idx in range(len(expected) // 16):
            actual = _ensure_page_hash(seq, 16, page_idx)
            assert actual == PrefixHashes(expected).page_hash(expected, 16, page_idx)
        del seq.token_ids[-4:]
        assert seq.hash_token_ids is source
        assert source[:prompt_len] == expected[:prompt_len]
        assert all(token >= 0 for token in source)
    assert len(source) == (len(expected) // 16) * 16


def test_multimodal_lookup_after_decode_extends_hash_source():
    from gllm.runtime.memory_manager import _hash_source

    seq = GenerationSequence(1, [7] * 19, [], output_len=64)
    seq.hash_token_ids = [900] * 19
    seq.token_ids.extend(range(40))
    assert _hash_source(seq) == [900] * 19 + list(range(40))


def test_multimodal_hash_rejects_placeholder_without_mutating_source():
    seq = GenerationSequence(1, [7] * 15, [], output_len=64)
    seq.hash_token_ids = [900] * 15
    seq.token_ids.append(-1)
    with pytest.raises(ValueError, match="unfinalized"):
        _ensure_page_hash(seq, 16, 0)
    assert seq.hash_token_ids == [900] * 15
