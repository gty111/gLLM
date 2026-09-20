"""The maximum context must fit together with one request's working states."""
from types import SimpleNamespace

import pytest

from gllm.runtime.cache_arena import CacheArenaAllocator
from gllm.runtime.memory_manager import MemoryManager


def manager(*, pages=40, state=None, mtp_k=0, dummy_kv=True):
    allocator = CacheArenaAllocator(pages)
    allocator.register_type("kv", 16, 16)
    mm = MemoryManager.__new__(MemoryManager)
    mm.cache_arena = SimpleNamespace(allocator=allocator)
    mm.page_size = 16
    mm.mtp_k = mtp_k
    mm.ssm_segment = mm.dsv4_state_segment = None
    if state is not None:
        # 33 bytes needs three physical pages, including alignment padding.
        allocator.register_type("state", 33, 16, prefer_high=True)
        allocator.allocate("state", slot=0)
        segment = SimpleNamespace(arena_type="state")
        if state == "ssm":
            mm.ssm_segment = segment
        else:
            mm.dsv4_state_segment = segment
    if dummy_kv:
        allocator.allocate("kv")
    mm.segment = SimpleNamespace(
        get_num_free_pages=lambda: allocator.num_available_slots("kv")
    )
    return mm, allocator


@pytest.mark.parametrize("state,mtp_k,lookahead,expected", [
    (None, 0, 0, 39 * 16),
    (None, 0, 3, 39 * 16 - 3),
    ("ssm", 0, 0, (40 - 3 - 3 - 1) * 16),
    ("ssm", 3, 3, (40 - 3 - 12 - 1) * 16 - 3),
    ("dsv4", 0, 3, (40 - 3 - 3 - 1) * 16 - 3),
])
def test_capacity_includes_aligned_states_dummy_slots_and_lookahead(
    state, mtp_k, lookahead, expected
):
    mm, allocator = manager(state=state, mtp_k=mtp_k)
    owners = list(allocator._owners)
    assert mm.validate_model_max_length(expected, mtp_lookahead=lookahead) == expected
    assert allocator._owners == owners
    with pytest.raises(RuntimeError, match="exceeds single-request cache capacity"):
        mm.validate_model_max_length(expected + 1, mtp_lookahead=lookahead)
    assert allocator._owners == owners


def test_no_graph_dummy_reservation_leaves_all_kv_pages_available():
    mm, _ = manager(pages=4, dummy_kv=False)
    assert mm.validate_model_max_length(64) == 64


def test_recurrent_only_viability_does_not_guarantee_max_context():
    mm, allocator = manager(pages=18, state="ssm", mtp_k=3)
    # Four working states fit; the naive KV-only budget also fits 128 tokens.
    assert allocator.num_available_slots("state") >= 4
    assert mm.get_num_free_pages() * 16 >= 128
    with pytest.raises(RuntimeError, match="Reduce --model-max-length to <= 29"):
        mm.validate_model_max_length(128, mtp_lookahead=3)


def test_failed_working_state_claim_leaves_no_partial_reservation():
    mm, allocator = manager(pages=9, state="ssm", mtp_k=3)
    owners = list(allocator._owners)
    with pytest.raises(RuntimeError, match="cannot hold the recurrent working states"):
        mm.validate_model_max_length(16, mtp_lookahead=3)
    assert allocator._owners == owners
