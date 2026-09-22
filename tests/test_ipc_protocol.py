"""Frontend session protocol (IPCPackage alignment contract) tests.

Covers the centralized helpers introduced with the frontend/worker
decoupling PR: drain merge alignment (merge_aligned), packet validity
(abort_stamps_valid / output_stamps_valid), and the frontend row gates
(act_session_at / free_session_at) with their fail-closed semantics.

These run without GPUs or sockets: IPCPackage is pure data.
"""

from gllm.distributed.comm import IPCPackage


def _pkg(abort_ids=(), abort_sessions=None):
    p = IPCPackage([])
    p.abort_ids = list(abort_ids)
    p.abort_sessions = abort_sessions
    return p


# ---------------------------------------------------------------------------
# merge_aligned: the ONLY sanctioned drain merge
# ---------------------------------------------------------------------------


def test_merge_keeps_abort_sessions_positionally_aligned():
    cum = _pkg([0, 1], ["e1", "e1"])
    cum.merge_aligned(_pkg([2, 3, 4], ["e2", "e2", None]))
    assert cum.abort_ids == [0, 1, 2, 3, 4]
    assert cum.abort_sessions == ["e1", "e1", "e2", "e2", None]


def test_merge_backfills_legacy_packages_into_stamped_aggregate():
    # A monolith-style (unstamped) package landing in the middle of a
    # stamped drain must NOT shift the alignment of later stamps.
    cum = _pkg([0], ["e1"])
    cum.merge_aligned(_pkg([1, 2], None))
    cum.merge_aligned(_pkg([3], ["e2"]))
    assert cum.abort_ids == [0, 1, 2, 3]
    assert cum.abort_sessions == ["e1", None, None, "e2"]


def test_merge_of_legacy_packages_stays_consistently_aligned():
    # Fully legacy drain: stamps exist but are all None (backfilled),
    # which is exactly what _route_frontend_aborts interprets as
    # "every holder" monolith semantics.
    cum = _pkg([0], None)
    cum.merge_aligned(_pkg([1], None))
    assert cum.abort_ids == [0, 1]
    assert cum.abort_sessions == [None, None]


def test_merge_extends_schedule_lists():
    cum = IPCPackage(["seq-a"])
    cum.merge_aligned(IPCPackage(["seq-b"]))
    assert cum.schedule_lists == ["seq-a", "seq-b"]


def test_merge_same_client_id_twice_preserves_both_rows():
    # THE regression that motivated per-session ids: two packages,
    # same client id 0, different sessions. Collapsing by id (dict)
    # would lose one of the two aborts.
    cum = _pkg([0], ["e1"])
    cum.merge_aligned(_pkg([0], ["e2"]))
    assert cum.abort_ids == [0, 0]
    assert cum.abort_sessions == ["e1", "e2"]


# ---------------------------------------------------------------------------
# Validity predicates (fail-closed on malformed packets)
# ---------------------------------------------------------------------------


def test_abort_stamps_valid_accepts_legacy_and_aligned():
    assert _pkg([], None).abort_stamps_valid()
    assert _pkg([1], None).abort_stamps_valid()
    assert _pkg([1, 2], ["e", "e"]).abort_stamps_valid()


def test_abort_stamps_valid_rejects_misaligned():
    assert not _pkg([1, 2], ["e"]).abort_stamps_valid()
    assert not _pkg([1], ["e", "e"]).abort_stamps_valid()


def test_output_stamps_valid_checks_both_pairs():
    p = IPCPackage([])
    p.act_schedule_ids = [7]
    p.sessions = ["e1"]
    p.free_ids = [8, 9]
    p.free_sessions = ["e1", None]
    assert p.output_stamps_valid()

    p.sessions = ["e1", "extra"]
    assert not p.output_stamps_valid()

    p.sessions = ["e1"]
    p.free_sessions = ["only-one"]
    assert not p.output_stamps_valid()


def test_output_stamps_valid_legacy_all_none():
    p = IPCPackage([])
    p.act_schedule_ids = [1, 2]
    p.free_ids = [3]
    p.sessions = None
    p.free_sessions = None
    assert p.output_stamps_valid()


# ---------------------------------------------------------------------------
# Frontend row gates: positional + fail-closed
# ---------------------------------------------------------------------------


def _out_pkg(act=(), act_sessions=None, free=(), free_sessions=None):
    p = IPCPackage([])
    p.act_schedule_ids = list(act)
    p.sessions = act_sessions
    p.free_ids = list(free)
    p.free_sessions = free_sessions
    return p


def test_act_gate_legacy_means_every_row_ours():
    p = _out_pkg(act=[0, 1, 2], act_sessions=None)
    for i in range(3):
        assert p.act_session_at(i, "whatever-epoch")


def test_act_gate_filters_other_session_positionally():
    # Same client id 0 in BOTH rows (two sessions); only row 1 is ours.
    p = _out_pkg(act=[0, 0], act_sessions=["e1", "e2"])
    assert not p.act_session_at(0, "e2")
    assert p.act_session_at(1, "e2")


def test_act_gate_fail_closed_on_out_of_range_index():
    # In-range rows of a validly-shaped list are judged positionally;
    # an index past the end of the stamp list is unconditionally
    # foreign (no row may be trusted beyond its stamp).
    p = _out_pkg(act=[0], act_sessions=["e2"])
    assert p.act_session_at(0, "e2")
    assert not p.act_session_at(1, "e2")
    assert not p.act_session_at(5, "e2")


def test_free_gate_mirrors_act_gate():
    p = _out_pkg(free=[0, 0], free_sessions=["e1", "e2"])
    assert p.free_session_at(1, "e2")
    assert not p.free_session_at(0, "e2")

    legacy = _out_pkg(free=[5], free_sessions=None)
    assert legacy.free_session_at(0, "anything")


def test_free_gate_none_entry_is_not_our_session():
    # Free-only rows of a dead session may carry a None stamp; a None
    # stamp never equals a live frontend epoch, so the row is foreign.
    p = _out_pkg(free=[0, 1], free_sessions=["e2", None])
    assert p.free_session_at(0, "e2")
    assert not p.free_session_at(1, "e2")


def test_act_gate_packet_invalid_fails_wholesale():
    # The frontend checks output_stamps_valid() ONCE per package and
    # threads the answer in: one O(1) check fails every row closed.
    p = _out_pkg(act=[0, 1], act_sessions=["e2"])  # short list
    assert not p.output_stamps_valid()
    assert not p.act_session_at(0, "e2", p.output_stamps_valid())
    assert not p.act_session_at(1, "e2", p.output_stamps_valid())


def test_free_gate_packet_invalid_fails_wholesale():
    p = _out_pkg(free=[0], free_sessions=["e2", "extra"])
    assert not p.output_stamps_valid()
    assert not p.free_session_at(0, "e2", p.output_stamps_valid())
