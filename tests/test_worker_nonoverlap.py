"""CPU-only tests for the non-overlap :class:`Worker` driver path.

The overlap path has FutureMap-level coverage in ``test_overlap_pp.py`` /
``test_overlap_preemption.py``; the plain worker loop (still the fallback
for PP+DP combos) had none. Everything here uses ``Worker.__new__`` plus
``SimpleNamespace`` duck-typed runners/comms, mirroring the overlap tests.
GPU-bound pieces (``forward_pp``'s ``recv_pp_data``/``step_once``, real
``InputData.cal_input``) are stubbed at the module seam and called out per
test.
"""
from types import SimpleNamespace

import pytest

import gllm.workers.worker as worker_mod
from gllm.distributed.comm import IPCPackage
from gllm.scheduling.distributed import SchedulePayload
from gllm.workers.worker import Worker, run_worker


def make_worker(**attrs):
    worker = Worker.__new__(Worker)
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


class FakeInputData:
    """Stand-in for the GPU-touched ``InputData`` (records instead of computes)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cal_input_seqs = None
        self.mrope = None

    def cal_input(self, seqs):
        self.cal_input_seqs = list(seqs)

    def set_mrope_position(self, positions):
        self.mrope = positions


# ------------------------------------------------------------------
# Process entry point
# ------------------------------------------------------------------


def test_overlap_entry_point_is_the_shared_one():
    from gllm.workers.overlap import run_overlap_worker

    assert run_overlap_worker is run_worker


@pytest.mark.parametrize("pp_rank,expected", [(0, "pp0"), (2, "other")])
def test_run_worker_dispatches_on_pp_rank(pp_rank, expected):
    events = []

    class FakeWorker:
        def __init__(self):
            self.pp_rank = pp_rank

        def init(self):
            events.append("init")

        def run_pp0(self):
            events.append("pp0")
            raise KeyboardInterrupt

        def run_other(self):
            events.append("other")
            raise KeyboardInterrupt

        def handle_keyboardInterrupt(self):
            events.append("kb")

        def handle_exception(self, e):
            events.append(("exc", type(e).__name__))

    run_worker(FakeWorker())
    assert events == ["init", expected, "kb"]


def test_run_worker_routes_exceptions_to_handler():
    events = []

    class FakeWorker:
        pp_rank = 0

        def init(self):
            events.append("init")

        def run_pp0(self):
            raise RuntimeError("boom")

        def handle_keyboardInterrupt(self):
            events.append("kb")

        def handle_exception(self, e):
            events.append(("exc", type(e).__name__))

    run_worker(FakeWorker())
    assert events == ["init", ("exc", "RuntimeError")]


# ------------------------------------------------------------------
# Per-iter loop skeletons
# ------------------------------------------------------------------


def test_run_pp0_ordering():
    worker = make_worker()
    order = []
    for name in (
        "check_abort_seqs",
        "recv_ipc_package",
        "recv_next_tokens",
        "schedule_forward",
        "process_output",
    ):
        setattr(worker, name, lambda n=name: order.append(n))
    worker.run_pp0()
    assert order == [
        "check_abort_seqs",
        "recv_ipc_package",
        "recv_next_tokens",
        "schedule_forward",
        "process_output",
    ]


def test_run_other_ordering():
    worker = make_worker()
    order = []
    worker.recv_schedule_payload = lambda: order.append("recv")
    worker.forward_pp = lambda: order.append("forward")
    worker.run_other()
    assert order == ["recv", "forward"]


# ------------------------------------------------------------------
# Frontend polling role
# ------------------------------------------------------------------


def test_polls_frontend_non_dp_is_rank0_only(monkeypatch):
    monkeypatch.setattr(worker_mod, "is_dp_attn", lambda: False)
    assert make_worker(rank=0)._polls_frontend()
    assert not make_worker(rank=3)._polls_frontend()


def test_polls_frontend_dp_is_tp0_only(monkeypatch):
    monkeypatch.setattr(worker_mod, "is_dp_attn", lambda: True)
    monkeypatch.setattr(worker_mod, "get_tp_rank", lambda: 0)
    assert make_worker(rank=4)._polls_frontend()
    monkeypatch.setattr(worker_mod, "get_tp_rank", lambda: 1)
    assert not make_worker(rank=4)._polls_frontend()


# ------------------------------------------------------------------
# DP forward barrier: dummy/skip logic
# ------------------------------------------------------------------


def _barrier_worker(bucket=8, dp_size=2):
    return make_worker(
        dp_size=dp_size,
        model_runner=SimpleNamespace(
            dp_select_bucket=lambda n: bucket if n <= bucket else None
        ),
    )


def test_dp_barrier_skips_forward_when_every_group_idle(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([0, 0], [True, True]))
    assert _barrier_worker()._dp_forward_barrier(0, True) is None


def test_dp_barrier_publishes_common_bucket_for_pure_decode(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([2, 3], [True, True]))
    counts, padded = _barrier_worker()._dp_forward_barrier(2, True)
    assert padded == 8
    assert counts == [8, 8]


def test_dp_barrier_pads_idle_group_to_one_token_in_eager_mode(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([2, 0], [False, True]))
    counts, padded = _barrier_worker()._dp_forward_barrier(2, False)
    assert padded is None
    assert counts == [2, 1]


def test_dp_barrier_falls_back_to_eager_on_bucket_miss(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([9, 1], [True, True]))
    counts, padded = _barrier_worker(bucket=8)._dp_forward_barrier(9, True)
    assert padded is None
    assert counts == [9, 1]


# ------------------------------------------------------------------
# _dp_prepare_and_barrier (shared PP=1 / PP>1 DP prologue)
# ------------------------------------------------------------------


class _Shape:
    def __init__(self, n):
        self.shape = (n,)


def _dp_runner(decode=True):
    calls = []
    runner = SimpleNamespace(
        input_data=None,
        dp_select_bucket=lambda n: 8,
        create_dummy_seqs=lambda n, runtime=True: ["dummy"] * n,
        check_decode_batch=lambda: decode,
    )

    def prepare_input(seqs):
        calls.append(seqs)
        runner.input_data = SimpleNamespace(tokens_cpu=_Shape(len(seqs)))

    runner.prepare_input = prepare_input
    return runner, calls


def test_dp_prepare_returns_none_when_world_idle(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([0, 0], [True, True]))
    runner, calls = _dp_runner()
    worker = make_worker(dp_size=2, model_runner=runner)
    assert worker._dp_prepare_and_barrier([]) is None
    assert calls == []


def test_dp_prepare_pads_idle_local_group_with_dummy(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([0, 4], [True, True]))
    runner, calls = _dp_runner()
    worker = make_worker(dp_size=2, model_runner=runner)
    real_ntok, counts, padded = worker._dp_prepare_and_barrier([])
    assert real_ntok == 0
    assert (counts, padded) == ([8, 8], 8)
    assert calls == [["dummy"]]


def test_dp_prepare_measures_real_batch(monkeypatch):
    monkeypatch.setattr(worker_mod, "dp_all_gather_meta", lambda n, d: ([n, 0], [d, True]))
    runner, calls = _dp_runner(decode=False)
    worker = make_worker(dp_size=2, model_runner=runner)
    real_ntok, counts, padded = worker._dp_prepare_and_barrier(["a", "b"])
    assert real_ntok == 2
    assert padded is None  # this group flagged prefill -> eager for everyone
    assert counts == [2, 1]
    assert calls == [["a", "b"]]


# ------------------------------------------------------------------
# Dummy input builder (shared with the overlap driver)
# ------------------------------------------------------------------


def test_build_dummy_input_uses_runtime_dummy_seqs(monkeypatch):
    monkeypatch.setattr(worker_mod, "InputData", FakeInputData)
    mm = SimpleNamespace()
    runner = SimpleNamespace(
        memory_manager=mm,
        model_max_length=16,
        create_dummy_seqs=lambda n, runtime=True: ["d"] * n,
    )
    worker = make_worker(model_runner=runner)
    dummy = worker._build_dummy_input(3)
    assert dummy.cal_input_seqs == ["d", "d", "d"]
    assert dummy.kwargs == {
        "use_buffer": False,
        "memory_manager": mm,
        "max_seq_length": 16,
    }


# ------------------------------------------------------------------
# Schedule payload build
# ------------------------------------------------------------------


def test_build_schedule_payload_short_circuits_single_rank(monkeypatch):
    monkeypatch.setattr(worker_mod, "get_world_size", lambda: 1)
    worker = make_worker()
    assert worker._build_schedule_payload(["seq"]) is None


def test_build_schedule_payload_without_ssm_passes_frees_through(monkeypatch):
    monkeypatch.setattr(worker_mod, "get_world_size", lambda: 4)
    monkeypatch.setattr(worker_mod, "get_pp_size", lambda: 1)
    built = {}
    worker = make_worker(
        model_runner=SimpleNamespace(memory_manager=SimpleNamespace(), use_mm=False),
        scheduler=SimpleNamespace(consume_pending_follower_frees=lambda: ["f1"]),
        payload_builder=SimpleNamespace(
            build=lambda **kw: built.setdefault("kw", kw) or kw
        ),
    )
    result = worker._build_schedule_payload(["s1", "s2"])
    assert result is built["kw"]
    assert built["kw"]["scheduled_seqs"] == ["s1", "s2"]
    assert built["kw"]["frees"] == ["f1"]
    assert built["kw"]["mrope_positions"] is None
    assert built["kw"]["use_mm"] is False
    assert built["kw"]["ssm_page2snap"] is None
    assert built["kw"]["ssm_restores"] is None


# ------------------------------------------------------------------
# PP-follower payload receive (InputData stubbed: cal_input needs GPU state)
# ------------------------------------------------------------------


def _follower_worker(monkeypatch, payload, applied_seqs=()):
    monkeypatch.setattr(worker_mod, "InputData", FakeInputData)
    sent_control = []
    worker = make_worker(
        comm=SimpleNamespace(recv_schedule_payload=lambda: payload),
        follower_store=SimpleNamespace(
            get=lambda sid: None,
            apply_payload=lambda p: list(applied_seqs),
        ),
        model_runner=SimpleNamespace(
            memory_manager=SimpleNamespace(),
            model_max_length=8,
            create_dummy_seqs=lambda n, runtime=True: ["d"] * n,
            free_follower_state=lambda sid: None,
        ),
        schedule_queue=worker_mod.deque(),
    )
    worker._apply_control_cmd = lambda code, data: sent_control.append((code, data))
    return worker, sent_control


def test_recv_schedule_payload_noop_when_socket_empty(monkeypatch):
    worker, _ = _follower_worker(monkeypatch, None)
    worker.recv_schedule_payload()
    assert not worker.schedule_queue


def test_recv_schedule_payload_routes_control_cmd(monkeypatch):
    worker, sent_control = _follower_worker(
        monkeypatch, SchedulePayload(control_cmd=1, control_data="/tmp/x")
    )
    worker.recv_schedule_payload()
    assert sent_control == [(1, "/tmp/x")]
    assert not worker.schedule_queue


def test_recv_schedule_payload_builds_dummy_for_idle_dp_group(monkeypatch):
    payload = SchedulePayload(dp_dummy_size=3, dp_counts=[1, 3], dp_padded_size=4)
    worker, _ = _follower_worker(monkeypatch, payload)
    worker.recv_schedule_payload()
    assert len(worker.schedule_queue) == 1
    input_data = worker.schedule_queue[0]
    assert input_data.cal_input_seqs == ["d", "d", "d"]
    assert input_data.dp_dummy is True
    assert input_data.dp_counts == [1, 3]
    assert input_data.dp_padded_size == 4


def test_recv_schedule_payload_builds_real_input_with_mrope(monkeypatch):
    payload = SchedulePayload(mrope_positions="mrope-sentinel")
    worker, _ = _follower_worker(monkeypatch, payload, applied_seqs=["s1", "s2"])
    worker.recv_schedule_payload()
    assert len(worker.schedule_queue) == 1
    input_data = worker.schedule_queue[0]
    assert input_data.cal_input_seqs == ["s1", "s2"]
    assert input_data.mrope == "mrope-sentinel"
    assert input_data.dp_dummy is False


# ------------------------------------------------------------------
# Token feedback (PP>1)
# ------------------------------------------------------------------


def test_recv_next_tokens_noop_for_pp1(monkeypatch):
    monkeypatch.setattr(worker_mod, "get_pp_size", lambda: 1)
    worker = make_worker(
        scheduler=SimpleNamespace(
            add_next_tokens=lambda *a: pytest.fail("PP=1 must not consume tokens")
        )
    )
    worker.recv_next_tokens()


def test_recv_next_tokens_poller_unpacks_triple(monkeypatch):
    monkeypatch.setattr(worker_mod, "get_pp_size", lambda: 2)
    monkeypatch.setattr(worker_mod, "get_tp_size", lambda: 2)
    received = []
    worker = make_worker(
        comm=SimpleNamespace(
            recv_tokens=lambda: ([5, 6], "lp", "plp"),
            broadcast_tokens_to_tp=lambda t: t,
        ),
        scheduler=SimpleNamespace(
            add_next_tokens=lambda *a: received.append(a)
        ),
    )
    worker._polls_frontend = lambda: True
    worker.recv_next_tokens()
    assert received == [([5, 6], "lp", "plp")]


def test_recv_next_tokens_non_poller_takes_broadcast(monkeypatch):
    monkeypatch.setattr(worker_mod, "get_pp_size", lambda: 2)
    monkeypatch.setattr(worker_mod, "get_tp_size", lambda: 2)
    received = []
    worker = make_worker(
        comm=SimpleNamespace(broadcast_tokens_to_tp=lambda t: [7]),
        scheduler=SimpleNamespace(
            add_next_tokens=lambda *a: received.append(a)
        ),
    )
    worker._polls_frontend = lambda: False
    worker.recv_next_tokens()
    assert received == [([7], None, None)]


# ------------------------------------------------------------------
# Abort / output replies reach the frontend only from the poller
# ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["check_abort_seqs", "process_output"])
@pytest.mark.parametrize("polls", [True, False])
def test_driver_replies_only_when_polling_frontend(method, polls):
    sent = []
    package = IPCPackage([])
    worker = make_worker(
        scheduler=SimpleNamespace(**{method: lambda: package}),
        comm=SimpleNamespace(send_output=sent.append),
    )
    worker._polls_frontend = lambda: polls
    getattr(worker, method)()
    assert sent == ([package] if polls else [])


# ------------------------------------------------------------------
# IPC input fan-out application
# ------------------------------------------------------------------


def _ipc_worker(monkeypatch, tp_size=1, pp_size=1):
    monkeypatch.setattr(worker_mod, "get_tp_size", lambda: tp_size)
    monkeypatch.setattr(worker_mod, "get_pp_size", lambda: pp_size)
    admitted, aborted, applied, logs = [], [], [], []
    worker = make_worker(
        rank=0,
        _disagg_coord=None,
        _disagg_recv=None,
        _is_disagg_lm=False,
        scheduler=SimpleNamespace(
            add_new_requests=admitted.append,
            add_abort_ids=aborted.append,
            set_log=logs.append,
        ),
    )
    worker._apply_control_cmd = lambda code, data: applied.append((code, data))
    return worker, admitted, aborted, applied, logs


def test_recv_ipc_package_non_poller_applies_broadcast(monkeypatch):
    worker, admitted, aborted, applied, _ = _ipc_worker(monkeypatch, tp_size=2)
    worker.rank = 1  # not the coordinator rank; skips the disagg poll
    package = IPCPackage([])
    package.log = None
    package.schedule_lists = ["seq"]
    package.abort_ids = [11]
    worker.comm = SimpleNamespace(broadcast_input_to_tp=lambda cum: package)
    worker._polls_frontend = lambda: False
    worker.recv_ipc_package()
    assert admitted == [["seq"]]
    assert aborted == [[11]]
    assert applied == []


def test_recv_ipc_package_poller_aggregates_and_translates(monkeypatch):
    worker, admitted, aborted, applied, logs = _ipc_worker(monkeypatch)
    p1 = IPCPackage([])
    p1.log = None
    p1.schedule_lists = ["seq"]
    p1.control_cmd = "stop_profile"
    p2 = IPCPackage([])
    p2.log = "lvl"
    p2.abort_ids = [5]
    incoming = iter([p1, p2, None])
    worker.comm = SimpleNamespace(recv_ipc_package=lambda: next(incoming))
    worker._polls_frontend = lambda: True
    worker.recv_ipc_package()
    assert admitted == [["seq"]]
    assert aborted == [[5]]
    assert applied == [(2, None)]  # stop_profile -> code 2
    assert logs == ["lvl"]  # the one log override survives aggregation


def test_recv_ipc_package_drops_empty_poll(monkeypatch):
    worker, admitted, aborted, applied, _ = _ipc_worker(monkeypatch)
    worker.comm = SimpleNamespace(recv_ipc_package=lambda: None)
    worker._polls_frontend = lambda: True
    worker.recv_ipc_package()
    assert admitted == [] and aborted == [] and applied == []


# ------------------------------------------------------------------
# Control command translation
# ------------------------------------------------------------------


def test_translate_control_cmd_known_and_unknown():
    worker = make_worker(profile_output_dir="/tmp/prof_test")
    code, data = worker._translate_control_cmd("start_profile")
    assert code == 1
    assert data.startswith("/tmp/prof_test/trace_session_")
    assert worker._translate_control_cmd("stop_profile") == (2, None)
    assert worker._translate_control_cmd("bogus") == (0, None)


# ------------------------------------------------------------------
# Disagg admission routing / event application
# ------------------------------------------------------------------


def test_admit_requests_monolith_goes_straight_to_scheduler():
    added = []
    worker = make_worker(
        _is_disagg_lm=False,
        scheduler=SimpleNamespace(add_new_requests=added.append),
    )
    worker._admit_requests(["a", "b"])
    assert added == [["a", "b"]]


def test_admit_requests_disagg_routes_mm_seqs_to_coordinator():
    submitted, added = [], []
    mm_seq = SimpleNamespace(mm_items=["img"])
    text_seq = SimpleNamespace(mm_items=None)
    worker = make_worker(
        _is_disagg_lm=True,
        _disagg_coord=SimpleNamespace(submit=submitted.append),
        scheduler=SimpleNamespace(add_new_requests=added.append),
    )
    worker._admit_requests([mm_seq, text_seq])
    assert submitted == [mm_seq]
    assert added == [[text_seq]]


def test_admit_requests_disagg_non_tp0_drops_mm_seqs():
    added = []
    worker = make_worker(
        _is_disagg_lm=True,
        _disagg_coord=None,
        scheduler=SimpleNamespace(add_new_requests=added.append),
    )
    worker._admit_requests([SimpleNamespace(mm_items=["img"])])
    assert added == []


def test_apply_disagg_events_none_is_noop():
    worker = make_worker()
    worker._apply_disagg_events(None)  # must not touch any attribute


def test_apply_disagg_events_admits_and_embeddings():
    calls, added = [], []
    seq = SimpleNamespace(seq_id=3)
    events = SimpleNamespace(
        admits=[(seq, "state")],
        allocs=[(4, 0, 2)],
        emb_ready=[(3, 0, 0, 12, True)],
        frees=[8],
        aborts=[9],
    )
    worker = make_worker(
        model_runner=SimpleNamespace(
            disagg_register=lambda *a: calls.append(("register",) + a),
            disagg_mark_ready=lambda *a: calls.append(("ready",) + a),
            disagg_alloc_pages=lambda reqs: calls.append(("alloc", reqs)) or [[5, 6]],
            disagg_free_pages=lambda ids: calls.append(("free", ids)),
        ),
        scheduler=SimpleNamespace(
            add_new_requests=added.append,
            add_abort_ids=lambda ids: calls.append(("abort", ids)),
        ),
        _disagg_coord=SimpleNamespace(
            on_pages=lambda reqs, pages: calls.append(("on_pages", reqs, pages))
        ),
    )
    worker._apply_disagg_events(events)
    assert calls == [
        ("free", [8]),
        ("alloc", [(4, 0, 2)]),
        ("on_pages", [(4, 0, 2)], [[5, 6]]),
        ("register", 3, "state"),
        ("ready", 3, 0, 12, True),
        ("abort", [9]),
    ]
    assert added == [[seq]]


# ------------------------------------------------------------------
# Hybrid/SSM mirrors on PP followers
# ------------------------------------------------------------------


def test_mirror_ssm_snapshot_slots_noop_without_segment():
    worker = make_worker(model_runner=SimpleNamespace(memory_manager=SimpleNamespace()))
    payload = SchedulePayload()
    worker._mirror_ssm_snapshot_slots(payload)  # no segment -> no-op


def test_mirror_ssm_snapshot_slots_applies_driver_assignments():
    page2snap = {10: 1}
    seg = SimpleNamespace(page2ssm_snapshot=page2snap)
    worker = make_worker(
        model_runner=SimpleNamespace(memory_manager=SimpleNamespace(segment=seg))
    )
    payload = SchedulePayload(
        updates=[
            SimpleNamespace(new_page_snap_slots=[5, -1], new_page_ids=[10, 11]),
            SimpleNamespace(new_page_snap_slots=[], new_page_ids=[]),
        ]
    )
    worker._mirror_ssm_snapshot_slots(payload)
    assert page2snap == {10: 5, 11: None}


def test_apply_ssm_restores_noop_without_ssm_segment():
    worker = make_worker(model_runner=SimpleNamespace(memory_manager=SimpleNamespace()))
    worker._apply_ssm_restores([SimpleNamespace()])  # no ssm_segment -> no-op


def test_apply_ssm_restores_replays_copy_state():
    copied = []
    ssm_segment = SimpleNamespace(copy_state=lambda *a: copied.append(a))
    worker = make_worker(
        model_runner=SimpleNamespace(
            memory_manager=SimpleNamespace(ssm_segment=ssm_segment)
        )
    )
    seqs = [
        SimpleNamespace(ssm_restore_src_slot=3, recurrent_state_slot=9),
        SimpleNamespace(ssm_restore_src_slot=None, recurrent_state_slot=1),
    ]
    worker._apply_ssm_restores(seqs)
    assert copied == [("snapshot", 3, "working", 9)]


# ------------------------------------------------------------------
# schedule_forward fast path
# ------------------------------------------------------------------


def test_schedule_forward_empty_schedule_returns_early(monkeypatch):
    monkeypatch.setattr(worker_mod, "is_dp_attn", lambda: False)
    worker = make_worker(
        scheduler=SimpleNamespace(schedule_once=lambda: []),
        # Any attribute access on the runner proves the early return broke.
        model_runner=SimpleNamespace(),
    )
    assert worker.schedule_forward() is None
