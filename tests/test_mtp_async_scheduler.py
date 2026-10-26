from collections import deque
from types import SimpleNamespace

import pytest

from gllm.runtime.forward_metadata import ForwardMetadataPlan
from gllm.workers.overlap import OverlapWorker
from gllm.scheduling.scheduler import OverlapScheduler
from gllm.runtime.sequence import GenerationSequence
from gllm.speculative.mtp import MtpMixin


class _Runner:
    def __init__(self):
        self.freed = []
        self.hash_positions = []
        self.mtp_relay = {}

    def register_decode_page_hash(self, seq, position):
        self.hash_positions.append((seq.seq_id, position))

    def free(self, seq):
        self.freed.append(seq.seq_id)

    def take_mtp_relay_token(self, seq_id):
        return self.mtp_relay.pop(seq_id)


def _scheduler():
    scheduler = OverlapScheduler.__new__(OverlapScheduler)
    scheduler.batch_running = deque()
    scheduler.seqs_to_decode = deque()
    scheduler._pending_follower_frees = []
    scheduler.model_runner = _Runner()
    return scheduler


def _decode_seq(seq_id=1, output_len=32):
    seq = GenerationSequence(
        seq_id,
        [10, 11, 12],
        [99],
        output_len=output_len,
        temperature=0,
        top_k=1,
    )
    seq.raw_prompt_len = seq.prompt_len = 2
    seq.computed_token_num = 2
    seq.to_compute_token_num = 1
    return seq


def test_mtp_deferred_compacts_rejected_tail():
    scheduler = _scheduler()
    seq = _decode_seq()
    scheduler.batch_running.append([seq])

    deferred = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    assert seq.token_ids == [10, 11, 12, -1, -1, -1, -1]
    assert seq.computed_token_num == 6
    assert seq._mtp_async_pending

    package = scheduler.process_mtp_output_finalize(deferred, [[20, 21]])
    assert seq.token_ids == [10, 11, 12, 20, 21]
    assert seq.computed_token_num == 4
    assert not seq._mtp_async_pending
    assert package.next_tokens == [[20, 21]]
    assert scheduler.model_runner.hash_positions == [(1, 3), (1, 4)]


def test_mtp_deferred_truncates_at_max_tokens_and_frees():
    scheduler = _scheduler()
    seq = _decode_seq(seq_id=7, output_len=2)
    scheduler.batch_running.append([seq])

    deferred = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    package = scheduler.process_mtp_output_finalize(
        deferred, [[20, 21, 22, 23]]
    )

    assert seq.token_ids == [10, 11, 12, 20]
    assert package.next_tokens == [[20]]
    assert package.free_ids == [7]
    assert scheduler.model_runner.freed == [7]


def test_mtp_deferred_keeps_batch_row_mapping_after_stale_row_finishes():
    scheduler = _scheduler()
    finished = _decode_seq(seq_id=3, output_len=1)
    live = _decode_seq(seq_id=4)
    scheduler.batch_running.append([finished, live])

    deferred = scheduler.process_mtp_output_deferred(decode_rows=2, width=4)
    protected = []
    package = scheduler.process_mtp_output_finalize(
        deferred,
        [[30, 31], [40, 41]],
        defer_frees=protected,
    )

    # Row 0 retires logically but is protected from physical release because a
    # successor may already be touching it. Row 1 must still consume completion
    # row 1 rather than being shifted onto row 0.
    assert package.next_tokens == [[30], [40, 41]]
    assert protected == [finished]
    assert scheduler.model_runner.freed == []
    assert live.token_ids[-2:] == [40, 41]


def test_mtp_finalize_then_reserve_next_keeps_placeholder_positions_stable():
    scheduler = _scheduler()
    seq = _decode_seq()

    scheduler.batch_running.append([seq])
    first = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    scheduler.process_mtp_output_finalize(first, [[20, 21]])

    # gLLM's async cadence reserves the successor only after compacting the
    # predecessor, so its absolute placeholder position remains valid.
    seq.to_compute_token_num = 1
    scheduler.batch_running.append([seq])
    second = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    scheduler.process_mtp_output_finalize(second, [[22]])
    assert seq.token_ids == [10, 11, 12, 20, 21, 22]


def test_mtp_mixed_deferred_finishes_prefill_with_relay_placeholder():
    scheduler = _scheduler()
    decode = _decode_seq(seq_id=11)
    prefill = GenerationSequence(
        12,
        [50, 51, 52],
        [],
        output_len=8,
        temperature=0,
        top_k=1,
    )
    prefill.computed_token_num = 0
    prefill.to_compute_token_num = 3
    scheduler.batch_running.append([decode, prefill])

    deferred = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    assert decode.token_ids[-4:] == [-1, -1, -1, -1]
    assert prefill.token_ids == [50, 51, 52, -1]
    assert decode.computed_token_num == 6
    assert prefill.computed_token_num == 3

    package = scheduler.process_mtp_output_finalize(
        deferred, [[60, 61], [70]]
    )
    assert decode.token_ids[-2:] == [60, 61]
    # The prefill sample is the successor's GPU relay x1 and is intentionally
    # not committed twice at this boundary.
    assert prefill.token_ids == [50, 51, 52]
    assert package.act_schedule_ids == [11]
    assert package.next_tokens == [[60, 61]]
    assert [seq.seq_id for seq in scheduler.seqs_to_decode] == [11, 12]

    # First successor verify consumes the GPU-only relay x1. Its reservation
    # skips the ordinary base-compute increment exactly once, preserving the
    # decode invariant computed_token_num == len(token_ids) - 1.
    scheduler.seqs_to_decode.clear()
    prefill.to_compute_token_num = 1
    scheduler.batch_running.append([prefill])
    successor = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    assert len(prefill.token_ids) == 7
    assert prefill.computed_token_num == 6
    successor_package = scheduler.process_mtp_output_finalize(
        successor, [[70]]
    )
    assert prefill.token_ids == [50, 51, 52, 70]
    assert prefill.computed_token_num == 3
    assert successor_package.next_tokens == [[70]]


def test_mtp_mixed_deferred_does_not_emit_partial_prefill_sample():
    scheduler = _scheduler()
    decode = _decode_seq(seq_id=21)
    prefill = GenerationSequence(
        22,
        [80, 81, 82, 83],
        [],
        output_len=8,
        temperature=0,
        top_k=1,
    )
    prefill.computed_token_num = 0
    prefill.to_compute_token_num = 2
    scheduler.batch_running.append([decode, prefill])

    deferred = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    assert all(item.batch_idx != 1 for item in deferred)
    package = scheduler.process_mtp_output_finalize(
        deferred, [[90], [91]]
    )
    assert prefill.token_ids == [80, 81, 82, 83]
    assert prefill.computed_token_num == 2
    assert package.act_schedule_ids == [21]


def test_mtp_relay_only_materializes_before_plain_decode():
    scheduler = _scheduler()
    prefill = GenerationSequence(
        23,
        [50, 51, 52],
        [],
        output_len=8,
        temperature=0,
        top_k=1,
    )
    prefill.computed_token_num = 0
    prefill.to_compute_token_num = 3
    scheduler.batch_running.append([prefill])

    deferred = scheduler.process_mtp_output_deferred(decode_rows=0, width=4)
    scheduler.process_mtp_output_finalize(deferred, [[70]])
    assert prefill._mtp_relay_only_next
    assert prefill.token_ids == [50, 51, 52]
    assert prefill.computed_token_num == 3

    scheduler.model_runner.mtp_relay[23] = 70
    package = scheduler.materialize_mtp_relay_only()

    assert not prefill._mtp_relay_only_next
    assert prefill.token_ids == [50, 51, 52, 70]
    # x1 is committed but still uncached, ready for an ordinary decode input.
    assert prefill.computed_token_num == 3
    assert package.act_schedule_ids == [23]
    assert package.next_tokens == [[70]]
    assert scheduler.model_runner.mtp_relay == {}
    assert scheduler.model_runner.hash_positions == [(23, 3)]


def test_mtp_verify_rows_form_a_contiguous_mixed_batch_prefix():
    seqs = [
        SimpleNamespace(
            _mtp_verify=True, to_compute_token_num=4, computed_prompt=True
        ),
        SimpleNamespace(
            _mtp_verify=True, to_compute_token_num=4, computed_prompt=True
        ),
        SimpleNamespace(
            _mtp_verify=False, to_compute_token_num=11, computed_prompt=False
        ),
    ]
    plan = ForwardMetadataPlan.from_sequences(seqs)
    assert plan.num_mtp_verify_rows == 2
    assert plan.num_decodes == 0

    pure = ForwardMetadataPlan.from_sequences(seqs[:2])
    assert pure.num_mtp_verify_rows == 2
    assert pure.num_mtp_verify_rows == pure.batch_size


def test_mtp_verify_rows_reject_noncontiguous_mixed_layout():
    seqs = [
        SimpleNamespace(
            _mtp_verify=True, to_compute_token_num=4, computed_prompt=True
        ),
        SimpleNamespace(
            _mtp_verify=False, to_compute_token_num=11, computed_prompt=False
        ),
        SimpleNamespace(
            _mtp_verify=True, to_compute_token_num=4, computed_prompt=True
        ),
    ]
    try:
        ForwardMetadataPlan.from_sequences(seqs)
    except ValueError as exc:
        assert "contiguous batch prefix" in str(exc)
    else:
        raise AssertionError("noncontiguous MTP rows were accepted")


@pytest.mark.parametrize("top_k", [1, 20, -1])
def test_overlap_worker_builds_one_mtp_plan_for_pure_and_mixed_batches(top_k):
    worker = OverlapWorker.__new__(OverlapWorker)
    worker._dp = False
    decisions = []
    worker.model_runner = SimpleNamespace(
        mtp_enabled=True,
        mtp_begin_iter=lambda n, **kwargs: decisions.append(n) or bool(n),
    )

    pure_seqs = [_decode_seq(31), _decode_seq(32)]
    for seq in pure_seqs:
        seq.top_k = top_k
    worker._prefetched_input = SimpleNamespace(
        seqs=pure_seqs, num_decodes=2, num_prefills=0
    )
    pure = worker._plan_mtp_batch()
    assert pure.speculate and pure.use_async
    assert pure.decode_ids == (31, 32)

    prefill = GenerationSequence(33, [1, 2], [], output_len=8, temperature=0, top_k=1)
    prefill.computed_token_num = 0
    prefill.to_compute_token_num = 2
    worker._prefetched_input = SimpleNamespace(
        seqs=[pure_seqs[0], prefill], num_decodes=1, num_prefills=1
    )
    mixed = worker._plan_mtp_batch()
    assert mixed.speculate and mixed.use_async and mixed.decode_ids == (31,)
    assert decisions == [2, 1]


def test_filter_prefetched_freed_releases_empty_inflight_batch_slot():
    worker = OverlapWorker.__new__(OverlapWorker)
    freed = SimpleNamespace(seq_id=41, _overlap_freed=True)
    scheduled = [freed]
    worker.scheduler = SimpleNamespace(batch_running=deque([scheduled]))

    kept = worker._filter_prefetched_freed(scheduled)

    assert kept == []
    assert list(worker.scheduler.batch_running) == []


def test_filter_prefetched_freed_keeps_surviving_inflight_rows():
    worker = OverlapWorker.__new__(OverlapWorker)
    freed = SimpleNamespace(seq_id=42, _overlap_freed=True)
    live = SimpleNamespace(seq_id=43, _overlap_freed=False)
    scheduled = [freed, live]
    worker.scheduler = SimpleNamespace(batch_running=deque([scheduled]))

    kept = worker._filter_prefetched_freed(scheduled)

    assert kept == [live]
    assert list(worker.scheduler.batch_running) == [[live]]


def test_plain_decode_transition_publishes_mtp_relay_before_input_refresh():
    worker = OverlapWorker.__new__(OverlapWorker)
    worker._dp = False
    worker._mtp_pending = deque([object()])
    worker._prefetched_input = SimpleNamespace(seqs=[object()])
    calls = []

    worker.check_abort_seqs = lambda: None
    worker.recv_ipc_package = lambda: None
    worker._plan_mtp_batch = lambda: SimpleNamespace(speculate=False)
    worker._drain_mtp_pending = lambda: (
        calls.append("drain"), worker._mtp_pending.clear()
    )
    worker._publish_mtp_relay_only = lambda: calls.append("publish") or True

    def refresh():
        calls.append("refresh")
        worker._prefetched_input = None

    worker._refresh_prefetched_input = refresh
    worker._build_prefetched_input = lambda: calls.append("build")

    worker.run_pp0()

    assert calls == ["drain", "publish", "refresh", "build"]


def _boundary_runner(limit=32):
    runner = MtpMixin()
    runner.mtp_enabled = True
    runner._mtp_k = 3
    runner._mtp_max_batch = 0
    runner._mtp_spec_decision = None
    runner.max_num_batched_tokens = 8192
    runner.model_max_length = limit
    runner._mtp_relay = {}
    return runner


@pytest.mark.parametrize("limit,length,output_len,expected", [
    (32, 28, 64, True),
    (32, 29, 64, False),
    (33, 29, 64, True),
    (33, 30, 64, False),
    (240000, 239996, 240000, True),
    (240000, 239997, 240000, False),
    (64, 28, 30, True),  # Two prompt tokens plus thirty output tokens.
    (64, 29, 30, False),
])
def test_mtp_window_respects_context_and_output_budget(limit, length, output_len, expected):
    runner = _boundary_runner(limit)
    seq = _decode_seq(output_len=output_len)
    seq.token_ids = [10] * length
    assert runner.mtp_begin_iter(1, seqs=[seq]) is expected
    # Input prep and execution must agree with the iteration decision.
    assert runner.mtp_speculate_batch(1) is expected
    assert runner.mtp_prep_eligible([seq]) is expected


@pytest.mark.parametrize("mixed", [False, True])
def test_overlap_plan_declines_entire_cohort_when_one_window_does_not_fit(mixed):
    worker = OverlapWorker.__new__(OverlapWorker)
    worker._dp = False
    worker.model_runner = _boundary_runner()
    safe, tail = _decode_seq(51, 64), _decode_seq(52, 64)
    safe.token_ids = [10] * 20
    tail.token_ids = [10] * 29
    seqs = [safe, tail]
    if mixed:
        seqs.append(GenerationSequence(53, [10] * 8, [], output_len=8))
    worker._prefetched_input = SimpleNamespace(
        seqs=seqs, num_decodes=2, num_prefills=int(mixed),
    )
    assert not worker._plan_mtp_batch().speculate
    assert not worker.model_runner.mtp_prep_eligible([safe, tail])


@pytest.mark.parametrize("accepted", [1, 4])
def test_context_boundary_drains_optimistic_mtp_before_replanning(accepted):
    scheduler = _scheduler()
    seq = _decode_seq(61, output_len=64)
    seq.token_ids = [10] * 25
    seq.computed_token_num = 24
    scheduler.batch_running.append([seq])
    deferred = scheduler.process_mtp_output_deferred(decode_rows=1, width=4)
    assert len(seq) == 29  # The pending result may accept fewer than four.
    worker = OverlapWorker.__new__(OverlapWorker)
    worker._dp = False
    worker.model_runner = _boundary_runner()
    worker._mtp_pending = deque([deferred])
    worker._prefetched_input = SimpleNamespace(
        seqs=[seq], num_decodes=1, num_prefills=0,
    )
    calls = []
    worker.check_abort_seqs = lambda: None
    worker.recv_ipc_package = lambda: None
    worker._polls_frontend = lambda: False
    worker._publish_mtp_relay_only = lambda: False
    worker._refresh_prefetched_input = lambda: calls.append("refresh")

    def drain():
        calls.append("drain")
        scheduler.process_mtp_output_finalize(
            worker._mtp_pending.popleft(), [[20] * accepted],
        )

    class PlainDecodeReached(Exception):
        pass

    def launch_plain(*args, **kwargs):
        assert calls == ["drain", "refresh"]
        assert not seq._mtp_async_pending and -1 not in seq.token_ids
        assert len(seq) == 25 + accepted
        raise PlainDecodeReached

    worker._drain_mtp_pending = drain
    worker._run_mtp = lambda plan: calls.append("mtp")
    worker._launch_batch = launch_plain
    if accepted == 1:
        worker.run_pp0()
        assert calls == ["drain", "refresh", "mtp"]
        assert worker.model_runner.mtp_context_compatible([seq])
    else:
        with pytest.raises(PlainDecodeReached):
            worker.run_pp0()


def test_plain_overlap_launch_invalidates_only_its_rows_relay():
    worker = OverlapWorker.__new__(OverlapWorker)
    worker.model_runner = _boundary_runner()
    worker.model_runner._mtp_relay = {71: object(), 72: object()}

    class PreparationReached(Exception):
        pass

    def prepare(data):
        assert set(worker.model_runner._mtp_relay) == {72}
        raise PreparationReached

    worker.model_runner.prepare_input_cpu = prepare
    with pytest.raises(PreparationReached):
        worker._launch_batch(SimpleNamespace(seqs=[_decode_seq(71)]))


@pytest.mark.parametrize("pp_size", [1, 2])
def test_last_plain_overlap_token_is_finalized_without_scheduling_successor(pp_size):
    scheduler = _scheduler()
    scheduler.pp_size = pp_size
    seq = _decode_seq(81, output_len=2)
    scheduler.batch_running.append([seq])
    deferred = scheduler.process_output_deferred([5])
    assert len(seq) - seq.raw_prompt_len == seq.output_len
    assert list(scheduler.seqs_to_decode) == []
    package = scheduler.process_output_finalize(deferred, [20])
    assert package.next_tokens == [20]
    assert package.free_ids == [81]
    assert seq.token_ids == [10, 11, 12, 20]
