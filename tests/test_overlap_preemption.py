"""Cache-pressure retraction must retire GPU readers and token futures first."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.sequence import GenerationSequence
from gllm.scheduling.scheduler import OverlapScheduler
from gllm.workers.overlap import OverlapWorker, _PendingBatch, _PendingMtpBatch


@pytest.fixture
def setup_pending(monkeypatch, prefix_manager, make_scheduler):
    def setup(*, mtp=False, length=32, result=(40,)):
        mm, _, _ = prefix_manager(pages=2)
        s = make_scheduler(mm)
        s.__class__ = OverlapScheduler
        seq = GenerationSequence(1, [10] * length, [99], output_len=64)
        seq.raw_prompt_len = seq.prompt_len = length - 1
        seq.computed_token_num = length - 1
        seq.to_compute_token_num = 1
        # Keep both pages reserved, hashing only complete computed pages.
        seq.page_table = [
            mm.segment.allocate(seq, end)
            if end <= seq.computed_token_num else mm.segment.allocate()
            for end in (16, 32)
        ]
        s.batch_running.append([seq])
        w = OverlapWorker.__new__(OverlapWorker)
        w.scheduler, w.model_runner = s, s.model_runner
        w._gpu_pending, w._mtp_pending = deque(), deque()
        w._pending_frees = []
        w._prefetched_input = None
        w._prefetched_payload = None
        w._async_retire = False
        w._polls_frontend = lambda: True
        emitted, events = [], []
        w.comm = SimpleNamespace(send_output=emitted.append)
        old_free = s.model_runner.free

        def free(request):
            assert "settled" in events
            assert all(token >= 0 for token in request.token_ids)
            events.append("free")
            old_free(request)

        s.model_runner.free = free
        if mtp:
            deferred = s.process_mtp_output_deferred(decode_rows=1, width=4)
            def finalize(*args, **kwargs):
                events.append("settled")
                return [list(result)]
            s.model_runner.finalize_mtp_async = finalize
            w._mtp_pending.append(_PendingMtpBatch(
                completion=SimpleNamespace(), seqs=[seq], decode_seqs=[seq],
                deferred=deferred,
            ))
        else:
            deferred = s.process_output_deferred([7])
            s.model_runner.next_tokens_bufs = [torch.tensor(result)]
            w._gpu_pending.append(_PendingBatch(
                copy_done=SimpleNamespace(synchronize=lambda: events.append("settled")),
                batch_size=1, buf_idx=0, future_slot_ids=[7], deferred=deferred,
                input_data=SimpleNamespace(seqs=[seq]), is_dummy=False, lp_k=None,
            ))
        import gllm.workers.overlap as module
        monkeypatch.setattr(module, "is_first_pp_rank", lambda: True)
        monkeypatch.setattr(module, "is_last_pp_rank", lambda: True)
        monkeypatch.setattr(module, "get_pp_size", lambda: 1)
        s.preemption_barrier = w._preemption_barrier
        return s, seq, w, events, emitted

    return setup


@pytest.mark.parametrize("mtp", [False, True])
def test_retraction_finalizes_before_reset_and_embeds_real_history(setup_pending, make_mm_runner, mtp):
    s, seq, w, events, emitted = setup_pending(mtp=mtp)
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "free"]
    assert not w._gpu_pending and not w._mtp_pending
    assert seq.computed_token_num == 0
    assert seq.prompt_len == len(seq.token_ids) == 33
    assert seq.token_ids[-1] == 40
    assert list(s.seqs_to_prefill) == [seq]
    assert s.num_preempt_seqs == 1
    assert len(emitted) == 1 and emitted[0].act_schedule_ids == [1]
    # Re-prefill can use a prefix hit: only the final span needs embedding.
    runner, weight, _ = make_mm_runner(False, False)
    seq.computed_token_num, seq.to_compute_token_num = 30, 3
    ctx = runner._mm_prepare_cpu([seq])
    output = runner._mm_prepare_gpu(ctx)
    torch.testing.assert_close(output, weight[torch.tensor([10, 10, 40])])


@pytest.mark.parametrize("lookahead", [0, 4])
def test_mtp_compaction_recalculates_demand_and_avoids_preemption(setup_pending, lookahead):
    length = 29 - lookahead
    s, seq, _, events, emitted = setup_pending(
        mtp=True, length=length, result=(40, 41)
    )
    if lookahead:
        s.model_runner.speculative_num_draft_tokens = 3
        s.model_runner.model = SimpleNamespace(mtp=object())
        # The fixture installs its simulated MTP head after construction.
        from gllm.speculative.backend import select_speculative_backend
        s.model_runner.mtp_enabled = True
        s.speculative_backend = select_speculative_backend(
            dspark_enabled=False, mtp_enabled=True,
        )
    # Optimistic history plus lookahead needs a third page; actual history fits.
    assert len(seq) + lookahead == 33
    assert s.schedule_decode_batch(1) == [seq]
    assert events == ["settled"]
    assert seq.computed_token_num == length + 1
    assert len(seq) == length + 2 and seq.prompt_len == length - 1
    assert s.num_preempt_seqs == 0
    assert emitted[0].next_tokens == [[40, 41]]


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("finish", ["eos", "length"])
def test_completion_frees_without_requeue(setup_pending, mtp, finish):
    s, seq, _, events, emitted = setup_pending(
        mtp=mtp, result=(99 if finish == "eos" else 40,)
    )
    if finish == "length":
        seq.output_len = 2
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "free"]
    assert s.num_preempt_seqs == 0
    assert not s.seqs_to_prefill and not s.seqs_to_decode
    assert emitted[0].free_ids == [seq.seq_id]


def test_no_pressure_keeps_pending_work_overlapped(setup_pending):
    s, seq, w, events, _ = setup_pending(length=30)
    s.preemption_barrier = lambda: pytest.fail("normal decode drained GPU work")
    assert s.schedule_decode_batch(1) == [seq]
    assert not events
    assert w._gpu_pending
    assert seq.token_ids[-1] == -7  # FutureMap still owns this decode input.


def test_stalled_prefill_retires_readers_before_recovery(stalled_requests):
    s, requests = stalled_requests()
    events = []
    def barrier():
        events.append("drain")
        s.preemption_barrier = lambda: False
        return True
    s.preemption_barrier = barrier
    assert not s.schedule_once()
    assert events == ["drain"]
    assert s.num_preempt_seqs == 0
    assert all(request.page_table for request in requests)
    assert not s.schedule_once()
    assert s.num_preempt_seqs == 1


def test_barrier_materializes_mixed_prefill_relay_before_retraction(setup_pending):
    s, seq, w, events, emitted = setup_pending(mtp=True)
    # Replace the decode reservation with a completed mixed prefill handoff.
    seq.token_ids = [10] * 32
    seq.prompt_len = 32
    seq.computed_token_num = 31
    seq.to_compute_token_num = 1
    s.seqs_to_decode.clear()
    s.batch_running.append([seq])
    w._mtp_pending[0].deferred = s.process_mtp_output_deferred(0, 4)
    s.model_runner.take_mtp_relay_token = lambda sid: 40
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "free"]
    assert seq.token_ids == [10] * 32 + [40]
    assert seq.prompt_len == 33 and seq.computed_token_num == 0
    assert not seq._mtp_async_pending and not seq._mtp_relay_only_next
    assert len(emitted) == 1 and emitted[0].next_tokens == [[40]]


def test_barrier_waits_for_all_gpu_readers_before_deferred_free(setup_pending):
    s, seq, w, events, _ = setup_pending(result=(99,))
    # An already-launched successor still reads the old cache after EOS.
    w._gpu_pending.append(_PendingBatch(
        copy_done=SimpleNamespace(synchronize=lambda: events.append("successor")),
        batch_size=1, buf_idx=0, future_slot_ids=[8], deferred=None,
        input_data=SimpleNamespace(seqs=[seq]), is_dummy=False, lp_k=None,
    ))
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "successor", "free"]


def test_multiple_future_slots_are_finalized_before_reprefill(setup_pending):
    s, seq, w, events, emitted = setup_pending()
    s.seqs_to_decode.clear()
    s.batch_running.append([seq])
    deferred = s.process_output_deferred([8])
    s.model_runner.next_tokens_bufs.append(torch.tensor([41]))
    w._gpu_pending.append(_PendingBatch(
        copy_done=SimpleNamespace(synchronize=lambda: events.append("settled")),
        batch_size=1, buf_idx=1, future_slot_ids=[8], deferred=deferred,
        input_data=SimpleNamespace(seqs=[seq]), is_dummy=False, lp_k=None,
    ))
    assert seq.token_ids[-2:] == [-7, -8]
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "settled", "free"]
    assert seq.token_ids[-2:] == [40, 41]
    assert seq.prompt_len == len(seq) == 34
    assert [p.next_tokens for p in emitted] == [[40], [41]]


@pytest.mark.parametrize('method', ['chunked_prefill', 'split_pd', 'token_throttling'])
def test_completion_during_barrier_retries_stale_prefill_reserve(setup_pending, method):
    s, seq, _, _, _ = setup_pending(result=(99,))
    s.schedule_method = method
    s.schedule = s.dispatch_schedule_method()
    newcomer = GenerationSequence(2, [11]*32, [99], output_len=1)
    s.add_new_requests([newcomer])
    # Force a positive prefill reserve until the pending EOS is collected.
    s._decode_reserve_pages = lambda: 1 if s.seqs_to_decode else 0
    first = s.schedule_once()
    assert not s._pending_request_errors
    assert not seq.page_table or seq._overlap_freed
    if not first:
        assert s.schedule_once() == [newcomer]
    else:
        assert first == [newcomer]
    assert not s._pending_request_errors

def _prefetch_cancel_owner(scheduler, worker):
    """Leave the owner outside waiting queues, as a real prefetch does."""
    seqs = list(scheduler.seqs_to_decode)
    scheduler.seqs_to_decode.clear()
    if seqs:
        scheduler.batch_running.append(seqs)
        worker._prefetched_input = SimpleNamespace(seqs=seqs)
    worker._make_scheduled_input = lambda rows: SimpleNamespace(seqs=rows)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("prefetched", [False, True])
def test_abort_settles_readers_before_retiring_owner(setup_pending, mtp, prefetched):
    s, seq, w, events, emitted = setup_pending(mtp=mtp)
    if prefetched:
        _prefetch_cancel_owner(s, w)
    s.add_abort_ids([seq.seq_id, seq.seq_id])
    w.check_abort_seqs()
    assert events == ["settled", "free"]
    assert s.freed == [seq.seq_id]
    assert seq._overlap_freed
    assert not s.abort_ids and not s.seqs_to_decode and not s.batch_running
    assert not w._gpu_pending and not w._mtp_pending
    assert w._prefetched_input is None
    assert [sid for package in emitted for sid in package.free_ids] == [seq.seq_id]
    assert emitted[-1].act_schedule_ids == []
    assert s._pending_follower_frees == [seq.seq_id]
    w.check_abort_seqs()
    assert s.freed == [seq.seq_id]


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("finish", ["eos", "budget"])
def test_abort_drain_handles_natural_retirement_once(setup_pending, mtp, finish):
    s, seq, w, events, emitted = setup_pending(
        mtp=mtp, result=(99 if finish == "eos" else 40,)
    )
    if finish == "budget":
        seq.output_len = 2
    _prefetch_cancel_owner(s, w)
    s.add_abort_ids([seq.seq_id])
    w.check_abort_seqs()
    assert events == ["settled", "free"]
    assert s.freed == [seq.seq_id]
    assert not s.abort_ids and not s.batch_running
    assert w._prefetched_input is None
    assert [sid for package in emitted for sid in package.free_ids] == [seq.seq_id]


def test_abort_waits_for_successor_before_free(setup_pending):
    s, seq, w, events, emitted = setup_pending(result=(99,))
    _prefetch_cancel_owner(s, w)
    w._gpu_pending.append(_PendingBatch(
        copy_done=SimpleNamespace(synchronize=lambda: events.append("successor")),
        batch_size=1, buf_idx=0, future_slot_ids=[8], deferred=None,
        input_data=SimpleNamespace(seqs=[seq]), is_dummy=False, lp_k=None,
    ))
    s.add_abort_ids([seq.seq_id])
    w.check_abort_seqs()
    assert events == ["settled", "successor", "free"]
    assert s.freed == [seq.seq_id]
    assert [sid for package in emitted for sid in package.free_ids] == [seq.seq_id]


def test_no_abort_preserves_inflight_overlap(setup_pending):
    s, _, w, events, emitted = setup_pending(mtp=True)
    w.check_abort_seqs()
    assert w._mtp_pending and not events and not emitted
    assert not s.freed


def test_queue_only_abort_does_not_drain_peer(setup_pending):
    s, _, w, events, emitted = setup_pending(mtp=True)
    newcomer = GenerationSequence(2, [11] * 8, [99], output_len=4)
    s.add_new_requests([newcomer])
    original_free = s.model_runner.free
    s.model_runner.free = lambda seq: (
        s.memory_manager.free(seq) if seq is newcomer else original_free(seq)
    )
    s.add_abort_ids([newcomer.seq_id])
    w.check_abort_seqs()
    assert not events and w._mtp_pending
    assert not s.seqs_to_prefill and newcomer._overlap_freed
    assert not s.abort_ids and emitted[-1].free_ids == [newcomer.seq_id]


def test_prefetched_abort_preserves_peer_and_next_admission(setup_pending):
    s, victim, w, _, emitted = setup_pending(mtp=True)
    peer = GenerationSequence(2, [11] * 32, [99], output_len=4)
    peer.raw_prompt_len = peer.prompt_len = peer.computed_token_num = 31
    peer.to_compute_token_num = 1
    s.seqs_to_decode.append(peer)
    _prefetch_cancel_owner(s, w)
    arrival = GenerationSequence(3, [12] * 8, [99], output_len=1)
    s.add_new_requests([arrival])
    # Exercise a duplicate queue reference as well as the prefetched owner.
    s.seqs_to_decode.append(victim)
    s.add_abort_ids([victim.seq_id])
    w.check_abort_seqs()
    assert s.freed == [victim.seq_id]
    assert w._prefetched_input.seqs == [peer]
    assert list(s.batch_running) == [[peer]]
    assert list(s.seqs_to_prefill) == [arrival]
    assert not s.abort_ids and not getattr(peer, "_overlap_freed", False)
    # The surviving batch can retire and the next queued prompt is scheduled.
    s.add_next_tokens([99])
    package = s.process_output()
    assert package.free_ids == [peer.seq_id]
    assert s.schedule_once() == [arrival]
    assert emitted[-1].free_ids == [victim.seq_id]


def test_received_abort_is_handled_before_next_launch(setup_pending):
    s, seq, w, _, _ = setup_pending(mtp=True)
    _prefetch_cancel_owner(s, w)
    w.recv_ipc_package = lambda: s.add_abort_ids([seq.seq_id])
    def stop_before_launch():
        assert s.freed == [seq.seq_id]
        assert not s.abort_ids and not s.batch_running
        raise RuntimeError("cancelled before launch")
    w._build_prefetched_input = stop_before_launch
    with pytest.raises(RuntimeError, match="cancelled before launch"):
        w.run_pp0()

def test_abort_does_not_materialize_cancelled_mtp_relay(setup_pending):
    s, seq, w, events, emitted = setup_pending(mtp=True)
    seq.token_ids = [10] * 32
    seq.raw_prompt_len = seq.prompt_len = 32
    seq.computed_token_num = 31
    seq.to_compute_token_num = 1
    s.seqs_to_decode.clear()
    s.batch_running.append([seq])
    w._mtp_pending[0].deferred = s.process_mtp_output_deferred(0, 4)
    s.model_runner.take_mtp_relay_token = lambda sid: pytest.fail(
        "cancelled relay was published"
    )
    _prefetch_cancel_owner(s, w)
    s.add_abort_ids([seq.seq_id])
    w.check_abort_seqs()
    assert events == ["settled", "free"]
    assert seq.token_ids == [10] * 32
    assert not s.abort_ids and s.freed == [seq.seq_id]
    assert len(emitted) == 1 and emitted[0].free_ids == [seq.seq_id]
    assert not emitted[0].act_schedule_ids


def test_abort_prefetched_owner_without_inflight_work(prefix_manager, make_scheduler):
    mm, arena, _ = prefix_manager(pages=4)
    s = make_scheduler(mm)
    s.__class__ = OverlapScheduler
    seq = GenerationSequence(1, [10] * 16, [99], output_len=4)
    seq.raw_prompt_len = seq.prompt_len = seq.computed_token_num = 15
    seq.to_compute_token_num = 1
    seq.page_table = [mm.segment.allocate()]
    scheduled = [seq]
    s.batch_running.append(scheduled)
    w = OverlapWorker.__new__(OverlapWorker)
    w.scheduler, w.model_runner = s, s.model_runner
    w._gpu_pending, w._mtp_pending = deque(), deque()
    w._pending_frees = []
    w._prefetched_payload = None
    w._prefetched_input = SimpleNamespace(seqs=scheduled)
    w._polls_frontend = lambda: True
    emitted = []
    w.comm = SimpleNamespace(send_output=emitted.append)
    w._make_scheduled_input = lambda rows: SimpleNamespace(seqs=rows)
    assert w._prefetched_input.seqs is s.batch_running[-1]
    s.add_abort_ids([seq.seq_id])
    w.check_abort_seqs()
    assert not w._prefetched_input and not s.batch_running
    assert s.freed == [seq.seq_id] and not s.abort_ids
    assert emitted[0].free_ids == [seq.seq_id]
    assert arena.allocator.num_used_physical_pages == 0

def test_abort_settles_partial_prefill_before_free(setup_pending):
    s, seq, w, events, emitted = setup_pending()
    seq.token_ids = [10] * 64
    seq.raw_prompt_len = seq.prompt_len = 64
    w._gpu_pending[0].deferred = []
    s.seqs_to_decode.clear()
    s.seqs_to_prefill.append(seq)
    scheduled = [seq]
    s.batch_running.append(scheduled)
    w._prefetched_input = SimpleNamespace(seqs=scheduled)
    w._make_scheduled_input = lambda rows: SimpleNamespace(seqs=rows)
    assert w._prefetched_input.seqs is s.batch_running[-1]
    s.add_abort_ids([seq.seq_id])
    w.check_abort_seqs()
    assert events == ["settled", "free"]
    assert not s.abort_ids and not s.seqs_to_prefill and not s.batch_running
    assert w._prefetched_input is None
    assert len(emitted) == 1 and emitted[0].free_ids == [seq.seq_id]
