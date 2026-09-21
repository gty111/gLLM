"""Cache-pressure retraction must retire GPU readers and token futures first."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.sequence import GenerationSequence
from gllm.scheduling.scheduler import OverlapScheduler
from gllm.workers.overlap import OverlapWorker, _PendingBatch, _PendingMtpBatch
from test_prefix_cache_reuse import prefix_manager
from test_scheduler_liveness import scheduler
from test_text_embedding_chunks import make_runner


def setup_pending(monkeypatch, *, mtp=False, length=32, result=(40,)):
    mm, _, _ = prefix_manager(pages=2)
    s = scheduler(mm)
    s.__class__ = OverlapScheduler
    seq = GenerationSequence(1, [10] * length, [99], output_len=64)
    seq.raw_prompt_len = seq.prompt_len = length - 1
    seq.computed_token_num = length - 1
    seq.to_compute_token_num = 1
    seq.page_table = [mm.segment.allocate(seq, n) for n in (16, 32)]
    s.batch_running.append([seq])
    w = OverlapWorker.__new__(OverlapWorker)
    w.scheduler, w.model_runner = s, s.model_runner
    w._gpu_pending, w._mtp_pending = deque(), deque()
    w._pending_frees = []
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
        s.model_runner._next_tokens_bufs = [torch.tensor(result)]
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


@pytest.mark.parametrize("mtp", [False, True])
def test_retraction_finalizes_before_reset_and_embeds_real_history(monkeypatch, mtp):
    s, seq, w, events, emitted = setup_pending(monkeypatch, mtp=mtp)
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
    runner, weight, _ = make_runner(False, False)
    seq.computed_token_num, seq.to_compute_token_num = 30, 3
    ctx = runner._mm_prepare_cpu([seq])
    output = runner._mm_prepare_gpu(ctx)
    torch.testing.assert_close(output, weight[torch.tensor([10, 10, 40])])


@pytest.mark.parametrize("lookahead", [0, 4])
def test_mtp_compaction_recalculates_demand_and_avoids_preemption(monkeypatch, lookahead):
    length = 29 - lookahead
    s, seq, _, events, emitted = setup_pending(
        monkeypatch, mtp=True, length=length, result=(40, 41)
    )
    if lookahead:
        s.model_runner._mtp_k = 3
        s.model_runner.model = SimpleNamespace(mtp=object())
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
def test_completion_frees_without_requeue(monkeypatch, mtp, finish):
    s, seq, _, events, emitted = setup_pending(
        monkeypatch, mtp=mtp, result=(99 if finish == "eos" else 40,)
    )
    if finish == "length":
        seq.output_len = 2
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "free"]
    assert s.num_preempt_seqs == 0
    assert not s.seqs_to_prefill and not s.seqs_to_decode
    assert emitted[0].free_ids == [seq.seq_id]


def test_no_pressure_keeps_pending_work_overlapped(monkeypatch):
    s, seq, w, events, _ = setup_pending(monkeypatch, length=30)
    s.preemption_barrier = lambda: pytest.fail("normal decode drained GPU work")
    assert s.schedule_decode_batch(1) == [seq]
    assert not events
    assert w._gpu_pending
    assert seq.token_ids[-1] == -7  # FutureMap still owns this decode input.


def test_stalled_prefill_retires_readers_before_recovery(monkeypatch):
    from test_prefill_retraction import stalled_requests
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


def test_barrier_materializes_mixed_prefill_relay_before_retraction(monkeypatch):
    s, seq, w, events, emitted = setup_pending(monkeypatch, mtp=True)
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


def test_barrier_waits_for_all_gpu_readers_before_deferred_free(monkeypatch):
    s, seq, w, events, _ = setup_pending(monkeypatch, result=(99,))
    # An already-launched successor still reads the old cache after EOS.
    w._gpu_pending.append(_PendingBatch(
        copy_done=SimpleNamespace(synchronize=lambda: events.append("successor")),
        batch_size=1, buf_idx=0, future_slot_ids=[8], deferred=None,
        input_data=SimpleNamespace(seqs=[seq]), is_dummy=False, lp_k=None,
    ))
    assert s.schedule_decode_batch(1) == []
    assert events == ["settled", "successor", "free"]


def test_multiple_future_slots_are_finalized_before_reprefill(monkeypatch):
    s, seq, w, events, emitted = setup_pending(monkeypatch)
    s.seqs_to_decode.clear()
    s.batch_running.append([seq])
    deferred = s.process_output_deferred([8])
    s.model_runner._next_tokens_bufs.append(torch.tensor([41]))
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
def test_completion_during_barrier_retries_stale_prefill_reserve(monkeypatch, method):
    s, seq, _, _, _ = setup_pending(monkeypatch, result=(99,))
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
