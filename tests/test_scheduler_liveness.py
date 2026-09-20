"""Cache pressure must not leave all requests parked without a runnable batch."""
from collections import deque
from types import SimpleNamespace

import pytest

from gllm.runtime.sequence import GenerationSequence
from gllm.scheduling.scheduler import Scheduler
from test_prefix_cache_reuse import populate, prefix_manager


def scheduler(manager, method='chunked_prefill'):
    manager._rep_pool = None
    manager._pending_ssm_restores = {}
    freed = []
    def free(seq):
        freed.append(seq.seq_id)
        manager.free(seq)
    runner = SimpleNamespace(
        memory_manager=manager, maxd=8, maxp=128, max_num_batched_tokens=128,
        minp=1, iterp=1, page_size=16, init_new_token_ratio=0.1,
        min_new_token_ratio=0.1, mtp_enabled=False, free=free,
        _mm_precompute_hash=lambda seq: None, disagg_prefill_limit=lambda seq: None,
    )
    result = Scheduler(1, runner, method)
    result.log = False
    result.freed = freed
    return result


def seq(id, n):
    return GenerationSequence(id, list(range(n)), [999], output_len=16)


@pytest.mark.parametrize('method', ['chunked_prefill', 'split_pd', 'token_throttling'])
def test_full_hit_runs_with_all_pages_pinned(method):
    mm, arena, cache = prefix_manager(pages=4)
    completed = seq(0, 64)
    populate(mm.segment, completed)
    mm.segment.free_many(completed.page_table)
    s = scheduler(mm, method)
    request = seq(1, 64)
    s.add_new_requests([request])
    batch = s.schedule_once()
    assert batch == [request]
    assert request.to_compute_token_num == 1
    assert mm.get_num_free_pages() == 0
    assert not s._pending_request_errors


def test_pinned_partial_prefill_runs_without_free_pages():
    mm, _, _ = prefix_manager(pages=4)
    request = seq(1, 64)
    populate(mm.segment, request)
    request.computed_token_num = 48
    s = scheduler(mm)
    s.add_new_requests([request])
    assert s.schedule_once() == [request]
    assert request.to_compute_token_num == 16


def test_blocked_prefix_releases_pins_and_allows_smaller_waiter():
    mm, _, _ = prefix_manager(pages=4)
    completed = seq(0, 64)
    populate(mm.segment, completed)
    mm.segment.free_many(completed.page_table)
    big, small = seq(1, 65), GenerationSequence(2, [777], [], output_len=1)
    s = scheduler(mm)
    s.add_new_requests([big, small])
    assert s.schedule_once() == [small]
    assert list(s.seqs_to_prefill) == [big]
    assert big.page_table == [] and big.computed_token_num == 0
    assert big.to_compute_token_num == 0
    # The failed hit must not inflate hit counters on each admission attempt.
    assert mm.num_hit_pages == 0
    assert not s._pending_request_errors


def test_failed_admission_releases_new_recurrent_state():
    mm, _, _ = prefix_manager(pages=1)
    completed = seq(0, 16)
    populate(mm.segment, completed)
    mm.segment.free_many(completed.page_table)
    mm.allocate_recurrent_slot = lambda request: setattr(request, 'recurrent_state_slot', 7) or True
    released = []
    def release(request):
        released.append(request.recurrent_state_slot)
        request.recurrent_state_slot = None
    mm.free_recurrent_slot = release
    s = scheduler(mm)
    request = seq(1, 17)
    s.add_new_requests([request])
    batch, _ = s.schedule_prefill_batch(128)
    assert not batch
    assert released == [7]
    assert request.recurrent_state_slot is None
    assert request.page_table == []
    assert mm.get_num_free_pages() == 1


def test_prefill_chunk_shrinks_to_available_capacity():
    mm, _, _ = prefix_manager(pages=2)
    s = scheduler(mm)
    request = seq(1, 64)
    s.add_new_requests([request])
    batch, tokens = s.schedule_prefill_batch(128, reserve_pages=1)
    assert batch == [request]
    assert tokens == 16
    assert mm.get_num_free_pages() == 1


@pytest.mark.parametrize('length', [8, 16])
def test_decode_does_not_preempt_when_existing_page_covers_step(length):
    mm, _, _ = prefix_manager(pages=1)
    request = seq(1, length)
    request.page_table = [mm.segment.allocate()]
    request.computed_token_num = length - 1
    request.prompt_len = length - 1
    s = scheduler(mm)
    s.seqs_to_decode.append(request)
    assert s.schedule_decode_batch(1) == [request]
    assert not s.freed


def test_impossible_waiter_gets_terminal_error_instead_of_idle_loop():
    mm, _, _ = prefix_manager(pages=1)
    request = seq(1, 17)
    populate(mm.segment, seq(0, 16))  # unrelated pinned allocation
    s = scheduler(mm)
    s.add_new_requests([request])
    assert s.schedule_once() == []
    package = s.check_abort_seqs()
    assert package.free_ids == [1]
    assert 'Insufficient cache capacity' in package.request_errors[1]
    assert not s.seqs_to_prefill
    assert s.check_abort_seqs() is None


def test_no_progress_releases_partial_owner_and_wakes_next_waiter():
    mm, _, _ = prefix_manager(pages=1)
    big, small = seq(1, 32), GenerationSequence(2, [777], [], output_len=1)
    big.page_table = [mm.segment.allocate()]
    big.computed_token_num = 16
    s = scheduler(mm)
    s.add_new_requests([big, small])
    assert not s.schedule_once()
    assert s.check_abort_seqs().free_ids == [1]
    assert mm.get_num_free_pages() == 1
    assert s.schedule_once() == [small]


def test_inflight_work_can_free_capacity_so_waiter_is_not_failed():
    mm, _, _ = prefix_manager(pages=1)
    s = scheduler(mm)
    s.pp_size = 2
    running = seq(0, 16)
    populate(mm.segment, running)
    s.batch_running.append([running])
    s.add_new_requests([GenerationSequence(1, [777] * 16, [], output_len=16)])
    assert not s.schedule_once()
    assert not s._pending_request_errors
    assert len(s.seqs_to_prefill) == 1


def test_image_wait_is_not_reported_as_cache_exhaustion():
    mm, _, _ = prefix_manager(pages=4)
    s = scheduler(mm)
    s.model_runner.disagg_prefill_limit = lambda request: 0
    s.add_new_requests([seq(1, 16)])
    assert not s.schedule_once()
    assert not s._pending_request_errors
    assert len(s.seqs_to_prefill) == 1
    assert mm.get_num_free_pages() == 4


def test_preemption_allows_retry_before_terminal_capacity_error():
    mm, _, _ = prefix_manager(pages=2)
    s = scheduler(mm)
    requests = [seq(1, 17), GenerationSequence(2, [777] * 17, [], output_len=16)]
    for request in requests:
        request.page_table = [mm.segment.allocate()]
        request.computed_token_num = 16
        request.prompt_len = request.raw_prompt_len = 16
    s.seqs_to_decode.extend(requests)
    # Both decoders need another page; preemption changes the population that
    # the reserve was based on. Retry without that stale reserve next tick.
    assert not s.schedule_once()
    assert s.num_preempt_seqs == 2
    assert not s._pending_request_errors
    assert s.schedule_once()
    assert not s._pending_request_errors
