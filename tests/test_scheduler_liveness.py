"""Cache pressure must not leave all requests parked without a runnable batch."""
import pytest

from gllm.runtime.sequence import GenerationSequence


@pytest.mark.parametrize('method', ['chunked_prefill', 'split_pd', 'token_throttling'])
def test_full_hit_runs_with_all_pages_pinned(prefix_manager, populate, make_seq, make_scheduler, method):
    mm, arena, cache = prefix_manager(pages=4)
    completed = make_seq(0, 64)
    populate(mm.segment, completed)
    mm.segment.free_many(completed.page_table)
    s = make_scheduler(mm, method)
    request = make_seq(1, 64)
    s.add_new_requests([request])
    batch = s.schedule_once()
    assert batch == [request]
    assert request.to_compute_token_num == 1
    assert mm.get_num_free_pages() == 0
    assert not s._pending_request_errors


def test_pinned_partial_prefill_runs_without_free_pages(prefix_manager, populate, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=4)
    request = make_seq(1, 64)
    populate(mm.segment, request)
    request.computed_token_num = 48
    s = make_scheduler(mm)
    s.add_new_requests([request])
    assert s.schedule_once() == [request]
    assert request.to_compute_token_num == 16


def test_blocked_prefix_releases_pins_and_allows_smaller_waiter(prefix_manager, populate, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=4)
    completed = make_seq(0, 64)
    populate(mm.segment, completed)
    mm.segment.free_many(completed.page_table)
    big, small = make_seq(1, 65), GenerationSequence(2, [777], [], output_len=1)
    s = make_scheduler(mm)
    s.add_new_requests([big, small])
    assert s.schedule_once() == [small]
    assert list(s.seqs_to_prefill) == [big]
    assert big.page_table == [] and big.computed_token_num == 0
    assert big.to_compute_token_num == 0
    # The failed hit must not inflate hit counters on each admission attempt.
    assert mm.num_hit_pages == 0
    assert not s._pending_request_errors


def test_failed_admission_releases_new_recurrent_state(prefix_manager, populate, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=1)
    completed = make_seq(0, 16)
    populate(mm.segment, completed)
    mm.segment.free_many(completed.page_table)
    mm.allocate_recurrent_slot = lambda request: setattr(request, 'recurrent_state_slot', 7) or True
    released = []
    def release(request):
        released.append(request.recurrent_state_slot)
        request.recurrent_state_slot = None
    mm.free_recurrent_slot = release
    s = make_scheduler(mm)
    request = make_seq(1, 17)
    s.add_new_requests([request])
    batch, _ = s.schedule_prefill_batch(128)
    assert not batch
    assert released == [7]
    assert request.recurrent_state_slot is None
    assert request.page_table == []
    assert mm.get_num_free_pages() == 1


def test_prefill_chunk_shrinks_to_available_capacity(prefix_manager, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=2)
    s = make_scheduler(mm)
    request = make_seq(1, 64)
    s.add_new_requests([request])
    batch, tokens = s.schedule_prefill_batch(128, reserve_pages=1)
    assert batch == [request]
    assert tokens == 16
    assert mm.get_num_free_pages() == 1


@pytest.mark.parametrize('length', [8, 16])
def test_decode_does_not_preempt_when_existing_page_covers_step(prefix_manager, make_seq, make_scheduler, length):
    mm, _, _ = prefix_manager(pages=1)
    request = make_seq(1, length)
    request.page_table = [mm.segment.allocate()]
    request.computed_token_num = length - 1
    request.prompt_len = length - 1
    s = make_scheduler(mm)
    s.seqs_to_decode.append(request)
    assert s.schedule_decode_batch(1) == [request]
    assert not s.freed


def test_impossible_waiter_gets_terminal_error_instead_of_idle_loop(prefix_manager, populate, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=1)
    request = make_seq(1, 17)
    populate(mm.segment, make_seq(0, 16))  # unrelated pinned allocation
    s = make_scheduler(mm)
    s.add_new_requests([request])
    assert s.schedule_once() == []
    package = s.check_abort_seqs()
    assert package.free_ids == [1]
    assert 'Insufficient cache capacity' in package.request_errors[1]
    assert not s.seqs_to_prefill
    assert s.check_abort_seqs() is None


def test_request_that_cannot_fit_alone_releases_capacity_for_smaller_waiter(prefix_manager, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=1)
    big, small = make_seq(1, 32), GenerationSequence(2, [777], [], output_len=1)
    big.page_table = [mm.segment.allocate()]
    big.computed_token_num = 16
    s = make_scheduler(mm)
    s.add_new_requests([big, small])
    assert not s.schedule_once()
    assert s.check_abort_seqs().free_ids == [1]
    assert mm.get_num_free_pages() == 1
    assert s.schedule_once() == [small]


def test_inflight_work_can_free_capacity_so_waiter_is_not_failed(prefix_manager, populate, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=1)
    s = make_scheduler(mm)
    s.pp_size = 2
    running = make_seq(0, 16)
    populate(mm.segment, running)
    s.batch_running.append([running])
    s.add_new_requests([GenerationSequence(1, [777] * 16, [], output_len=16)])
    assert not s.schedule_once()
    assert not s._pending_request_errors
    assert len(s.seqs_to_prefill) == 1


def test_image_wait_is_not_reported_as_cache_exhaustion(prefix_manager, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=4)
    s = make_scheduler(mm)
    s.model_runner.disagg_prefill_limit = lambda request: 0
    s.add_new_requests([make_seq(1, 16)])
    assert not s.schedule_once()
    assert not s._pending_request_errors
    assert len(s.seqs_to_prefill) == 1
    assert mm.get_num_free_pages() == 4


def test_preemption_allows_retry_before_terminal_capacity_error(prefix_manager, make_seq, make_scheduler):
    mm, _, _ = prefix_manager(pages=2)
    s = make_scheduler(mm)
    requests = [make_seq(1, 17), GenerationSequence(2, [777] * 17, [], output_len=16)]
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


@pytest.mark.parametrize("k,head,gate,budget,batch,eligible,dp,expected", [
    (3, True, 0, 8, 2, True, False, 4),
    (3, True, 0, 7, 2, True, False, 4),  # Reserve before shrinking the cohort.
    (3, True, 0, 4, 1, True, False, 4),
    (3, True, 0, 3, 1, True, False, 1),
    (3, True, 2, 8, 2, True, False, 4),
    (3, True, 2, 8, 3, True, False, 1),
    (3, True, 0, 8, 2, False, False, 1),
    (3, True, 0, 8, 2, True, True, 1),
    (3, True, 0, 8, 0, True, False, 1),
    (0, True, 0, 8, 2, True, False, 1),
    (3, False, 0, 8, 2, True, False, 1),
])
def test_speculative_scheduler_reservation_width(
    monkeypatch, k, head, gate, budget, batch, eligible, dp, expected,
):
    from types import SimpleNamespace
    from gllm.scheduling.scheduler import Scheduler
    from gllm.speculative.backend import select_speculative_backend

    runner = SimpleNamespace(
        speculative_num_draft_tokens=k, mtp_max_batch=gate, max_num_batched_tokens=budget,
        model=SimpleNamespace(mtp=object() if head else None),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_runner = runner
    scheduler.speculative_backend = select_speculative_backend(
        dspark_enabled=False, mtp_enabled=True,
    )
    monkeypatch.setattr("gllm.scheduling.scheduler.is_dp_attn", lambda: dp)
    assert scheduler._decode_tokens_per_seq(batch, mtp_eligible=eligible) == expected


@pytest.mark.parametrize("mtp,method", [
    (False, "chunked_prefill"), (False, "split_pd"), (False, "token_throttling"),
    (True, "chunked_prefill"), (True, "split_pd"), (True, "token_throttling"),
])
def test_speculative_backend_preserves_target_schedule_method(mtp, method):
    from types import SimpleNamespace
    from gllm.speculative.backend import select_speculative_backend

    chunked, throttled = object(), object()
    scheduler = SimpleNamespace(
        schedule_method=method, chunked_prefill=chunked, token_throttling=throttled,
    )
    backend = select_speculative_backend(dspark_enabled=False, mtp_enabled=mtp)
    assert backend.schedule_method(scheduler) is (
        throttled if method == "token_throttling" else chunked
    )


def test_speculative_reservation_can_exceed_actual_mtp_execution(monkeypatch):
    from types import SimpleNamespace
    from gllm.speculative.backend import select_speculative_backend
    from gllm.speculative.mtp import MtpMixin

    runner = SimpleNamespace(
        speculative_num_draft_tokens=3, mtp_max_batch=0, _mtp_max_batch=0,
        max_num_batched_tokens=7, model=SimpleNamespace(mtp=object()),
    )
    backend = select_speculative_backend(dspark_enabled=False, mtp_enabled=True)
    assert backend.reservation_width(runner, 2) == 4
    monkeypatch.setattr("gllm.speculative.mtp.is_dp_attn", lambda: False)
    assert MtpMixin._mtp_decide(runner, 2) is False
    assert runner._mtp_spec_decision == (2, False)


def test_speculative_budget_shrinks_mtp_cohort_before_forward(
    prefix_manager, make_seq, make_scheduler,
):
    from types import SimpleNamespace
    from gllm.speculative.backend import select_speculative_backend

    manager, _, _ = prefix_manager(pages=4)
    scheduler = make_scheduler(manager)
    runner = scheduler.model_runner
    runner.mtp_enabled, runner.speculative_num_draft_tokens, runner.mtp_max_batch = True, 3, 0
    runner.model = SimpleNamespace(mtp=object())
    runner.max_num_batched_tokens = 7
    scheduler.speculative_backend = select_speculative_backend(
        dspark_enabled=False, mtp_enabled=True,
    )
    first, second = make_seq(21, 10), make_seq(22, 10)
    scheduler.seqs_to_decode.extend([first, second])
    assert scheduler.schedule_decode_batch(2, token_budget=7) == [first]
    assert list(scheduler.seqs_to_decode) == [second]


def test_mtp_window_survives_readmission_of_its_preemption_victim(prefix_manager, populate, make_seq, make_scheduler):
    """A preempted request's cached prefix must not take the decode window.

    Preempting the long decoder returns its cached pages to the free pool;
    the same tick re-admits it at the head of the prefill queue, and its
    prefix hit re-pins those pages. The surviving decoder's speculative
    window (len + 1 + k) was checked against that pool, so it must already
    own its pages by then.
    """
    from types import SimpleNamespace

    from gllm.runtime.sequence import GenerationSequence
    from gllm.speculative.backend import select_speculative_backend

    k = 3
    mm, _, _ = prefix_manager(pages=5)
    s = make_scheduler(mm)
    s.model_runner.speculative_num_draft_tokens = k
    s.model_runner.model = SimpleNamespace(mtp=object())
    s.speculative_backend = select_speculative_backend(dspark_enabled=False, mtp_enabled=True)

    short = GenerationSequence(1, [7] * 16, [], output_len=16)
    short.page_table = [mm.segment.allocate()]
    short.computed_token_num = short.prompt_len = 15
    long = make_seq(2, 64)
    populate(mm.segment, long)  # four cached prefix pages
    long.computed_token_num = long.prompt_len = 63
    s.seqs_to_decode.extend([short, long])
    assert mm.get_num_free_pages() == 0

    batch = s.schedule_once()

    assert s.num_preempt_seqs == 1 and long not in s.seqs_to_decode
    assert short in batch
    # The runner allocates the verify window at launch; it must already be owned.
    mm.pre_allocate_page_for_lengths([short], [len(short) + k])
    assert len(short.page_table) * 16 >= len(short) + 1 + k
