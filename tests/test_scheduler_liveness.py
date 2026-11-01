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


def _dspark_scheduler_spy(prefix_manager, make_scheduler):
    from types import SimpleNamespace
    from gllm.scheduling.scheduler import Scheduler

    manager, _, _ = prefix_manager(pages=4)
    base = make_scheduler(manager).model_runner
    base.dspark_enabled, base.maxd, base.maxp = True, 1, 8
    base.max_num_batched_tokens = 8
    base._dspark_requests = {}
    base._validate_dspark_request = lambda seq: None
    base._get_dspark_request = lambda seq: SimpleNamespace(
        phase="verify" if seq.computed_prompt else "prefill"
    )
    base._dspark_verify_fits = lambda seq: True
    calls = []

    def reserve(seq, state, *, target_length, draft_length):
        calls.append((target_length, draft_length))
        return target_length <= getattr(base, "limit", 64)

    base._reserve_dspark_step = reserve
    scheduler = Scheduler(1, base, "chunked_prefill")
    scheduler.log = False
    return scheduler, calls


def test_dspark_scheduler_budgets_eight_verify_rows_without_mtp(prefix_manager, make_scheduler):
    scheduler, calls = _dspark_scheduler_spy(prefix_manager, make_scheduler)
    seq = GenerationSequence(91, list(range(13)), [], output_len=32, temperature=0)
    seq.computed_token_num = seq.prompt_len
    scheduler.seqs_to_decode.append(seq)
    assert scheduler.schedule_once() == [seq]
    assert seq.to_compute_token_num == 8
    assert calls == [(21, 21)]
    assert scheduler.model_runner.mtp_enabled is False


def test_dspark_scheduler_shrinks_joint_claim_and_rejects_an_impossible_owner(
    prefix_manager, make_scheduler
):
    scheduler, calls = _dspark_scheduler_spy(prefix_manager, make_scheduler)
    scheduler.model_runner.limit = 2
    big = GenerationSequence(92, list(range(13)), [], output_len=32, temperature=0)
    small = GenerationSequence(93, [1], [], output_len=1, temperature=0)
    scheduler.add_new_requests([big, small])
    assert scheduler.schedule_once() == [big]
    assert big.to_compute_token_num == 2 and calls == [(8, 8), (4, 4), (2, 2)]
    # Simulate successful completion of that prefix without allocating GPU data.
    scheduler.batch_running.popleft()
    big.computed_token_num, big.to_compute_token_num = 2, 0
    scheduler.seqs_to_prefill.appendleft(big)
    assert scheduler.schedule_once() == []
    package = scheduler.check_abort_seqs()
    assert package.free_ids == [big.seq_id]
    assert "Insufficient cache capacity" in package.request_errors[big.seq_id]
    assert scheduler.schedule_once() == [small]
    assert small.to_compute_token_num == 1


def _dspark_claim_failure_spy(prefix_manager, make_scheduler, *, owned=True):
    """CPU allocation/dispatch control; no model forward or CUDA fault is run."""
    from types import SimpleNamespace
    from gllm.workers.worker import Worker

    scheduler, _ = _dspark_scheduler_spy(prefix_manager, make_scheduler)
    runner = scheduler.model_runner
    owner = GenerationSequence(94, list(range(13)), [], output_len=32, temperature=0)
    waiter = GenerationSequence(95, [1], [], output_len=1, temperature=0)
    state = SimpleNamespace(phase="verify" if owned else "prefill")
    if owned:
        owner.computed_token_num = owner.prompt_len
        runner._dspark_requests[owner.seq_id] = state
        scheduler.seqs_to_decode.append(owner)
    else:
        scheduler.add_new_requests([owner])
    scheduler.add_new_requests([waiter])
    old_get, old_free = runner._get_dspark_request, runner.free
    runner._get_dspark_request = lambda seq: state if seq is owner else old_get(seq)
    events = []

    def complete(seq):
        assert runner._dspark_requests.get(seq.seq_id) is state
        events.append(("complete", seq.seq_id))

    def free(seq):
        events.append(("free", seq.seq_id))
        old_free(seq)
        runner._dspark_requests.pop(seq.seq_id, None)

    runner.complete_dspark_failed_step, runner.free = complete, free
    runner.prepare_input = lambda seqs: events.append(("prepare", seqs[0].seq_id))
    output = object()
    runner.step_once = lambda: output
    worker = Worker.__new__(Worker)
    worker.model_runner, worker.scheduler = runner, scheduler
    return worker, owner, waiter, events, output


@pytest.mark.parametrize("kind", ["validation", "capacity", "oom"])
def test_dspark_schedule_claim_error_finishes_owner_once_and_runs_waiter(
    prefix_manager, make_scheduler, kind
):
    import torch
    from gllm.runtime.sequence import RequestCapacityError

    worker, owner, waiter, events, output = _dspark_claim_failure_spy(
        prefix_manager, make_scheduler
    )
    runner, scheduler = worker.model_runner, worker.scheduler
    error_type = {
        "validation": ValueError, "capacity": RequestCapacityError,
        "oom": torch.cuda.OutOfMemoryError,
    }[kind]
    original = runner._reserve_dspark_step

    def reserve(seq, state, **kwargs):
        if seq is owner:
            raise error_type("joint claim failed")
        return original(seq, state, **kwargs)

    runner._reserve_dspark_step = reserve
    worker.schedule_forward()
    assert events == [("complete", owner.seq_id), ("free", owner.seq_id)]
    assert not scheduler.batch_running and not scheduler.next_tokens_queue
    assert not runner._dspark_requests
    package = scheduler.check_abort_seqs()
    assert package.free_ids == [owner.seq_id]
    assert package.request_errors == {owner.seq_id: "joint claim failed"}
    assert scheduler.check_abort_seqs() is None
    assert scheduler._pending_follower_frees == [owner.seq_id]
    worker.schedule_forward()
    assert events[-1] == ("prepare", waiter.seq_id)
    assert list(scheduler.batch_running) == [[waiter]]
    assert scheduler.next_tokens_queue[-1][0] is output


@pytest.mark.parametrize("kind,owned", [
    ("validation", False), ("capacity", False), ("oom", False),
    ("fatal", False), ("fatal", True),
])
def test_dspark_schedule_unsafe_claim_error_propagates_without_cleanup(
    prefix_manager, make_scheduler, kind, owned
):
    import torch
    from gllm.runtime.sequence import RequestCapacityError

    worker, _, _, events, _ = _dspark_claim_failure_spy(
        prefix_manager, make_scheduler, owned=owned
    )
    error_type = {
        "validation": ValueError, "capacity": RequestCapacityError,
        "oom": torch.cuda.OutOfMemoryError, "fatal": RuntimeError,
    }[kind]

    def fail(*args, **kwargs):
        raise error_type("initial claim failed")

    worker.model_runner._reserve_dspark_step = fail
    with pytest.raises(error_type, match="initial claim failed"):
        worker.schedule_forward()
    assert not events
    assert not worker.scheduler.batch_running and not worker.scheduler.next_tokens_queue
    assert not worker.scheduler._pending_request_errors
    assert bool(worker.model_runner._dspark_requests) == owned


def test_dspark_schedule_completion_failure_propagates_before_free(
    prefix_manager, make_scheduler
):
    worker, owner, _, events, _ = _dspark_claim_failure_spy(
        prefix_manager, make_scheduler
    )

    def fail_claim(*args, **kwargs):
        raise ValueError("joint claim failed")

    def fail_complete(seq):
        raise RuntimeError("completion event failed")

    runner = worker.model_runner
    runner._reserve_dspark_step = fail_claim
    runner.complete_dspark_failed_step = fail_complete
    with pytest.raises(RuntimeError, match="completion event failed"):
        worker.schedule_forward()
    assert not events and owner.seq_id in runner._dspark_requests
    assert not worker.scheduler._pending_request_errors


def test_dspark_finalize_error_notifies_frontend_once(
    prefix_manager, make_scheduler, monkeypatch
):
    from types import SimpleNamespace
    from gllm.distributed.comm import IPCPackage
    from gllm.speculative import dspark_runtime
    from gllm.speculative.dspark import DSparkStepResult

    scheduler, _ = _dspark_scheduler_spy(prefix_manager, make_scheduler)
    seq = GenerationSequence(96, [1], [], output_len=1, temperature=0)
    scheduler.model_runner._dspark_requests[seq.seq_id] = SimpleNamespace(phase="prefill")
    released = []

    def release(request):
        released.append(request.seq_id)
        scheduler.model_runner._dspark_requests.pop(request.seq_id)

    scheduler.model_runner.free = release

    def fail_commit(*args, **kwargs):
        raise ValueError("stale result")

    monkeypatch.setattr(dspark_runtime, "commit_dspark_prompt_step", fail_commit)
    result = DSparkStepResult(seq.seq_id, seq._cache_epoch, 0, 1, "prefill", ready=True)
    package = scheduler._process_dspark_output([seq], result, IPCPackage([]))
    assert released == [seq.seq_id] and package.free_ids == [seq.seq_id]
    assert package.request_errors == {seq.seq_id: "stale result"}
    assert package.act_schedule_ids == package.next_tokens == []
    assert not scheduler.model_runner._dspark_requests
    assert not scheduler._pending_request_errors
    assert scheduler.check_abort_seqs() is None
    assert scheduler._pending_follower_frees == [seq.seq_id]


def test_dspark_scheduler_batch_skips_blocked_admission_and_runs_resident_decode(
    prefix_manager, make_scheduler
):
    from types import SimpleNamespace

    scheduler, calls = _dspark_scheduler_spy(prefix_manager, make_scheduler)
    runner = scheduler.model_runner
    runner.maxd, runner.max_num_batched_tokens = 2, 16
    states = {}
    active = []
    for index in (101, 102):
        seq = GenerationSequence(index, [1, 2, 3], [], output_len=16, temperature=0)
        seq.computed_token_num = seq.prompt_len
        active.append(seq)
        states[index] = SimpleNamespace(phase="verify")
    runner._dspark_requests = states
    runner._get_dspark_request = lambda seq: states[seq.seq_id]
    waiting = GenerationSequence(103, [1, 2], [], output_len=16, temperature=0)
    scheduler.add_new_requests([waiting])
    scheduler.seqs_to_decode.extend(active)
    scheduler._dspark_prefer_prefill = True
    assert scheduler.schedule_once() == active
    assert sum(seq.to_compute_token_num for seq in active) == 16
    assert list(scheduler.seqs_to_prefill) == [waiting]
    assert calls == [(11, 11), (11, 11)]


def test_dspark_scheduler_batch_groups_phases_and_defers_other_ready_owners(
    prefix_manager, make_scheduler
):
    from types import SimpleNamespace

    scheduler, calls = _dspark_scheduler_spy(prefix_manager, make_scheduler)
    runner = scheduler.model_runner
    runner.maxd, runner.max_num_batched_tokens = 3, 24
    seqs = [GenerationSequence(index, [1, 2, 3], [], output_len=16, temperature=0)
            for index in (111, 112, 113)]
    for seq in seqs:
        seq.computed_token_num = seq.prompt_len
    states = {seq.seq_id: SimpleNamespace(phase=phase)
              for seq, phase in zip(seqs, ("verify", "bootstrap", "verify"))}
    runner._dspark_requests = states
    runner._get_dspark_request = lambda seq: states[seq.seq_id]
    scheduler.seqs_to_decode.extend(seqs)
    assert scheduler.schedule_once() == [seqs[0], seqs[2]]
    assert calls == [(11, 11), (11, 11)]
    assert list(scheduler.seqs_to_decode) == [seqs[1]]
    assert seqs[1].to_compute_token_num == 0



def _dspark_capacity_scheduler(batch=2):
    """Real scheduler/request cleanup with CPU joint-reservation failure control."""
    from types import MethodType, SimpleNamespace
    from gllm.scheduling.scheduler import Scheduler
    from gllm.speculative.dspark import DSparkRequestState
    from gllm.speculative.dspark_execution import DSparkExecutionMixin

    calls, freed = [], []
    manager = SimpleNamespace()
    runner = SimpleNamespace(
        dspark_enabled=True, mtp_enabled=False, memory_manager=manager,
        maxd=batch, max_running_seqs=batch, maxp=8 * batch,
        max_num_batched_tokens=8 * batch, minp=1, iterp=1, page_size=16,
        init_new_token_ratio=0.1, min_new_token_ratio=0.1,
        _dspark_requests={}, can_reserve=lambda seq: True,
    )
    runner._validate_dspark_request = lambda seq: None
    runner._get_dspark_request = MethodType(DSparkExecutionMixin._get_dspark_request, runner)
    runner._reserve_dspark_step = MethodType(DSparkExecutionMixin._reserve_dspark_step, runner)
    runner._dspark_verify_fits = lambda seq: True

    def reserve(seq, cache, *, target_length, draft_length):
        allowed = runner.can_reserve(seq)
        calls.append((seq.seq_id, target_length, draft_length, allowed))
        if not allowed:
            return False
        for table, length in ((seq.page_table, target_length), (cache.page_table, draft_length)):
            while len(table) * 16 < length:
                table.append(seq.seq_id * 16 + len(table))
        if seq.ssm_block_table is None:
            seq.ssm_block_table = list(range(seq.seq_id * 8, seq.seq_id * 8 + 8))
            seq.recurrent_state_slot = seq.ssm_block_table[0]
        return True

    def release(seq):
        freed.append(seq.seq_id)
        seq.page_table.clear()
        seq.ssm_block_table = None
        seq.recurrent_state_slot = None

    def release_owned(seq, cache):
        cache.reset()
        release(seq)

    manager.reserve_dspark_pages = reserve
    manager.free_dspark_cache = release_owned

    def free(seq):
        if not DSparkExecutionMixin._free_dspark_request(runner, seq):
            release(seq)
    runner.free = free
    scheduler = Scheduler(1, runner, "chunked_prefill")
    scheduler.log = False

    class Event:
        waited = False

        def synchronize(self):
            self.waited = True

    def own(seq, phase="verify"):
        # Prime an already computed owner; fresh public requests must start at zero.
        state = DSparkRequestState(seq.seq_id, seq._cache_epoch, phase=phase)
        assert runner._reserve_dspark_step(
            seq, state, target_length=seq.computed_token_num,
            draft_length=seq.computed_token_num,
        )
        state.cache.context_length = seq.computed_token_num
        state.last_use = Event()
        return state

    return SimpleNamespace(scheduler=scheduler, runner=runner, own=own, calls=calls, freed=freed)


def test_dspark_capacity_defers_new_admission_until_peer_retires():
    from gllm.distributed.comm import IPCPackage

    fixture = _dspark_capacity_scheduler()
    scheduler, runner = fixture.scheduler, fixture.runner
    peer = GenerationSequence(201, list(range(13)), [], output_len=32, temperature=0)
    peer.computed_token_num = peer.prompt_len
    fixture.own(peer)
    waiting = GenerationSequence(202, [1, 2], [], output_len=4, temperature=0)
    scheduler.seqs_to_decode.append(peer)
    scheduler.add_new_requests([waiting])
    runner.can_reserve = lambda seq: seq is peer or peer.seq_id not in runner._dspark_requests
    assert scheduler.schedule_once() == [peer]
    assert list(scheduler.seqs_to_prefill) == [waiting]
    assert not scheduler._pending_request_errors and scheduler.num_preempt_seqs == 0
    scheduler.batch_running.popleft()
    scheduler._retire_seq(peer, IPCPackage([]))
    assert scheduler.schedule_once() == [waiting]
    assert not scheduler._pending_request_errors


@pytest.mark.parametrize("phase", ["prefill", "verify"])
def test_dspark_capacity_retracts_blocked_owners_and_keeps_readmissions_out(phase):
    from gllm.distributed.comm import IPCPackage

    fixture = _dspark_capacity_scheduler()
    scheduler, runner = fixture.scheduler, fixture.runner
    seqs = [
        GenerationSequence(sid, list(range(32 if phase == "prefill" else 13)), [],
                           output_len=32, temperature=0)
        for sid in (211, 212)
    ]
    for seq in seqs:
        seq.computed_token_num = 4 if phase == "prefill" else seq.prompt_len
        fixture.own(seq, phase)
    first, victim = seqs
    old_state = runner._dspark_requests[victim.seq_id]
    event, history, epoch = old_state.last_use, list(victim.token_ids), victim._cache_epoch
    queue = scheduler.seqs_to_prefill if phase == "prefill" else scheduler.seqs_to_decode
    queue.extend(seqs)
    runner.can_reserve = lambda seq: len(runner._dspark_requests) < 2
    assert scheduler.schedule_once() == []
    assert scheduler.num_preempt_seqs == 1 and fixture.freed == [victim.seq_id]
    assert event.waited and victim._cache_epoch == epoch + 1
    assert victim.token_ids == history and victim.computed_token_num == 0
    assert old_state.cache.context_length == 0 and not old_state.cache.page_table
    assert scheduler.check_abort_seqs() is None
    assert not scheduler._pending_follower_frees
    assert scheduler.schedule_once() == [first]
    assert victim.seq_id not in runner._dspark_requests
    assert not scheduler._pending_request_errors
    scheduler.batch_running.popleft()
    scheduler._retire_seq(first, IPCPackage([]))
    assert scheduler.schedule_once() == [victim]
    assert runner._dspark_requests[victim.seq_id] is not old_state
    assert runner._dspark_requests[victim.seq_id].cache_epoch == epoch + 1
    assert scheduler._dspark_recovery_owner is None


def test_dspark_capacity_preserves_a_selected_batch_and_its_waiting_peer():
    fixture = _dspark_capacity_scheduler()
    scheduler, runner = fixture.scheduler, fixture.runner
    seqs = [GenerationSequence(sid, list(range(13)), [], output_len=32, temperature=0)
            for sid in (221, 222)]
    for seq in seqs:
        seq.computed_token_num = seq.prompt_len
        fixture.own(seq)
    scheduler.seqs_to_decode.extend(seqs)
    runner.can_reserve = lambda seq: seq is seqs[0]
    assert scheduler.schedule_once() == [seqs[0]]
    assert list(scheduler.seqs_to_decode) == [seqs[1]]
    assert not fixture.freed and not scheduler._pending_request_errors
    assert scheduler.num_preempt_seqs == 0


def test_dspark_capacity_retries_joint_demand_after_more_than_one_retraction():
    fixture = _dspark_capacity_scheduler(batch=3)
    scheduler, runner = fixture.scheduler, fixture.runner
    seqs = [GenerationSequence(sid, list(range(13)), [], output_len=32, temperature=0)
            for sid in (231, 232, 233)]
    for seq in seqs:
        seq.computed_token_num = seq.prompt_len
        fixture.own(seq)
    scheduler.seqs_to_decode.extend(seqs)
    runner.can_reserve = lambda seq: len(runner._dspark_requests) == 1
    assert scheduler.schedule_once() == []
    assert scheduler.schedule_once() == []
    assert scheduler.num_preempt_seqs == 2
    assert scheduler.schedule_once() == [seqs[0]]
    assert len(scheduler.seqs_to_prefill) == 2
    assert not scheduler._pending_request_errors
    assert not scheduler._pending_follower_frees


def test_dspark_capacity_reports_exclusive_failure_once_and_runs_next_request():
    fixture = _dspark_capacity_scheduler()
    scheduler, runner = fixture.scheduler, fixture.runner
    owner = GenerationSequence(241, list(range(13)), [], output_len=32, temperature=0)
    owner.computed_token_num = owner.prompt_len
    state = fixture.own(owner)
    event = state.last_use
    waiting = GenerationSequence(242, [1], [], output_len=1, temperature=0)
    scheduler.seqs_to_decode.append(owner)
    scheduler.add_new_requests([waiting])
    scheduler._dspark_recovery_owner = state
    runner.can_reserve = lambda seq: seq is waiting
    assert scheduler.schedule_once() == []
    package = scheduler.check_abort_seqs()
    assert package.free_ids == [owner.seq_id]
    assert "without concurrent owners" in package.request_errors[owner.seq_id]
    assert scheduler.check_abort_seqs() is None
    assert fixture.freed == [owner.seq_id] and event.waited
    assert scheduler.schedule_once() == [waiting]


def test_dspark_capacity_completion_failure_propagates_before_queue_reset():
    from types import SimpleNamespace

    fixture = _dspark_capacity_scheduler()
    scheduler, runner = fixture.scheduler, fixture.runner
    seqs = [GenerationSequence(sid, list(range(13)), [], output_len=32, temperature=0)
            for sid in (251, 252)]
    for seq in seqs:
        seq.computed_token_num = seq.prompt_len
        fixture.own(seq)
    scheduler.seqs_to_decode.extend(seqs)
    runner.can_reserve = lambda seq: False

    def fail():
        raise RuntimeError("completion failed")
    runner._dspark_requests[seqs[1].seq_id].last_use = SimpleNamespace(synchronize=fail)
    with pytest.raises(RuntimeError, match="completion failed"):
        scheduler.schedule_once()
    assert not fixture.freed and scheduler.num_preempt_seqs == 0
    assert set(runner._dspark_requests) == {seq.seq_id for seq in seqs}
    assert list(scheduler.seqs_to_decode) == seqs
    assert not scheduler._pending_request_errors


def test_dspark_capacity_aborted_recovery_owner_does_not_block_admission():
    fixture = _dspark_capacity_scheduler()
    scheduler, runner = fixture.scheduler, fixture.runner
    seqs = [GenerationSequence(sid, list(range(13)), [], output_len=32, temperature=0)
            for sid in (261, 262)]
    for seq in seqs:
        seq.computed_token_num = seq.prompt_len
        fixture.own(seq)
    scheduler.seqs_to_decode.extend(seqs)
    runner.can_reserve = lambda seq: len(runner._dspark_requests) < 2
    assert scheduler.schedule_once() == []
    scheduler.add_abort_ids([seqs[0].seq_id])
    package = scheduler.check_abort_seqs()
    assert package.free_ids == [seqs[0].seq_id] and not package.request_errors
    assert scheduler.schedule_once() == [seqs[1]]
    assert scheduler._dspark_recovery_owner is None
