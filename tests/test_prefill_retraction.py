"""Cache pressure retracts live requests without terminating their streams."""
from collections import defaultdict

import pytest

from gllm.runtime.sequence import GenerationSequence
from gllm.scheduling.distributed import DriverPayloadBuilder, FollowerSeqStore
from test_prefix_cache_reuse import prefix_manager
from test_scheduler_liveness import scheduler


def stalled_requests(*, method="chunked_prefill", count=2, pages_each=5):
    mm, _, _ = prefix_manager(pages=count * pages_each)
    s = scheduler(mm, method)
    requests = []
    for sid in range(1, count + 1):
        request = GenerationSequence(sid, [sid] * 128, [], output_len=3)
        request.page_table = [
            mm.segment.allocate(request, end)
            for end in range(16, pages_each * 16 + 1, 16)
        ]
        request.computed_token_num = pages_each * 16
        requests.append(request)
    s.add_new_requests(requests)
    return s, requests


@pytest.mark.parametrize("method", ["chunked_prefill", "split_pd", "token_throttling"])
@pytest.mark.parametrize("chunk", [16, 64, 128])
@pytest.mark.parametrize("count", [2, 3])
def test_retracted_requests_all_finish_without_errors_or_retraction_loop(method, chunk, count):
    s, requests = stalled_requests(method=method, count=count)
    s.maxp = chunk
    victim = requests[0]
    assert not s.schedule_once()
    assert s.num_preempt_seqs == 1
    assert not victim.page_table and victim.computed_token_num == 0
    assert list(s.seqs_to_prefill)[-1] is victim
    assert not s._pending_request_errors
    assert not s._pending_follower_frees
    assert s.check_abort_seqs() is None

    # New arrivals must not immediately consume the capacity just released.
    newcomer = GenerationSequence(99, [99] * 32, [], output_len=3)
    s.add_new_requests([newcomer])
    emitted = defaultdict(list)
    finished = []
    for tick in range(100):
        batch = s.schedule_once()
        assert not s._pending_request_errors
        if batch:
            if any(seq.seq_id == victim.seq_id for seq in batch):
                assert set(range(2, count + 1)) <= set(finished)
            if any(seq.seq_id == newcomer.seq_id for seq in batch):
                assert victim.seq_id in finished
            s.add_next_tokens([100 + seq.seq_id for seq in batch])
            output = s.process_output()
            assert not output.request_errors
            for sid, token in zip(output.act_schedule_ids, output.next_tokens):
                emitted[sid].append(token)
            finished.extend(output.free_ids)
        if len(finished) == count + 1:
            break
    else:
        pytest.fail("Requests did not finish within the bounded scheduling loop")
    assert sorted(finished) == list(range(1, count + 1)) + [99]
    assert dict(emitted) == {sid: [100 + sid] * 3 for sid in finished}
    assert s.num_preempt_seqs <= count
    assert not s.schedule_once()
    assert not s._prefill_recovery_ids


def test_full_retraction_releases_state_and_preserves_generated_history():
    s, requests = stalled_requests()
    victim = requests[0]
    # This request already generated tokens before an earlier recompute.
    victim.raw_prompt_len = 64
    victim.output_len = 80
    tokens = list(victim.token_ids)
    victim.recurrent_state_slot = 7
    victim.ssm_block_table = [7, 8, 9, 10]
    victim.ssm_num_accepted = 3
    victim._mtp_relay_only_next = True
    victim.to_compute_tokens = [123]
    s.memory_manager._pending_ssm_restores[victim.seq_id] = 17
    released = []
    def free_state(seq):
        released.extend(seq.ssm_block_table or [])
        seq.recurrent_state_slot = seq.ssm_block_table = None
    s.memory_manager.free_recurrent_slot = free_state

    assert not s.schedule_once()
    assert released == [7, 8, 9, 10]
    assert victim.recurrent_state_slot is None and victim.ssm_block_table is None
    assert victim.ssm_num_accepted == 1
    assert not victim._mtp_relay_only_next
    assert victim.to_compute_tokens is None and victim.to_compute_token_num == 0
    assert victim.token_ids == tokens
    assert victim.raw_prompt_len == 64 and victim.output_len == 80
    assert victim.prompt_len == len(tokens)
    assert victim.seq_id not in s.memory_manager._pending_ssm_restores
    assert not s._pending_request_errors and not s._pending_follower_frees


def test_shared_prefix_remains_owned_by_survivor_until_it_finishes():
    mm, _, _ = prefix_manager(pages=8)
    s = scheduler(mm)
    s.maxp = 16
    a = GenerationSequence(1, [7] * 32 + [1] * 80, [], output_len=3)
    b = GenerationSequence(2, [7] * 32 + [2] * 80, [], output_len=3)
    a.page_table = [mm.segment.allocate(a, end) for end in range(16, 81, 16)]
    a.computed_token_num = 80
    mm.pre_allocate_computed_page([b])
    assert b.page_table == a.page_table[:2]
    b.page_table.extend(mm.segment.allocate(b, end) for end in range(48, 81, 16))
    b.computed_token_num = 80
    s.add_new_requests([a, b])
    assert not s.schedule_once()
    # A had five logical pages, but its two shared pages stay pinned by B.
    assert mm.get_num_free_pages() == 3
    finished = []
    for _ in range(40):
        batch = s.schedule_once()
        assert not s._pending_request_errors
        if batch:
            s.add_next_tokens([100 + seq.seq_id for seq in batch])
            finished.extend(s.process_output().free_ids)
        if len(finished) == 2:
            break
    assert finished == [2, 1]
    assert s.num_preempt_seqs == 1
    assert mm.get_num_free_pages() == 8


def test_cancelling_retracted_request_restores_normal_admission():
    s, requests = stalled_requests()
    assert not s.schedule_once()
    s.abort_ids.add(requests[0].seq_id)
    assert s.check_abort_seqs().free_ids == [requests[0].seq_id]
    s.schedule_once()
    assert not s._prefill_recovery_ids


def test_external_image_wait_does_not_fail_a_capacity_blocked_neighbor():
    mm, _, _ = prefix_manager(pages=1)
    s = scheduler(mm)
    image = GenerationSequence(1, [1] * 16, [], output_len=1)
    image.page_table = [mm.segment.allocate()]
    image.computed_token_num = 8
    neighbor = GenerationSequence(2, [2], [], output_len=1)
    s.model_runner.disagg_prefill_limit = lambda seq: 8 if seq is image else None
    s.add_new_requests([image, neighbor])
    assert not s.schedule_once()
    assert not s._pending_request_errors
    assert s.num_preempt_seqs == 0
    s.model_runner.disagg_prefill_limit = lambda seq: None
    assert s.schedule_once() == [image]


@pytest.mark.parametrize("new_pages", [[8], [8, 9], [8, 9, 10]])
def test_follower_replaces_reallocated_page_table_after_preemption(new_pages):
    builder = DriverPayloadBuilder()
    follower = FollowerSeqStore()
    request = GenerationSequence(1, [1] * 64, [], output_len=3)
    request.page_table = [2, 3]
    request.to_compute_token_num = 16
    original = follower.apply_payload(builder.build([request], []))[0]
    request.preempt()
    request.page_table = list(new_pages)
    request.to_compute_token_num = 16
    payload = builder.build([request], [])
    assert not payload.frees and not payload.registers
    assert payload.updates[0].page_table_reset == []
    resumed = follower.apply_payload(payload)[0]
    assert resumed is original
    assert resumed.page_table == new_pages
    assert builder.build([request], []).updates[0].page_table_reset is None
    builder.forget(request.seq_id)
    assert request.seq_id not in builder._last_cache_epoch
