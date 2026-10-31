"""DSpark phase batching and worker glue for synchronous execution."""

import torch

from gllm.runtime.sequence import RequestCapacityError
from gllm.speculative.dspark import (
    DSparkStepResult, commit_dspark_prompt_step, commit_dspark_tail_step,
    commit_dspark_verify_step,
)


def reject_dspark_request(scheduler, seq, message):
    scheduler.model_runner.free(seq)
    seq.to_compute_token_num = 0
    scheduler._pending_follower_frees.append(seq.seq_id)
    scheduler._pending_request_errors[seq.seq_id] = message


def schedule_dspark(scheduler):
    """Select a homogeneous phase, respecting admission and total token limits.

    Multi-request execution alternates prefill and decode opportunities.
    Existing active prefills can still advance when new arrivals cannot be
    admitted; a blocked prefill queue must not prevent decoding resident work.
    """
    capacity = scheduler.maxd
    prefer_prefill = capacity > 1 and getattr(scheduler, "_dspark_prefer_prefill", True)
    queues = (
        (scheduler.seqs_to_prefill, scheduler.seqs_to_decode) if prefer_prefill
        else (scheduler.seqs_to_decode, scheduler.seqs_to_prefill)
    )
    runner = scheduler.model_runner
    for queue in queues:
        selected, phase = [], None
        budget = scheduler.max_num_batched_tokens
        for _ in range(len(queue)):
            if len(selected) >= capacity or budget == 0:
                break
            seq = queue.popleft()
            if seq.seq_id in scheduler.abort_ids or seq.is_abort:
                seq.is_abort = True
                scheduler.abort_ids.add(seq.seq_id)
                queue.append(seq)
                if capacity == 1:
                    return []
                continue
            if seq.seq_id not in runner._dspark_requests and len(runner._dspark_requests) >= capacity:
                queue.append(seq)
                continue
            try:
                runner._validate_dspark_request(seq)
                state = runner._get_dspark_request(seq)
            except ValueError as exc:
                scheduler._reject_dspark_request(seq, str(exc))
                if capacity == 1:
                    return []
                continue
            if state.phase == "verify" and not runner._dspark_verify_fits(seq):
                state.phase = "tail"
            if state.phase not in ("prefill", "bootstrap", "verify", "tail"):
                scheduler._reject_dspark_request(seq, "Unknown DSpark request phase")
                if capacity == 1:
                    return []
                continue
            if phase is not None and state.phase != phase:
                queue.append(seq)
                continue
            if state.phase == "prefill":
                count = min(
                    seq.prompt_len - seq.computed_token_num,
                    scheduler.maxp, budget,
                )
            else:
                count = 8 if state.phase == "verify" else 1
            if count > budget:
                queue.append(seq)
                continue
            if count <= 0:
                scheduler._reject_dspark_request(seq, "DSpark phase has no valid input rows")
                if capacity == 1:
                    return []
                continue
            reserved = False
            while True:
                end = seq.computed_token_num + count
                try:
                    reserved = runner._reserve_dspark_step(
                        seq, state, target_length=end,
                        draft_length=state.cache.context_length if state.phase == "tail" else end,
                    )
                except (ValueError, RequestCapacityError, torch.cuda.OutOfMemoryError) as exc:
                    # Initial GDN allocation may fail before its ownership is
                    # published. Propagate that case; only known owners can be
                    # covered by a completion event and reclaimed here.
                    if runner._dspark_requests.get(seq.seq_id) is not state:
                        raise
                    runner.complete_dspark_failed_step(seq)
                    scheduler._reject_dspark_request(seq, str(exc))
                    break
                if reserved:
                    break
                if state.phase != "prefill" or count == 1:
                    scheduler._reject_dspark_request(
                        seq, "Insufficient cache capacity for the DSpark phase"
                    )
                    break
                count = max(1, count // 2)
            if not reserved:
                if capacity == 1:
                    return []
                continue
            seq.to_compute_token_num = count
            selected.append(seq)
            phase = state.phase
            budget -= count
        if selected:
            scheduler._dspark_prefer_prefill = queue is scheduler.seqs_to_decode
            return selected
    return []


def process_dspark_output(scheduler, seqs, result, ipc_package):
    results = [result] if isinstance(result, DSparkStepResult) else result
    if (
        not seqs or not isinstance(results, list) or len(results) != len(seqs)
        or any(not isinstance(item, DSparkStepResult) or item.seq_id != seq.seq_id
               for seq, item in zip(seqs, results))
    ):
        raise ValueError("DSpark finalize requires one aligned result per request")
    for seq, result in zip(seqs, results):
        state = scheduler.model_runner._dspark_requests.get(seq.seq_id)
        if state is None:
            raise ValueError("DSpark finalize has no live request record")
        if seq.seq_id in scheduler.abort_ids or seq.is_abort:
            seq.is_abort = True
            scheduler.abort_ids.discard(seq.seq_id)
            scheduler._retire_seq(seq, ipc_package)
            continue
        try:
            if result.phase in ("prefill", "bootstrap"):
                published = commit_dspark_prompt_step(seq, state, result)
            elif result.phase == "tail":
                published = commit_dspark_tail_step(seq, state, result)
            else:
                published = commit_dspark_verify_step(seq, state, result)
        except ValueError as exc:
            # Completion covers the shared GPU batch, but failure/retirement
            # belongs to this owner. Other results can still be committed.
            scheduler._reject_dspark_request(seq, str(exc))
            ipc_package.free_ids.append(seq.seq_id)
            ipc_package.request_errors[seq.seq_id] = scheduler._pending_request_errors.pop(seq.seq_id)
            continue
        if published:
            ipc_package.act_schedule_ids.append(seq.seq_id)
            ipc_package.next_tokens.append(published)
            ipc_package.logprobs.append(None)
        if seq.is_finish:
            scheduler._retire_seq(seq, ipc_package)
        else:
            queue = scheduler.seqs_to_decode if seq.computed_prompt else scheduler.seqs_to_prefill
            if scheduler.maxd == 1:
                queue.appendleft(seq)
            else:
                queue.append(seq)
    return ipc_package


def schedule_forward_dspark(worker):
    """Dispatch one batch; a partial shared forward is never retried.

    Per-request validation/capacity failures are handled before dispatch.
    Once a shared forward may have started, all batch members are affected;
    cover their use before freeing them. Other queued/active requests remain
    runnable. Unknown CUDA/context or completion failures still exit Worker.
    """
    seqs = worker.scheduler.schedule_once()
    if not seqs:
        return
    try:
        worker.model_runner.prepare_input(seqs)
        worker.scheduler.add_next_tokens(worker.model_runner.step_once())
    except (ValueError, RequestCapacityError, torch.cuda.OutOfMemoryError) as exc:
        owners = {seq.seq_id: seq for seq in seqs}
        # Establish every affected owner's completion boundary before freeing
        # any member. A failed event must propagate without partial recovery.
        for seq in owners.values():
            worker.model_runner.complete_dspark_failed_step(seq)
        for seq in owners.values():
            worker.scheduler._reject_dspark_request(seq, str(exc))
        worker.scheduler.batch_running.pop()
