"""DSpark scheduling and worker glue for synchronous single-request execution."""

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
    """Select one live phase without speculative progress or deep copies."""
    queue = scheduler.seqs_to_decode if scheduler.seqs_to_decode else scheduler.seqs_to_prefill
    if not queue:
        return []
    seq = queue.popleft()
    if seq.seq_id in scheduler.abort_ids or seq.is_abort:
        seq.is_abort = True
        queue.appendleft(seq)
        return []
    try:
        scheduler.model_runner._validate_dspark_request(seq)
        state = scheduler.model_runner._get_dspark_request(seq)
    except ValueError as exc:
        scheduler._reject_dspark_request(seq, str(exc))
        return []
    if state.phase == "prefill":
        count = min(
            seq.prompt_len - seq.computed_token_num,
            scheduler.maxp, scheduler.max_num_batched_tokens,
        )
    elif state.phase == "bootstrap":
        count = 1
    elif state.phase in ("verify", "tail"):
        if state.phase == "verify" and not scheduler.model_runner._dspark_verify_fits(seq):
            state.phase = "tail"
        count = 1 if state.phase == "tail" else 8
    else:
        raise ValueError("Unknown DSpark request phase")
    if not 0 < count <= scheduler.max_num_batched_tokens:
        scheduler._reject_dspark_request(seq, "DSpark phase exceeds the target token budget")
        return []
    while True:
        end = seq.computed_token_num + count
        try:
            reserved = scheduler.model_runner._reserve_dspark_step(
                seq, state, target_length=end,
                draft_length=state.cache.context_length if state.phase == "tail" else end,
            )
        except (ValueError, RequestCapacityError, torch.cuda.OutOfMemoryError) as exc:
            # Only a published request record gives cleanup known ownership.
            # Initial GDN allocation may fail before its slots reach the seq;
            # propagate that case rather than pretending it was reclaimed.
            if scheduler.model_runner._dspark_requests.get(seq.seq_id) is not state:
                raise
            scheduler.model_runner.complete_dspark_failed_step(seq)
            scheduler._reject_dspark_request(seq, str(exc))
            return []
        if reserved:
            break
        if state.phase != "prefill" or count == 1:
            scheduler._reject_dspark_request(seq, "Insufficient cache capacity for the DSpark phase")
            return []
        # A failed joint claim rolls back every new page/state slot.
        # Halving bounds retry work while retaining a contiguous chunk.
        count = max(1, count // 2)
    seq.to_compute_token_num = count
    return [seq]


def process_dspark_output(scheduler, seqs, result, ipc_package):
    if len(seqs) != 1 or not isinstance(result, DSparkStepResult):
        raise ValueError("DSpark finalize requires one request and its step result")
    seq = seqs[0]
    state = scheduler.model_runner._dspark_requests.get(seq.seq_id)
    if state is None:
        raise ValueError("DSpark finalize has no live request record")
    if seq.seq_id in scheduler.abort_ids or seq.is_abort:
        seq.is_abort = True
        scheduler.abort_ids.discard(seq.seq_id)
        scheduler._retire_seq(seq, ipc_package)
        return ipc_package
    try:
        if result.phase in ("prefill", "bootstrap"):
            published = commit_dspark_prompt_step(seq, state, result)
        elif result.phase == "tail":
            published = commit_dspark_tail_step(seq, state, result)
        else:
            published = commit_dspark_verify_step(seq, state, result)
    except ValueError as exc:
        # Forward completed, but this result cannot be committed. Terminate
        # the request rather than replaying an already-mutated rolling state.
        scheduler._reject_dspark_request(seq, str(exc))
        ipc_package.free_ids.append(seq.seq_id)
        ipc_package.request_errors[seq.seq_id] = scheduler._pending_request_errors.pop(seq.seq_id)
        return ipc_package
    if published:
        ipc_package.act_schedule_ids.append(seq.seq_id)
        ipc_package.next_tokens.append(published)
        ipc_package.logprobs.append(None)
    if seq.is_finish:
        scheduler._retire_seq(seq, ipc_package)
    elif seq.computed_prompt:
        scheduler.seqs_to_decode.appendleft(seq)
    else:
        scheduler.seqs_to_prefill.appendleft(seq)
    return ipc_package


def schedule_forward_dspark(worker):
    """Run one synchronous DSpark batch and preserve request-level failures."""
    seqs = worker.scheduler.schedule_once()
    if not seqs:
        return
    try:
        worker.model_runner.prepare_input(seqs)
        worker.scheduler.add_next_tokens(worker.model_runner.step_once())
    except (ValueError, RequestCapacityError, torch.cuda.OutOfMemoryError) as exc:
        # Scheduling handles its own claims before batch_running is updated.
        # This boundary covers only the already-dispatched live batch.
        seq = seqs[0]
        worker.model_runner.complete_dspark_failed_step(seq)
        worker.scheduler._reject_dspark_request(seq, str(exc))
        worker.scheduler.batch_running.pop()
