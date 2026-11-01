"""DSpark execution and validation for the synchronous runner."""

import torch

from gllm.runtime.sequence import GenerationSequence, RequestCapacityError
from gllm.speculative.dspark import (
    DSparkRequestState, DSparkStepResult, verify_dspark_target, verify_dspark_target_batched,
)


class DSparkExecutionMixin:
    """Operate on the runner-owned draft model and request-state dictionary.

    The runner creates and owns these fields; this mixin adds no separate
    request container or lifecycle. The runner retains its thin entry points.
    """

    def _validate_dspark_request(self, seq: GenerationSequence) -> None:
        if not self.dspark_enabled:
            raise ValueError("DSpark steps require the DSpark backend")
        if torch.is_autocast_enabled("cuda"):
            raise ValueError("DSpark steps do not support autocast")
        if (
            seq.mm_contents is not None
            or (seq.temperature != 0 and seq.top_k != 1)
            or seq.repetition_penalty != 1 or seq.structured_output is not None
            or seq.logprobs_enabled or seq.prompt_logprobs_enabled
        ):
            raise ValueError("DSpark requires a pure-text greedy request without logprobs")
        if (
            not 0 < seq.prompt_len < self.model_max_length
            or seq.is_abort or seq.is_finish
            or seq.output_len <= 0
            or seq.raw_prompt_len + seq.output_len > self.model_max_length
            or seq.to_compute_tokens is not None or getattr(seq, "_mtp_verify", False)
        ):
            raise ValueError("DSpark request has invalid length, lifetime or input override")

    def _get_dspark_request(self, seq: GenerationSequence) -> DSparkRequestState:
        state = self._dspark_requests.get(seq.seq_id)
        if state is None:
            capacity = getattr(self, "max_running_seqs", getattr(self, "maxd", 1))
            if len(self._dspark_requests) >= capacity or seq.computed_token_num != 0:
                raise ValueError("DSpark needs available request capacity and a position-zero start")
            state = DSparkRequestState(seq.seq_id, seq._cache_epoch)
        if (
            state.seq_id != seq.seq_id or state.cache_epoch != seq._cache_epoch
            or state.inflight is not None
        ):
            raise ValueError("DSpark request has a stale epoch or an uncommitted step")
        return state

    def _reserve_dspark_step(self, seq, state, *, target_length, draft_length):
        if not self.memory_manager.reserve_dspark_pages(
            seq, state.cache, target_length=target_length, draft_length=draft_length
        ):
            return False
        self._dspark_requests[seq.seq_id] = state
        return True

    def _dspark_verify_fits(self, seq):
        remaining = seq.output_len - (len(seq) - seq.raw_prompt_len)
        return (
            remaining >= 8 and seq.computed_token_num + 8 <= self.model_max_length
            and seq.computed_token_num + self.dspark.noise_width
            <= self.dspark.rotary_emb.cos_sin_cache.shape[0]
        )

    @torch.inference_mode()
    def _execute_dspark_prompt(self, seq: GenerationSequence) -> DSparkStepResult:
        """Execute one prefill chunk or a separate one-token bootstrap.

        This is the runner half of a step, not its scheduler commit. Reserve
        target/draft/GDN together before materializing metadata; leave target
        progress and published token_ids unchanged until the result is consumed.
        A partial failure leaves inflight set, so the caller must terminate the
        request rather than replay a mutated rolling state.
        """
        self._validate_dspark_request(seq)
        state = self._get_dspark_request(seq)
        start, count = seq.computed_token_num, seq.to_compute_token_num
        end = start + count
        if (
            not 0 < count <= self.max_num_batched_tokens
            or end > self.model_max_length
            or end > self.dspark.rotary_emb.cos_sin_cache.shape[0]
            or state.cache.context_length != start
        ):
            raise ValueError("DSpark prompt chunk exceeds its input/context limit or valid cache")
        if state.phase == "prefill":
            if not start < end <= seq.prompt_len or len(seq) != seq.prompt_len:
                raise ValueError("DSpark prefill must consume the next contiguous prompt chunk")
        elif state.phase == "bootstrap":
            if count != 1 or start != seq.prompt_len or len(seq) != end:
                raise ValueError("DSpark bootstrap must consume the published first token")
        else:
            raise ValueError("DSpark prompt step requires prefill or bootstrap phase")
        if any(not 0 <= token < self.dspark.config.vocab_size for token in seq[start:end]):
            raise ValueError("DSpark input token IDs are outside the target vocabulary")
        if not self._reserve_dspark_step(
            seq, state, target_length=end, draft_length=end
        ):
            raise RequestCapacityError("Insufficient cache capacity for the DSpark prompt step")

        result = DSparkStepResult(seq.seq_id, seq._cache_epoch, start, count, state.phase)
        state.inflight = result
        self.input_data.cal_and_set_input([seq])
        if self.uses_mrope:
            # Pure text has three identical axes; retain the wrapper's real
            # MRoPE path, including short noncontiguous views of its buffer.
            self.input_data.set_mrope_position(
                torch.arange(start, end, device="cpu").repeat(3, 1)
            )
        self._prepare_attention_metadata(self.input_data)
        hidden, taps = self.model.forward_with_hidden_taps(
            self.input_data, target_layer_ids=self.dspark.config.target_layer_ids
        )
        if state.phase == "prefill":
            state.paged_cache = self.dspark.prefill_paged(
                taps, segment=self.memory_manager.dspark_segment, cache=state.cache
            )
        else:
            # This input is already published, but its draft K/V is appended
            # only by the following draft step. Retain one independently owned
            # row, never the next (not-yet-consumed) anchor's features.
            result.hidden = taps.clone()
        if state.phase == "bootstrap" or end == seq.prompt_len:
            result.next_anchor = self.model.logits_from_hidden(hidden[-1:]).argmax(-1)
        if state.last_use is None:
            state.last_use = torch.cuda.Event()
        state.last_use.record(torch.cuda.current_stream())
        result.ready = True
        return result

    @torch.inference_mode()
    def _execute_dspark_verify(self, seq: GenerationSequence) -> DSparkStepResult:
        """Append the previous committed suffix once, then verify eight inputs."""
        self._validate_dspark_request(seq)
        state = self._get_dspark_request(seq)
        start = seq.computed_token_num
        pending = state.pending_hidden
        if (
            state.phase != "verify" or seq.to_compute_token_num != 8
            or len(seq) != start or state.anchor_pos != start or state.anchor is None
            or state.paged_cache is None or pending is None
            or pending.ndim != 3 or pending.shape[0] != 1
            or not 1 <= pending.shape[1] <= 8
            or state.cache.context_length + pending.shape[1] != start
        ):
            raise ValueError("DSpark verify needs contiguous pending taps and an unpublished anchor")
        if not self._dspark_verify_fits(seq):
            raise ValueError("DSpark verify window does not fit; use the single-token tail")
        if not self._reserve_dspark_step(
            seq, state, target_length=start + 8, draft_length=start + 8
        ):
            raise RequestCapacityError("Insufficient cache capacity for the DSpark verify window")
        result = DSparkStepResult(seq.seq_id, seq._cache_epoch, start, 8, "verify")
        state.inflight = result
        draft_hidden, _ = self.dspark.forward_draft(
            pending, state.anchor, start_pos=start, caches=state.paged_cache
        )
        state.pending_hidden = None
        proposed, _, _ = self.dspark.forward_head(draft_hidden, state.anchor)
        old_tokens, old_count = seq.to_compute_tokens, seq.to_compute_token_num
        old_verify = getattr(seq, "_mtp_verify", False)
        try:
            seq.to_compute_tokens = proposed[0].cpu().tolist()
            seq.to_compute_token_num, seq._mtp_verify = 8, True
            self.input_data.cal_and_set_input([seq])
            if self.uses_mrope:
                self.input_data.set_mrope_position(
                    torch.arange(start, start + 8, device="cpu").repeat(3, 1)
                )
            self._prepare_attention_metadata(self.input_data)
            result.verification = verify_dspark_target(
                self.model, self.input_data, block_size=self.dspark.block_size,
                target_layer_ids=self.dspark.config.target_layer_ids,
            )
        finally:
            # CPU-only restoration is safe even if CUDA has failed. Never
            # record/wait on a CUDA event in this failure cleanup.
            seq.to_compute_tokens, seq.to_compute_token_num = old_tokens, old_count
            seq._mtp_verify = old_verify
        if state.last_use is None:
            state.last_use = torch.cuda.Event()
        state.last_use.record(torch.cuda.current_stream())
        result.ready = True
        return result

    @torch.inference_mode()
    def _execute_dspark_tail(self, seq: GenerationSequence) -> DSparkStepResult:
        """Consume the private anchor with ordinary target decode, without taps."""
        self._validate_dspark_request(seq)
        state = self._get_dspark_request(seq)
        start = seq.computed_token_num
        if (
            state.phase != "tail" or seq.to_compute_token_num != 1
            or len(seq) != start or state.anchor_pos != start or state.anchor is None
            or start + 1 > self.model_max_length
        ):
            raise ValueError("DSpark tail needs one unpublished anchor within the context limit")
        if not self._reserve_dspark_step(
            seq, state, target_length=start + 1, draft_length=state.cache.context_length
        ):
            raise RequestCapacityError("Insufficient cache capacity for the DSpark tail")
        result = DSparkStepResult(seq.seq_id, seq._cache_epoch, start, 1, "tail")
        state.inflight = result
        old_tokens = seq.to_compute_tokens
        try:
            # token_ids ends at start: the input belongs to the private relay.
            # Metadata must read that input rather than an empty history slice.
            seq.to_compute_tokens = [int(state.anchor.item())]
            self.input_data.cal_and_set_input([seq])
            if self.uses_mrope:
                self.input_data.set_mrope_position(torch.tensor([[start]]).repeat(3, 1))
            self._prepare_attention_metadata(self.input_data)
            hidden = self.model(self.input_data)
            result.next_anchor = self.model.logits_from_hidden(hidden[-1:]).argmax(-1)
        finally:
            seq.to_compute_tokens = old_tokens
        if state.last_use is None:
            state.last_use = torch.cuda.Event()
        state.last_use.record(torch.cuda.current_stream())
        result.ready = True
        return result

    @torch.inference_mode()
    def _execute_dspark_batch(self, seqs) -> list[DSparkStepResult]:
        """Execute one homogeneous phase with independent request records.

        Scheduling validates and reserves each owner before dispatch. All
        members become inflight before shared GPU work; a partial batch
        failure cannot be retried using any member's mutated rolling state.
        """
        capacity = getattr(self, "max_running_seqs", getattr(self, "maxd", 1))
        if (
            not 1 < len(seqs) <= capacity
            or len({seq.seq_id for seq in seqs}) != len(seqs)
            or sum(seq.to_compute_token_num for seq in seqs) > self.max_num_batched_tokens
            or len(self._dspark_requests) + sum(
                seq.seq_id not in self._dspark_requests for seq in seqs
            ) > capacity
        ):
            raise ValueError("DSpark batch exceeds its request/token capacity or repeats an owner")
        states = []
        for seq in seqs:
            self._validate_dspark_request(seq)
            states.append(self._get_dspark_request(seq))
        phase = states[0].phase
        if phase not in ("prefill", "bootstrap", "verify", "tail") or any(
            state.phase != phase for state in states
        ):
            raise ValueError("DSpark batches require one homogeneous phase")

        starts = [seq.computed_token_num for seq in seqs]
        counts = [seq.to_compute_token_num for seq in seqs]
        ends = [start + count for start, count in zip(starts, counts)]
        for seq, state, start, count, end in zip(seqs, states, starts, counts, ends):
            if not 0 < count <= self.max_num_batched_tokens or end > self.model_max_length:
                raise ValueError("DSpark batch input exceeds its target context/token limit")
            if phase in ("prefill", "bootstrap"):
                if (
                    end > self.dspark.rotary_emb.cos_sin_cache.shape[0]
                    or state.cache.context_length != start
                    or any(not 0 <= token < self.dspark.config.vocab_size for token in seq[start:end])
                ):
                    raise ValueError("DSpark prompt batch has invalid context, position or token IDs")
                if phase == "prefill":
                    valid = start < end <= seq.prompt_len and len(seq) == seq.prompt_len
                else:
                    valid = count == 1 and start == seq.prompt_len and len(seq) == end
                if not valid:
                    raise ValueError("DSpark prompt batch does not cover contiguous phase inputs")
            elif (
                count != (8 if phase == "verify" else 1)
                or len(seq) != start or state.anchor_pos != start or state.anchor is None
                or state.anchor.numel() != 1
            ):
                raise ValueError("DSpark decode batch needs one unpublished anchor per request")
            if phase == "verify" and (
                state.paged_cache is None or state.pending_hidden is None
                or state.pending_hidden.ndim != 3 or state.pending_hidden.shape[0] != 1
                or not 1 <= state.pending_hidden.shape[1] <= 8
                or state.cache.context_length + state.pending_hidden.shape[1] != start
                or not self._dspark_verify_fits(seq)
            ):
                raise ValueError("DSpark verify batch needs contiguous pending taps and full windows")

        for seq, state, end in zip(seqs, states, ends):
            if not self._reserve_dspark_step(
                seq, state, target_length=end,
                draft_length=state.cache.context_length if phase == "tail" else end,
            ):
                raise RequestCapacityError("Insufficient cache capacity for the DSpark batch")
        results = [
            DSparkStepResult(seq.seq_id, seq._cache_epoch, start, count, phase)
            for seq, start, count in zip(seqs, starts, counts)
        ]
        for state, result in zip(states, results):
            state.inflight = result

        old = [(seq.to_compute_tokens, seq.to_compute_token_num,
                getattr(seq, "_mtp_verify", False)) for seq in seqs]
        try:
            if phase == "verify":
                anchors = torch.cat([state.anchor.reshape(1) for state in states])
                hidden, _ = self.dspark.forward_draft_batched(
                    [state.pending_hidden for state in states], anchors,
                    start_positions=starts, caches=[state.paged_cache for state in states],
                )
                for state in states:
                    state.pending_hidden = None
                proposed, _, _ = self.dspark.forward_head(hidden, anchors)
                proposed_ids = proposed.cpu().tolist()
                for seq, tokens in zip(seqs, proposed_ids):
                    seq.to_compute_tokens, seq._mtp_verify = tokens, True
            elif phase == "tail":
                anchors = torch.cat([state.anchor.reshape(1) for state in states]).cpu().tolist()
                for seq, token in zip(seqs, anchors):
                    seq.to_compute_tokens = [token]

            self.input_data.cal_and_set_input(seqs)
            if self.uses_mrope:
                positions = torch.cat([
                    torch.arange(start, end, device="cpu") for start, end in zip(starts, ends)
                ])
                self.input_data.set_mrope_position(positions.repeat(3, 1))
            self._prepare_attention_metadata(self.input_data)
            if phase == "verify":
                verified = verify_dspark_target_batched(
                    self.model, self.input_data, block_size=self.dspark.block_size,
                    target_layer_ids=self.dspark.config.target_layer_ids,
                )
                for result, verification in zip(results, verified):
                    result.verification = verification
            else:
                if phase == "tail":
                    hidden = self.model(self.input_data)
                    taps = None
                else:
                    hidden, taps = self.model.forward_with_hidden_taps(
                        self.input_data, target_layer_ids=self.dspark.config.target_layer_ids
                    )
                begin, chunks = 0, []
                for seq, state, result, count, end in zip(seqs, states, results, counts, ends):
                    if phase == "prefill":
                        chunks.append(taps[:, begin:begin + count])
                    elif phase == "bootstrap":
                        result.hidden = taps[:, begin:begin + count].clone()
                    if phase != "prefill" or end == seq.prompt_len:
                        result.next_anchor = self.model.logits_from_hidden(
                            hidden[begin + count - 1:begin + count]
                        ).argmax(-1)
                    begin += count
                if phase == "prefill":
                    caches = self.dspark.prefill_paged_batched(
                        chunks, segment=self.memory_manager.dspark_segment,
                        caches=[state.cache for state in states],
                    )
                    for state, cache in zip(states, caches):
                        state.paged_cache = cache
        finally:
            # Restore only host overrides here. Unknown CUDA failures must
            # propagate without event operations in this cleanup.
            for seq, (tokens, count, verify) in zip(seqs, old):
                seq.to_compute_tokens, seq.to_compute_token_num = tokens, count
                seq._mtp_verify = verify
        for state, result in zip(states, results):
            # Events stay request-owned. Re-recording B's next use must never
            # extend A's completion boundary after their shared batch.
            if state.last_use is None:
                state.last_use = torch.cuda.Event()
            state.last_use.record(torch.cuda.current_stream())
            result.ready = True
        return results

    def _complete_dspark_failed_step(self, seq: GenerationSequence) -> None:
        """Cover partial work after a recoverable host validation/capacity error.

        Call only for a known request-level exception. Unknown RuntimeError,
        illegal-address and other CUDA faults must propagate to worker exit.
        If recording this completion boundary itself fails, propagate as well;
        pages must never be recycled based on the previous step's event.
        """
        state = self._dspark_requests.get(seq.seq_id)
        if state is not None:
            # Reservation may already have queued GDN initialization, even
            # when input preparation failed before an inflight result existed.
            if state.last_use is None:
                state.last_use = torch.cuda.Event()
            state.last_use.record(torch.cuda.current_stream())
            if state.inflight is not None:
                state.inflight.ready = True

    def _free_dspark_request(self, seq: GenerationSequence) -> bool:
        """Release a runner-owned request after its GPU users finish.

        Return False when the runner should use its ordinary cache cleanup.
        Epoch and CUDA completion failures propagate before recycling pages.
        """
        state = (
            self._dspark_requests.get(seq.seq_id)
            if getattr(self, "dspark_enabled", False) else None
        )
        if state is None:
            return False
        if state.cache_epoch != seq._cache_epoch:
            raise ValueError("DSpark cleanup must precede request epoch reset")
        if state.inflight is not None and not state.inflight.ready:
            raise RuntimeError("DSpark forward failed before a safe completion event")
        if state.last_use is not None:
            state.last_use.synchronize()
        self.memory_manager.free_dspark_cache(seq, state.cache)
        # Drop the runner-owned tensor and relay references. Independently
        # owned step outputs live until their callers release them. A
        # reused request ID must start with an empty record.
        state.paged_cache = state.pending_hidden = state.anchor = None
        state.anchor_pos = state.inflight = state.last_use = None
        self._dspark_requests.pop(seq.seq_id)
        return True
