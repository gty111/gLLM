"""Opt-in, bounded host observations for synchronous DSpark drafting.

The runner owns this observer. It never selects a verify width or changes p,
q, acceptance, publication, or scheduler state. Graph/overlap collection uses
different ownership rules and is deliberately not supported here.
"""
from collections import deque
from dataclasses import dataclass
import math
import time
import weakref


@dataclass(frozen=True)
class DSparkConfidenceRecord:
    request_identity: tuple
    cache_epoch: int
    round_id: int
    anchor_pos: int
    anchor_token: int
    candidates: tuple
    confidence_logits: tuple
    accepted_drafts: int
    bonus_token: int
    sampling_signature: tuple
    previous_round_id: int | None
    started_ns: int
    completed_ns: int
    remaining_tokens: int
    finish_tokens: tuple
    ignore_eos: bool
    finite: bool

    @property
    def candidate_indices(self) -> tuple:
        return tuple(range(len(self.candidates)))

    @property
    def prefix_labels(self) -> tuple:
        return tuple(int(j < self.accepted_drafts) for j in self.candidate_indices)

    @property
    def conditional_labels(self) -> tuple:
        return tuple(
            int(j < self.accepted_drafts) if j <= self.accepted_drafts else None
            for j in self.candidate_indices
        )


@dataclass
class _RequestState:
    owner: object
    instance: int
    round_id: int = 0
    lifecycle: int = 0
    active: bool = True
    latest: DSparkConfidenceRecord | None = None


@dataclass
class _ObservationRow:
    state: _RequestState
    lifecycle: int
    request_identity: tuple
    cache_epoch: int
    round_id: int
    anchor_pos: int
    anchor_token: int
    sampling_signature: tuple
    previous_round_id: int | None
    started_ns: int
    remaining_tokens: int
    finish_tokens: tuple
    ignore_eos: bool
    consumed: bool = False


class DSparkConfidenceObserver:
    """One worker-lifetime observation sink; disabled unless explicitly attached.

    run_id must identify this worker lifetime. Records contain host values,
    never GenerationSequence or GPU tensor owners. Weak references retain the
    round identity across cache release/recompute without retaining requests.
    """

    def __init__(self, run_id: str, *, capacity: int = 1024) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must identify the worker lifetime")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.run_id = run_id
        self.capacity = capacity
        self.records = deque()
        self.dropped_records = 0
        self.stale_records = 0
        self.invalid_records = 0
        self._states = {}
        self._next_instance = 0

    @staticmethod
    def check_runner(runner, *, mixed=False, gpu_anchor=False) -> None:
        config = runner.config
        if (
            not getattr(runner, "dspark_enabled", False)
            or getattr(runner, "_dspark_drafter", None) is None
            or config.overlap_scheduling
            or not config.disable_cuda_graph
            or config.piecewise_cuda_graph is not False
            or getattr(runner, "_mtp_async_publish", False)
            or getattr(runner, "_piecewise_runner", None) is not None
            or getattr(runner, "_mtp_verify_graph", False)
            or getattr(runner, "_dspark_draft_graphs", None)
            or getattr(runner, "_dspark_sampled_graphs", None)
            or mixed or gpu_anchor
        ):
            raise ValueError("DSpark confidence observation requires strict synchronous Eager")

    def _state(self, seq):
        key = id(seq)
        state = self._states.get(key)
        if state is None or state.owner() is not seq:
            def expired(owner):
                current = self._states.get(key)
                if current is not None and current.owner is owner:
                    self._states.pop(key, None)
            self._next_instance += 1
            state = _RequestState(weakref.ref(seq, expired), self._next_instance)
            self._states[key] = state
        return state

    @property
    def active_requests(self) -> int:
        return sum(state.active for state in self._states.values())

    def forget(self, seq) -> None:
        """Invalidate history/tickets on release, including preemption.

        Release alone cannot distinguish retirement from recomputation; keep
        only the weak identity and round counter until this object dies.
        """
        state = self._states.get(id(seq))
        if state is not None and state.owner() is seq:
            state.latest = None
            state.active = False
            state.lifecycle += 1

    def begin(self, seqs, positions, anchors, k, *, sampled, sparse):
        if len(seqs) != len(positions) or len(seqs) != len(anchors) or k < 1:
            raise ValueError("observation rows must match the actual draft batch")
        temperature_gate = any(
            seq.temperature > 1e-5 and abs(seq.temperature - 1.0) > 1e-5 for seq in seqs
        )
        started = time.perf_counter_ns()
        rows = []
        for seq, position, anchor in zip(seqs, positions, anchors):
            state = self._state(seq)
            epoch = seq._cache_epoch
            signature = (
                seq.temperature, seq.top_k, seq.top_p, seq.repetition_penalty,
                sampled, sparse, temperature_gate, k,
            )
            previous = state.latest
            continuous = (
                state.active and previous is not None
                and previous.finite and previous.cache_epoch == epoch
                and previous.sampling_signature == signature
                and position == previous.anchor_pos + 1 + previous.accepted_drafts
            )
            state.round_id += 1
            state.active = True
            rows.append(_ObservationRow(
                state, state.lifecycle, (self.run_id, seq.seq_id, state.instance),
                epoch, state.round_id, int(position), int(anchor), signature,
                previous.round_id if continuous else None, started,
                max(0, seq.output_len - (len(seq.token_ids) - seq.raw_prompt_len)),
                tuple(seq.finish_tokens), seq.ignore_eos,
            ))
        return rows

    def complete(self, rows, scores, accepted, candidates, bonus) -> None:
        """Pair raw acceptance with this launch before EOS/budget publication.

        The caller already owns synchronous host acceptance. No tensor reads,
        RNG operations, scheduling, or output truncation occur in this sink.
        """
        if not all(len(values) == len(rows) for values in (scores, accepted, candidates, bonus)):
            raise ValueError("confidence completion batch mismatch")
        if any(row.consumed for row in rows):
            raise RuntimeError("confidence observation completed twice")
        for row, values, count, proposed in zip(rows, scores, accepted, candidates):
            k = row.sampling_signature[-1]
            if len(values) != k or len(proposed) != k or not 0 <= count <= k:
                raise ValueError("confidence completion position/acceptance mismatch")
        completed = time.perf_counter_ns()
        for row, values, count, proposed, token in zip(rows, scores, accepted, candidates, bonus):
            row.consumed = True
            state, seq = row.state, row.state.owner()
            if (seq is None or not state.active or state.lifecycle != row.lifecycle
                    or seq._cache_epoch != row.cache_epoch):
                self.stale_records += 1
                continue
            raw = tuple(float(value) for value in values)
            finite = all(math.isfinite(value) for value in raw)
            record = DSparkConfidenceRecord(
                row.request_identity, row.cache_epoch, row.round_id,
                row.anchor_pos, row.anchor_token, tuple(int(x) for x in proposed),
                raw, int(count), int(token), row.sampling_signature,
                row.previous_round_id, row.started_ns, completed,
                row.remaining_tokens, row.finish_tokens, row.ignore_eos, finite,
            )
            if not finite:
                self.invalid_records += 1
                state.latest = None
            elif state.latest is None or record.round_id > state.latest.round_id:
                state.latest = record
            if len(self.records) == self.capacity:
                self.records.popleft()
                self.dropped_records += 1
            self.records.append(record)

    def drain(self) -> list:
        records = list(self.records)
        self.records.clear()
        return records
