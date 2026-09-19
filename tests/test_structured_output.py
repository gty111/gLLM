import copy
import gc
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

import gllm.structured_output as so
from gllm.layers.sampler import Sampler
from gllm.runtime.sequence import GenerationSequence


SCHEMA = {
    "type": "object", "properties": {"answer": {"type": "integer"}},
    "required": ["answer"], "additionalProperties": False,
}


def fmt(schema=SCHEMA, strict=True):
    return {"type": "json_schema", "json_schema": {
        "name": "answer", "schema": schema, "strict": strict,
    }}


@pytest.fixture
def backend(monkeypatch):
    import xgrammar as xgr

    info = xgr.TokenizerInfo(
        [chr(i) for i in range(128)] + ["<eos>", "<think>", "</think>"],
        vocab_type=xgr.VocabType.RAW, stop_token_ids=[128],
    )
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    monkeypatch.setattr(so, "compiler", lambda *args: compiler)
    return so.StructuredSampler(None)


def seq(spec=None, seq_id=0):
    s = GenerationSequence(seq_id, [1], [128], 128,
                           structured_output=spec or so.StructuredOutput(so.normalize_format(fmt())))
    s.computed_token_num = 0
    s.to_compute_token_num = 1
    return s


def emit(backend, seqs, tokens):
    logits = torch.zeros(len(seqs), 131)
    for row, token in enumerate(tokens):
        logits[row, token] = 10
    active = backend.mask(logits, seqs)
    result = logits.argmax(-1)
    assert result.tolist() == tokens
    backend.record(active, result)
    for s in seqs:
        s.computed_token_num += 1
    return result


def test_format_normalization_and_pickle():
    assert so.normalize_format(None) is None
    assert so.normalize_format({"type": "text"}) is None
    assert json.loads(so.normalize_format({"type": "json_object"})) == {"type": "object"}
    chat = fmt()
    response = {"type": "json_schema", **chat["json_schema"]}
    assert so.normalize_format(chat) == so.normalize_format(response)
    spec = so.StructuredOutput(so.normalize_format(chat))
    assert pickle.loads(pickle.dumps(spec)) == spec


@pytest.mark.parametrize("schema", [
    {"type": "bad"}, {"type": "string", "pattern": "a"},
    {"type": "integer", "minimum": 2}, {"$ref": "https://example.com/schema"},
    {"type": "object", "additionalProperties": False, "properties": {"x": {"type": "string"}}},
    {"type": "object"}, {"type": "array", "items": True},
])
def test_bad_or_unsupported_schemas_rejected(schema):
    with pytest.raises(ValueError):
        so.normalize_format(fmt(schema))


def test_batched_different_schemas_and_row_reorder(backend):
    a = seq()
    b = seq(so.StructuredOutput('{"const":{"other":true}}'), 1)
    outputs = ['{"answer":42}', '{"other":true}']
    for index in range(max(map(len, outputs)) + 2):
        rows = [a, b] if index % 2 else [b, a]
        emit(backend, rows, [ord(outputs[s.seq_id][index]) if index < len(outputs[s.seq_id]) else 128 for s in rows])


def test_illegal_token_mask_and_plain_row(backend):
    a, plain = seq(), seq(seq_id=1)
    plain.structured_output = None
    logits = torch.zeros(2, 131)
    active = backend.mask(logits, [a, plain])
    assert torch.isneginf(logits[0, ord("x")])
    assert logits[0, ord("{")] == 0
    assert torch.equal(logits[1], torch.zeros(131))
    assert len(active) == 1


def test_authoritative_token_feedback_not_cpu_history(backend):
    a = seq(so.StructuredOutput('{"type":"string"}'))
    emit(backend, [a], [ord('"')])
    sampled = emit(backend, [a], [ord('x')])
    sampled[0] = ord('y')  # simulate the runner's TP broadcast
    emit(backend, [a], [ord('"')])
    assert backend.states[a].history == [ord('"'), ord('y')]
    assert a.token_ids == [1]  # overlap CPU token history deliberately unchanged


def test_async_feedback_waits_previous_copy_event(backend):
    a = seq()
    emit(backend, [a], [ord('{')])
    class Ready:
        waited = False
        def synchronize(self):
            self.waited = True
    ready = Ready()
    backend.states[a].pending = (torch.tensor(ord('{')), ready)
    emit(backend, [a], [ord('"')])
    assert ready.waited


def test_chunked_prefill_and_preemption(backend):
    a = seq()
    a.raw_prompt_len = a.prompt_len = 3
    active = backend.mask(torch.zeros(1, 131), [a])
    assert not active and not backend.states
    a.computed_token_num = 2
    emit(backend, [a], [ord('{')])
    emit(backend, [a], [ord('"')])
    # Recompute the first output's input; discard the optimistic second output.
    a.computed_token_num = 3
    emit(backend, [a], [ord('"')])
    assert backend.states[a].history == [ord('{')]


@pytest.mark.parametrize("prefilled", [True, False])
def test_reasoning_transition(backend, prefilled):
    a = seq(so.StructuredOutput('{"const":7}', 129, 130, prefilled, not prefilled))
    if not prefilled:
        emit(backend, [a], [129])
    for token in [ord('x'), 130, ord('7'), 128, 128]:
        emit(backend, [a], [token])


def test_request_lifetime_and_id_reuse(backend):
    a = seq()
    emit(backend, [a], [ord('{')])
    del a
    gc.collect()
    assert len(backend.states) == 0
    emit(backend, [seq()], [ord('{')])


def test_sampler_greedy_cannot_bypass_mask(backend):
    a = seq()
    a.top_k = 1
    sampler = Sampler()
    sampler._structured = backend
    logits = torch.zeros(1, 131)
    logits[0, ord('x')] = 100
    logits[0, ord('{')] = 10
    tokens, lp = sampler.forward_gpu(logits, SimpleNamespace(seqs=[a]), True, 20)
    assert tokens.tolist() == [ord('{')]
    assert torch.isfinite(lp[0]).all()
    assert torch.isfinite(lp[1]).all()


def test_prepared_sampling_does_not_wait_or_advance_twice(backend, monkeypatch):
    a = seq()
    a.top_k = 1
    emit(backend, [a], [ord('{')])
    waits = []

    class Ready:
        def synchronize(self):
            waits.append("ready")

    backend.states[a].pending = (torch.tensor(ord('{')), Ready())
    sampler = Sampler()
    sampler._structured = backend
    logits = torch.zeros(1, 131)
    prepared = sampler.prepare_structured(logits, [a])
    assert waits == ["ready"]
    assert backend.states[a].history == [ord('{')]
    monkeypatch.setattr(backend, "prepare", lambda *args: pytest.fail("sampling prepared grammar twice"))
    sampler.forward_gpu(logits, SimpleNamespace(seqs=[a]), structured=prepared)
    assert waits == ["ready"]
    assert backend.states[a].history == [ord('{')]


def test_sync_feedback_does_not_allocate_cuda_stream(backend, monkeypatch):
    a = seq()
    tokens = emit(backend, [a], [ord('{')])
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: pytest.fail("allocated a late stream"))
    backend.stage_feedback(tokens)
    assert backend.states[a].pending.item() == ord('{')


def test_reprefill_feedback_does_not_replace_previous_committed_token(backend):
    a = seq()
    a.raw_prompt_len = a.prompt_len = 3
    a.computed_token_num = 2
    tokens = emit(backend, [a], [ord('{')])
    backend.stage_feedback(tokens)
    previous = backend.states[a].pending
    a.computed_token_num = 0  # intermediate chunk after preemption
    active = backend.mask(torch.zeros(1, 131), [a])
    assert active == []
    backend.record(active, torch.tensor([0]))
    backend.stage_feedback(torch.tensor([0]))
    assert backend.states[a].pending is previous


def test_independent_microbatches_own_distinct_host_masks(backend):
    a = seq(so.StructuredOutput('{"const":7}'))
    b = seq(so.StructuredOutput('{"const":true}'), seq_id=1)
    first = backend.prepare([a], 131, torch.device("cpu"))
    saved = first[1].clone()
    second = backend.prepare([b], 131, torch.device("cpu"))
    assert first[1].data_ptr() != second[1].data_ptr()
    assert torch.equal(first[1], saved)
    logits = torch.zeros(1, 131)
    backend.apply(logits, first)
    assert logits[0, ord('7')] == 0
    assert torch.isneginf(logits[0, ord('t')])


def test_reasoning_cannot_finish_before_answer(backend):
    a = seq(so.StructuredOutput('{"const":7}', 129, 130, True, False))
    logits = torch.zeros(1, 131)
    backend.mask(logits, [a])
    assert torch.isneginf(logits[0, 128])
    assert logits[0, 130] == 0


def test_follower_spec_roundtrip():
    from gllm.scheduling.distributed import DriverPayloadBuilder, FollowerSeq

    a = seq()
    reg = DriverPayloadBuilder().build([a], []).registers[0]
    follower = FollowerSeq(pickle.loads(pickle.dumps(reg)))
    assert follower.structured_output == a.structured_output
    import weakref
    assert weakref.ref(follower)() is follower


@pytest.mark.parametrize("schema,text", [
    ({"type": "object", "properties": {"x": {"type": ["string", "null"]}},
      "required": ["x"], "additionalProperties": False}, '{"x":null}'),
    ({"type": "object", "properties": {"x": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}},
      "required": ["x"], "additionalProperties": False}, '{"x":true}'),
    ({"type": "object", "properties": {"x": {"$ref": "#/$defs/value"}},
      "$defs": {"value": {"type": "integer", "enum": [2, 4]}},
      "required": ["x"], "additionalProperties": False}, '{"x":4}'),
    ({"type": "object", "properties": {"x": {"type": "integer"}}}, '{}'),
    ({"type": "object", "properties": {"x": {"type": "integer"}}}, '{"extra":true}'),
])
def test_schema_subset_is_actually_enforced(backend, schema, text):
    import jsonschema

    a = seq(so.StructuredOutput(so.normalize_format(fmt(schema, strict=False))))
    for token in [*map(ord, text), 128]:
        emit(backend, [a], [token])
    jsonschema.validate(json.loads(text), schema)


def test_no_grammar_gpu_work_for_ordinary_requests(monkeypatch):
    sampler = Sampler()
    a = seq()
    a.structured_output = None
    a.top_k = 1
    monkeypatch.setattr(so.StructuredSampler, "__init__", lambda *a: pytest.fail("ordinary request initialized grammar"))
    result = sampler.forward_gpu(torch.tensor([[0., 1.]]), SimpleNamespace(seqs=[a]))
    assert result.tolist() == [1]


def test_api_rejects_incompatible_options():
    from gllm.entrypoints.api_server import _validate_output_format

    assert _validate_output_format(fmt(), "response_format") is None
    for kwargs in ({"tools": [{"type": "function"}]}, {"ignore_eos": True}):
        assert _validate_output_format(fmt(), "response_format", **kwargs).status_code == 400


@pytest.mark.parametrize("kind", ["json_schema", "json_object"])
def test_both_api_frontends_validate_structured_formats(monkeypatch, kind):
    from gllm.entrypoints import api_server
    from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest

    monkeypatch.setattr(api_server, "llm", SimpleNamespace(model_path="test"))
    chat_format = fmt() if kind == "json_schema" else {"type": kind}
    response_format = ({"type": kind, **fmt()["json_schema"]}
                       if kind == "json_schema" else {"type": kind})
    chat = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "JSON"}],
                                 response_format=chat_format)
    response = ResponseRequest(model="test", input="JSON", text={"format": response_format})
    assert api_server._validate_chat_capabilities(chat) is None
    assert api_server._validate_response_capabilities(response) is None
    assert so.normalize_format(chat.response_format) == so.normalize_format(response.text["format"])


def test_responses_frontend_rejects_unsupported_schema(monkeypatch):
    from gllm.entrypoints import api_server
    from gllm.entrypoints.protocol import ResponseRequest

    monkeypatch.setattr(api_server, "llm", SimpleNamespace(model_path="test"))
    request = ResponseRequest(model="test", input="JSON", text={"format": {
        "type": "json_schema", "name": "answer", "schema": {"type": "integer", "minimum": 1}}})
    error = api_server._validate_response_capabilities(request)
    assert error.status_code == 400
    assert json.loads(error.body)["error"]["param"] == "text.format"


def test_mtp_sampling_supports_constraints_but_preserves_optional_features():
    from gllm.runtime.model_runner import ModelRunner

    a = seq()
    assert ModelRunner.mtp_sampling_compatible([a])
    a.logprobs_enabled = True
    assert not ModelRunner.mtp_sampling_compatible([a])
    a.logprobs_enabled = False
    a.repetition_penalty = 1.1
    assert not ModelRunner.mtp_sampling_compatible([a])


def test_speculative_rejected_suffix_does_not_advance_grammar(backend):
    a = seq(so.StructuredOutput('{"const":{"x":1}}'))
    candidates = [[ord('{'), ord('x'), ord('!')]]
    logits = torch.zeros(3, 131)
    active = backend.mask_speculative(logits, [a], [[1]], candidates)
    assert torch.isneginf(logits[0, ord('x')])
    assert logits[0, ord('"')] == 0
    assert torch.isfinite(logits[1:]).all()  # unreachable, no all-inf softmax
    assert backend.states[a].history == []
    backend.commit_speculative(active, [[ord('{')]], [ord('"')])
    assert backend.states[a].history == [ord('{')]
    # Next relay consumes the bonus once and sees only committed history.
    logits = torch.zeros(3, 131)
    active = backend.mask_speculative(
        logits, [a], [[1, ord('{')]], [[ord('"'), ord('x'), ord('"')]])
    assert logits[0, ord('x')] == 0
    assert logits[2, ord(':')] == 0
    backend.commit_speculative(active, [[ord('"'), ord('x'), ord('"')]], [ord(':')])
    assert backend.states[a].history == list(map(ord, '{"x"'))
    # Relay-to-plain handoff stages ':' as the next uncached token.
    a.computed_token_num = 5
    logits = torch.zeros(1, 131)
    backend.mask(logits, [a])
    assert backend.states[a].history == list(map(ord, '{"x":'))
    assert logits[0, ord('1')] == 0


def test_speculative_thinking_boundary_eos_and_preemption(backend):
    a = seq(so.StructuredOutput('{"const":7}', 129, 130, True))
    logits = torch.zeros(3, 131)
    active = backend.mask_speculative(logits, [a], [[1]], [[ord('a'), 130, ord('7')]])
    assert torch.isneginf(logits[0, 128])
    assert logits[1, ord('7')] == 0 and torch.isneginf(logits[1, ord('x')])
    assert logits[2, 128] == 0
    backend.commit_speculative(active, [[ord('a'), 130, ord('7')]], [128])
    # A preempted context can rewind beyond the end-thinking boundary.
    backend.mask_speculative(torch.zeros(1, 131), [a], [[1]], [[ord('b')]])
    assert backend.states[a].history == [] and backend.states[a].thinking


def test_speculative_mixed_rows_and_prior_plain_pending(backend):
    a, plain = seq(so.StructuredOutput('{"const":12}')), seq(seq_id=1)
    plain.structured_output = None
    emit(backend, [a], [ord('1')])
    logits = torch.zeros(4, 131)
    active = backend.mask_speculative(logits, [plain, a], [[1], [1, ord('1')]],
                                      [[0, 0], [ord('2'), 128]])
    assert torch.isfinite(logits[:2]).all()
    assert logits[2, 128] == 0
    assert torch.isneginf(logits[3, ord('x')])
    backend.commit_speculative(active, [[0], [ord('2'), 128]], [0, 128])
    assert backend.states[a].history == [ord('1'), ord('2'), 128]


def test_structured_mtp_keeps_async_greedy_acceptance():
    from gllm.workers.overlap import _MtpBatchPlan

    assert _MtpBatchPlan(speculate=True, greedy=True).use_async
    assert not _MtpBatchPlan(speculate=True, greedy=False).use_async


def test_grammar_valid_but_target_rejected_draft_does_not_leak(backend):
    a = seq(so.StructuredOutput('{"enum":["abc","adc"]}'))
    logits = torch.zeros(3, 131)
    active = backend.mask_speculative(logits, [a], [[1]], [list(map(ord, '"ad'))])
    assert logits[1, ord('b')] == logits[1, ord('d')] == 0
    # Target chooses b, not grammar-valid d. Only quote+a were accepted.
    backend.commit_speculative(active, [list(map(ord, '"a'))], [ord('b')])
    logits = torch.zeros(1, 131)
    backend.mask_speculative(logits, [a], [[1] + list(map(ord, '"a'))], [[ord('b')]])
    assert logits[0, ord('c')] == 0
    assert torch.isneginf(logits[0, ord('d')])
    assert backend.states[a].history == list(map(ord, '"a'))


def test_mixed_structured_prefill_selects_async_mtp_plan():
    from gllm.workers.overlap import OverlapWorker

    worker = OverlapWorker.__new__(OverlapWorker)
    worker._dp = False
    worker.model_runner = SimpleNamespace(mtp_enabled=True, mtp_begin_iter=bool)
    decode, prefill = seq(seq_id=1), seq(seq_id=2)
    decode.structured_output = None
    decode.computed_token_num = 1
    decode.top_k = prefill.top_k = 1
    decode.temperature = prefill.temperature = 0
    worker._prefetched_input = SimpleNamespace(seqs=[decode, prefill], num_decodes=1, num_prefills=1)
    plan = worker._plan_mtp_batch()
    assert plan.speculate and plan.use_async
    prefill.logprobs_enabled = True
    assert not worker._plan_mtp_batch().speculate


def test_sparse_mtp_masked_padding_is_not_a_probability_tie_overflow():
    from gllm.runtime.model_runner import ModelRunner

    runner = SimpleNamespace(_mtp_tie_overflow=torch.tensor(0))
    logits = torch.full((1, 16), float('-inf'))
    logits[0, 3] = 4
    probs, ids = ModelRunner._mtp_sparse_probs(
        runner, logits, torch.ones(1, 1), torch.tensor([4]), torch.ones(1), 8)
    assert runner._mtp_tie_overflow.item() == 0
    assert probs.sum().item() == 1
    assert ids[0, probs.argmax()].item() == 3
    # Finite ties really can spill beyond the sparse window.
    ModelRunner._mtp_sparse_probs(
        runner, torch.zeros(1, 16), torch.ones(1, 1), torch.tensor([4]), torch.ones(1), 8)
    assert runner._mtp_tie_overflow.item() == 1


def completion(grid, seq_ids=(0,)):
    from gllm.speculative.async_state import MtpAsyncCompletion

    calls = []
    owner = SimpleNamespace(_host=[torch.tensor(grid)], _release=lambda slot: calls.append("release"))
    ready = SimpleNamespace(synchronize=lambda: calls.append("wait"))
    return MtpAsyncCompletion(owner, 0, seq_ids, len(grid), ready), calls


def test_acceptance_read_does_not_release_completion_ring():
    pending, calls = completion([[2, 11, 12, -1]])
    assert pending.read() == ([2], [[11, 12]])
    assert calls == ["wait"]
    assert pending.collect() == ([2], [[11, 12]])
    assert calls == ["wait", "release"]


def test_async_mask_uses_acceptance_not_scheduler_placeholders(backend):
    a = seq(so.StructuredOutput('{"enum":["abc","adc"]}'))
    active = backend.mask_speculative(torch.zeros(3, 131), [a], [[1]], [list(map(ord, '"ad'))])
    pending, calls = completion([[2, ord('"'), ord('a'), -1]])
    backend.record_speculative(active, pending)
    a.token_ids.extend([-1, -1, -1])  # optimistic, not authoritative
    logits = torch.zeros(2, 131)
    next_active = backend.mask_speculative(logits, [a], None, [[ord('b'), ord('c')]], positions=[3])
    assert backend.states[a].history == list(map(ord, '"a'))
    assert logits[0, ord('c')] == 0 and torch.isneginf(logits[0, ord('d')])
    assert calls == ["wait"]  # grammar didn't retire the scheduler's batch
    successor, _ = completion([[2, ord('b'), ord('c'), -1]])
    backend.record_speculative(next_active, successor)
    backend.finish_speculative([a], pending)  # collecting N must not commit N+1
    assert backend.states[a].history == list(map(ord, '"a'))
    pending.collect()
    backend.finish_speculative([a], successor)
    backend.finish_speculative([a], successor)  # idempotent across drain/barrier
    assert backend.states[a].history == list(map(ord, '"abc'))
    backend.stage_relay(a, ord('"'))
    a.computed_token_num = 5  # next ordinary sample consumes the relay quote
    logits = torch.zeros(1, 131)
    backend.mask(logits, [a])
    assert logits[0, 128] == 0
    assert backend.states[a].history == list(map(ord, '"abc"'))


def test_async_completed_prefill_does_not_consume_relay_x1_twice(backend):
    a = seq(so.StructuredOutput('{"const":12}'))
    emit(backend, [a], [ord('1')])
    logits = torch.zeros(2, 131)
    active = backend.mask_speculative(logits, [a], None, [[ord('1'), ord('2')]], positions=[1])
    assert backend.states[a].history == []
    pending, _ = completion([[2, ord('1'), ord('2'), -1]])
    backend.record_speculative(active, pending)
    backend.finish_speculative([a], pending)
    assert backend.states[a].history == [ord('1'), ord('2')]


def test_async_completion_remap_keeps_per_request_row_identity(backend):
    a, b = seq(so.StructuredOutput('{"const":12}')), seq(so.StructuredOutput('{"const":34}'), 1)
    active = backend.mask_speculative(torch.zeros(4, 131), [a, b], [[1], [1]],
                                      [[ord('1'), ord('2')], [ord('3'), ord('4')]])
    pending, calls = completion([[1, ord('1'), -1], [1, ord('3'), -1]], seq_ids=(0, 1))
    backend.record_speculative(active, pending)
    logits = torch.zeros(2, 131)
    backend.mask_speculative(logits, [b, a], None, [[ord('4')], [ord('2')]], positions=[2, 2])
    assert backend.states[a].history == [ord('1')]
    assert backend.states[b].history == [ord('3')]
    assert calls == ["wait"]
    assert (logits[:, 128] == 0).all()


@pytest.mark.parametrize("follower", [False, True])
@pytest.mark.parametrize("chunk_end", [1, 4])
@pytest.mark.parametrize("thinking", [False, True])
def test_reprefill_recovers_committed_history_after_chunk_copy(backend, follower, chunk_end, thinking):
    from gllm.scheduling.distributed import DriverPayloadBuilder, FollowerSeqStore

    spec = so.StructuredOutput(so.normalize_format(fmt()), 129, 130, thinking)
    s = seq(spec)
    builder, store = DriverPayloadBuilder(), FollowerSeqStore()
    mirror = store.apply_payload(builder.build([s], []))[0]
    prefix = ([ord('x'), 130] if thinking else []) + list(map(ord, '{"answer":'))
    for token in prefix:
        emit(backend, [s], [token])
        s.token_ids.append(token)
    s.preempt()
    s.to_compute_token_num = chunk_end
    if follower:
        payload = pickle.loads(pickle.dumps(builder.build([s], [])))
        assert payload.updates[0].structured_output_history is None
        current = store.apply_payload(payload)[0]
        assert current is mirror and current.token_ids is None
        assert current.prompt_len == s.prompt_len
    else:
        current = s
    assert backend.mask(torch.zeros(1, 131), [current]) == []

    # schedule_prefill_batch deep-copies each unfinished chunk. The resumed
    # driver object has no WeakKeyDictionary entry; the follower keeps its
    # object identity but receives a changed prefill boundary and history.
    resumed = copy.deepcopy(s)
    resumed.computed_token_num = chunk_end
    resumed.to_compute_token_num = resumed.prompt_len - chunk_end
    if follower:
        payload = builder.build([resumed], [])
        snapshot = payload.updates[0].structured_output_history
        assert snapshot == prefix
        assert snapshot is not resumed.token_ids
        current = store.apply_payload(pickle.loads(pickle.dumps(payload)))[0]
    else:
        current = resumed
    logits = torch.zeros(1, 131)
    active = backend.mask(logits, [current])
    assert backend.states[current].history == prefix
    assert not backend.states[current].thinking
    assert logits[0, ord('4')] == 0
    assert torch.isneginf(logits[0, ord('{')])
    backend.record(active, torch.tensor([ord('4')]))
    backend.stage_feedback(torch.tensor([ord('4')]))

    # The first ordinary decode must consume the new sample exactly once.
    resumed.computed_token_num = resumed.prompt_len
    resumed.to_compute_token_num = 1
    resumed.token_ids.append(ord('4'))
    if follower:
        payload = builder.build([resumed], [])
        assert payload.updates[0].structured_output_history is None
        current = store.apply_payload(payload)[0]
    else:
        current = resumed
    emit(backend, [current], [ord('2')])
    emit(backend, [current], [ord('}')])
    emit(backend, [current], [128])


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"x": {"$ref": "#/$defs/X", "enum": [1]}},
     "required": ["x"], "additionalProperties": False, "$defs": {"X": {"type": "integer"}}},
    {"type": "object", "properties": {"x": {"type": "string", "enum": [1, "x"]}},
     "required": ["x"], "additionalProperties": False},
    {"type": "object", "properties": {"x": {"type": "integer", "const": "x"}},
     "required": ["x"], "additionalProperties": False},
    {"type": "object", "properties": {"x": {"type": "integer", "anyOf": [{"const": 1}] }},
     "required": ["x"], "additionalProperties": False},
    {"type": "object", "properties": {"x": {"type": "integer"}},
     "required": ["x"], "additionalProperties": False, "enum": [{}, {"x": 1}]},
])
@pytest.mark.parametrize("strict", [False, True])
def test_unenforced_schema_intersections_return_400(schema, strict):
    from gllm.entrypoints.api_server import _validate_output_format

    chat = fmt(schema, strict=strict)
    response = {"type": "json_schema", **chat["json_schema"]}
    for value, param in ((chat, "response_format"), (response, "text.format")):
        error = _validate_output_format(value, param)
        assert error.status_code == 400
        assert json.loads(error.body)["error"]["code"] == "invalid_output_format"


def test_required_properties_without_schema_are_rejected():
    with pytest.raises(ValueError, match="Required properties"):
        so.normalize_format(fmt({"type": "object", "required": ["x"]}, strict=False))


def test_reprefill_same_object_discards_uncommitted_feedback(backend):
    s = seq()
    prefix = list(map(ord, '{"answer":1'))
    for token in prefix:
        emit(backend, [s], [token])
        s.token_ids.append(token)
    emit(backend, [s], [ord('2')])  # overlap sample not committed by scheduler
    previous_state = backend.states[s]
    s.preempt()
    s.to_compute_token_num = s.prompt_len
    emit(backend, [s], [ord('}')])
    assert backend.states[s] is previous_state
    assert previous_state.history == prefix


@pytest.mark.parametrize("value", [
    {"type": "integer", "enum": [1, 2]},
    {"type": "integer", "const": 1, "enum": [1, 2]},
    {"$ref": "#/$defs/X", "description": "An integer"},
    {"anyOf": [{"const": 1}, {"const": 2}], "title": "Choice"},
])
def test_supported_schema_combinations_still_enforced(backend, value):
    schema = {"type": "object", "properties": {"x": value}, "required": ["x"],
              "additionalProperties": False, "$defs": {"X": {"type": "integer", "enum": [1, 2]}}}
    s = seq(so.StructuredOutput(so.normalize_format(fmt(schema))))
    for token in [*map(ord, '{"x":1}'), 128]:
        emit(backend, [s], [token])
