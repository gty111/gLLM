"""Output limits use actual prompt lengths and are shared by all entrypoints."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gllm.engine.llm import LLM
from gllm.entrypoints import api_server
from gllm.entrypoints.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponseRequest,
)
from gllm.runtime.id_allocator import IDAllocator
from gllm.utils import StreamOutput, get_finish_reason


def make_engine(context=262144):
    engine = LLM.__new__(LLM)
    engine.model_max_length = context
    engine.model_path = "test"
    engine.generation_config = SimpleNamespace(
        temperature=None, top_p=None, repetition_penalty=None,
    )
    engine.finish_tokens = [99]
    engine.id_allocator = IDAllocator()
    return engine


@pytest.mark.parametrize("prompt_len,requested,expected", [
    (180000, None, 82144),
    (262143, None, 1),
    (180000, 16384, 16384),
    (180000, 82144, 82144),
])
def test_engine_resolves_budget(prompt_len, requested, expected):
    engine = make_engine()
    tokens = [1] * prompt_len
    assert engine.check_seq_length(tokens, requested)
    seq = engine.allocate_seq(tokens, requested)
    assert seq.output_len == expected
    assert seq.requested_output_len == requested


@pytest.mark.parametrize("prompt_len,requested", [
    (262144, None), (262145, None), (262144, 1),
    (180000, 82145), (1, 0), (1, -1),
])
def test_invalid_budget_is_rejected_before_id_allocation(prompt_len, requested):
    engine = make_engine()
    tokens = [1] * prompt_len
    assert not engine.check_seq_length(tokens, requested)
    with pytest.raises(ValueError):
        engine.allocate_seq(tokens, requested)
    assert engine.id_allocator.get_num_used_ids() == 0


def test_default_budget_survives_cache_progress_and_preemption():
    engine = make_engine(context=8)
    seq = engine.allocate_seq([1, 2, 3, 4])
    seq.computed_token_num = 3  # Prefix hits must not enlarge the budget.
    assert seq.output_len == 4
    seq.append(10)
    seq.preempt()
    assert seq.raw_prompt_len == 4
    assert seq.prompt_len == 5
    assert seq.output_len == 4
    seq.computed_token_num = 5
    seq.token_ids.extend([11, 12])
    assert not seq.is_finish
    seq.append(13)
    assert seq.is_finish
    assert get_finish_reason(seq) == "length"


def test_default_budget_still_stops_at_eos():
    seq = make_engine(context=8192).allocate_seq([1, 2])
    seq.computed_token_num = 2
    seq.append(99)
    assert seq.is_finish
    assert get_finish_reason(seq) == "stop"


@pytest.mark.parametrize("requested,expected", [(None, 2), (1, 1), (2, 2)])
def test_disagg_budget_uses_expanded_image_prompt(monkeypatch, requested, expected):
    from gllm.disagg.lm_manager import DisaggCoordinator, DisaggEvents, _PendingSeq
    from gllm.layers.rotary_embedding import MRotaryEmbedding
    from gllm.runtime.model_runner import ModelRunner

    seq = make_engine(context=8).allocate_seq([1, 126, 2], requested)
    pending = _PendingSeq(seq, [SimpleNamespace(
        item_idx=0, modality="image",
        meta=SimpleNamespace(num_tokens=4, grid_thw=(1, 2, 2), content_hash="image"),
    )])
    coordinator = DisaggCoordinator.__new__(DisaggCoordinator)
    coordinator.image_token_id = 126
    coordinator.video_token_id = 127
    coordinator.mr = SimpleNamespace(model=SimpleNamespace(
        config=None, get_mm_placeholder_token_ids=lambda: [126, 127],
    ))
    monkeypatch.setattr(ModelRunner, "_splice_mm_pad_ids", lambda *args: args[0])
    monkeypatch.setattr(MRotaryEmbedding, "get_input_positions", lambda **kwargs: (None, 0))
    events = DisaggEvents()
    coordinator._admit_meta_complete(pending, events)
    assert seq.token_ids == [1, 126, 126, 126, 126, 2]
    assert seq.raw_prompt_len == 6
    assert seq.output_len == expected
    assert len(events.admits) == 1


@pytest.mark.parametrize("requested,image_tokens", [(None, 6), (None, 7), (3, 4)])
def test_disagg_rejects_budget_that_no_longer_fits(requested, image_tokens):
    from gllm.disagg.lm_manager import DisaggCoordinator, DisaggEvents, _PendingSeq

    seq = make_engine(context=8).allocate_seq([1, 126, 2], requested)
    pending = _PendingSeq(seq, [SimpleNamespace(meta=SimpleNamespace(num_tokens=image_tokens))])
    coordinator = DisaggCoordinator.__new__(DisaggCoordinator)
    coordinator.image_token_id = 126
    coordinator.video_token_id = 127
    events = DisaggEvents()
    with pytest.raises(ValueError):
        coordinator._admit_meta_complete(pending, events)
    assert not events.admits
    assert not pending.admitted


class FakeStream:
    def __init__(self, seq):
        self.seq = seq

    async def __aiter__(self):
        self.seq.append(99)
        yield StreamOutput("hello")


REQUESTS = {
    "chat": (ChatCompletionRequest, api_server.create_chat_completion,
             {"messages": [{"role": "user", "content": "hi"}]}, "max_completion_tokens"),
    "responses": (ResponseRequest, api_server.create_response,
                  {"input": "hi"}, "max_output_tokens"),
    "completions": (CompletionRequest, api_server.create_completion,
                    {"prompt": "hi"}, "max_tokens"),
}


@pytest.mark.parametrize("endpoint", REQUESTS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("limit", ["omitted", None, 7])
def test_api_budget_reaches_engine(monkeypatch, endpoint, streaming, limit):
    engine = make_engine(context=8192)
    engine.model_runner = SimpleNamespace(
        encode=lambda *args, **kwargs: [1] * 100,
        extract_modify_mm=lambda messages: None,
    )
    allocated = []

    async def add_requests(raw, *args, **kwargs):
        kwargs.pop("dp_index", None)
        seq = engine.allocate_seq(*args, **kwargs)
        allocated.append(seq)
        return FakeStream(seq)

    engine.add_requests_async = add_requests
    monkeypatch.setattr(api_server, "llm", engine)
    monkeypatch.setattr(api_server, "tool_parser", None)
    cls, handler, fields, limit_field = REQUESTS[endpoint]
    fields = dict(fields)
    if limit != "omitted":
        fields[limit_field] = limit
    request = cls(model="test", stream=streaming, **fields)
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    async def run():
        result = await handler(request, raw)
        assert result.status_code == 200
        if streaming:
            chunks = [chunk async for chunk in result.body_iterator]
            assert chunks
        else:
            assert json.loads(result.body)

    asyncio.run(run())
    expected = 7 if limit == 7 else 8092
    if endpoint == "completions" and limit == "omitted":
        expected = 16  # Preserve the legacy Completions API default.
    assert len(allocated) == 1
    assert allocated[0].output_len == expected


@pytest.mark.parametrize("endpoint", REQUESTS)
@pytest.mark.parametrize("prompt_len,limit", [(8, None), (9, None), (7, 2)])
def test_api_returns_context_error_without_enqueuing(monkeypatch, endpoint, prompt_len, limit):
    engine = make_engine(context=8)
    engine.model_runner = SimpleNamespace(
        encode=lambda *args, **kwargs: [1] * prompt_len,
        extract_modify_mm=lambda messages: None,
    )
    # No add_requests_async: a regression that enqueues must fail the test.
    monkeypatch.setattr(api_server, "llm", engine)
    cls, handler, fields, limit_field = REQUESTS[endpoint]
    request = cls(model="test", **fields, **{limit_field: limit})
    response = asyncio.run(handler(request, SimpleNamespace()))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["code"] == "context_length_exceeded"


@pytest.mark.parametrize("endpoint", REQUESTS)
@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_api_limits_are_invalid(endpoint, limit):
    cls, _, fields, limit_field = REQUESTS[endpoint]
    with pytest.raises(ValidationError):
        cls(model="test", **fields, **{limit_field: limit})


@pytest.mark.parametrize("fields,expected", [
    ({"max_tokens": 9}, 9),
    ({"max_tokens": 9, "max_completion_tokens": 7}, 7),
])
def test_legacy_chat_limit_and_modern_precedence(monkeypatch, fields, expected):
    captured = []
    engine = make_engine(context=8192)
    engine.model_runner = SimpleNamespace(
        encode=lambda *args, **kwargs: [1, 2],
        extract_modify_mm=lambda messages: None,
    )

    async def add_requests(raw, *args, **kwargs):
        kwargs.pop("dp_index", None)
        seq = engine.allocate_seq(*args, **kwargs)
        captured.append(seq.output_len)
        return FakeStream(seq)

    engine.add_requests_async = add_requests
    monkeypatch.setattr(api_server, "llm", engine)
    monkeypatch.setattr(api_server, "tool_parser", None)
    request = ChatCompletionRequest(model="test", messages=[], **fields)
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert asyncio.run(api_server.create_chat_completion(request, raw)).status_code == 200
    assert captured == [expected]
