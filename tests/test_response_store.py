"""Server-side Responses history and bounded store tests."""

import asyncio
import json
import time
from types import SimpleNamespace

from gllm.entrypoints import api_server
from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.response_store import ResponseStore


def test_store_put_get_roundtrip():
    store = ResponseStore()
    record = {
        "response": {"id": "resp_1", "output": []},
        "input_items": [{"type": "message", "role": "user", "content": "hi"}],
        "model": "test",
    }
    store.put("resp_1", record)
    assert store.get("resp_1") == record
    store.delete("resp_1")
    assert store.get("resp_1") is None


def test_store_evicts_oldest():
    store = ResponseStore(max_entries=2)
    store.put("a", {"response": {"id": "a"}, "input_items": [], "model": "test"})
    store.put("b", {"response": {"id": "b"}, "input_items": [], "model": "test"})
    assert store.get("a") is not None  # Refresh its LRU position.
    store.put("c", {"response": {"id": "c"}, "input_items": [], "model": "test"})
    assert store.get("b") is None
    assert store.get("a") is not None


def test_store_expires():
    store = ResponseStore(ttl_seconds=1)
    store.put("a", {"response": {"id": "a"}, "input_items": [], "model": "test"})
    time.sleep(1.01)
    assert store.get("a") is None


def _mock_endpoint(monkeypatch, *, output=None):
    seen = {}
    runner = SimpleNamespace(
        extract_modify_mm=lambda messages: None,
        encode=lambda messages, **kwargs: seen.setdefault("messages", messages) and [1, 2],
        tokenizer=None,
    )

    async def add_requests_async(*args, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(api_server, "llm", SimpleNamespace(
        model_path="test", model_runner=runner,
        check_seq_length=lambda *args: True,
        add_requests_async=add_requests_async,
    ))
    monkeypatch.setattr(api_server, "response_store", ResponseStore())
    monkeypatch.setattr(api_server, "create_reasoning_parser", lambda *args: None)

    async def complete(*args):
        return {"id": "resp_new", "model": "test", "output": output or [], "status": "completed"}

    monkeypatch.setattr(api_server, "response_completion_generator", complete)
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(dp_index=None)))
    return seen, raw_request


def test_previous_response_id_not_found(monkeypatch):
    _, raw_request = _mock_endpoint(monkeypatch)
    request = ResponseRequest(model="test", input="hi", previous_response_id="resp_missing")
    result = asyncio.run(api_server.create_response(request, raw_request))
    assert result.status_code == 404
    error = json.loads(result.body)["error"]
    assert error["param"] == "previous_response_id"
    assert error["code"] == "response_not_found"


def test_previous_response_id_splices_history(monkeypatch):
    seen, raw_request = _mock_endpoint(monkeypatch)
    api_server.response_store.put("resp_old", {
        "model": "test",
        "input_items": [{"type": "message", "role": "user", "content": "My name is Zhang San."}],
        "response": {
            "id": "resp_old", "model": "test", "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Your name is Zhang San."}]},
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            ],
        },
    })
    request = ResponseRequest(model="test", input="And my age?", previous_response_id="resp_old")
    result = asyncio.run(api_server.create_response(request, raw_request))
    assert result.status_code == 200
    assert seen["messages"][0] == {"role": "user", "content": "My name is Zhang San."}
    assert seen["messages"][1]["role"] == "assistant"
    assert seen["messages"][1]["content"] == "Your name is Zhang San."
    assert list(seen["messages"][1]["tool_calls"])[0]["id"] == "call_1"
    assert seen["messages"][2] == {"role": "user", "content": "And my age?"}


def test_store_false_does_not_persist(monkeypatch):
    _, raw_request = _mock_endpoint(monkeypatch)
    request = ResponseRequest(model="test", input="hi", store=False)
    result = asyncio.run(api_server.create_response(request, raw_request))
    assert result.status_code == 200
    assert api_server.response_store.get("resp_new") is None


def test_streaming_terminal_response_is_stored(monkeypatch):
    _, raw_request = _mock_endpoint(monkeypatch)

    async def stream(*args):
        yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_stream","model":"test","output":[]}}\n\n'

    monkeypatch.setattr(api_server, "response_stream_generator", stream)
    request = ResponseRequest(model="test", input="hi", store=True, stream=True)

    async def run():
        result = await api_server.create_response(request, raw_request)
        return [chunk async for chunk in result.body_iterator]

    assert len(asyncio.run(run())) == 1
    assert api_server.response_store.get("resp_stream")["response"]["model"] == "test"


def test_string_input_is_normalized_before_store(monkeypatch):
    """Regression: string input must be stored as a list, not a raw string."""
    seen, raw_request = _mock_endpoint(monkeypatch)
    request = ResponseRequest(model="test", input="hello world", store=True)
    result = asyncio.run(api_server.create_response(request, raw_request))
    assert result.status_code == 200
    record = api_server.response_store.get("resp_new")
    assert record is not None
    # Stored input_items must be a list of dicts, not the raw string
    assert isinstance(record["input_items"], list)
    assert record["input_items"][0] == {"type": "message", "role": "user", "content": "hello world"}


def test_namespace_is_preserved_in_playback(monkeypatch):
    """Regression: namespace must survive output-to-input conversion."""
    from gllm.entrypoints.serving_responses import _previous_output_to_input_items
    prev = {
        "output": [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "run",
                "arguments": "{}",
                "namespace": "editor",
            },
            {
                "type": "custom_tool_call",
                "call_id": "c2",
                "name": "lookup",
                "input": "q",
                "namespace": "search",
            },
        ]
    }
    items = _previous_output_to_input_items(prev)
    assert items[0]["namespace"] == "editor"
    assert items[1]["namespace"] == "search"


def test_streaming_store_false_does_not_persist(monkeypatch):
    """Regression: streaming + store=false must not store anything."""
    _, raw_request = _mock_endpoint(monkeypatch)

    async def stream(*args):
        yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_nostore","model":"test","output":[]}}\n\n'

    monkeypatch.setattr(api_server, "response_stream_generator", stream)
    request = ResponseRequest(model="test", input="hi", store=False, stream=True)

    async def run():
        result = await api_server.create_response(request, raw_request)
        return [chunk async for chunk in result.body_iterator]

    assert len(asyncio.run(run())) == 1
    # store=false → must NOT be in store
    assert api_server.response_store.get("resp_nostore") is None


def test_chained_multiturn_accumulates_history(monkeypatch):
    """Regression: turn 3 references turn 2, which itself references turn 1."""
    seen, raw_request = _mock_endpoint(monkeypatch)

    # Turn 1: "My name is Zhang San"
    api_server.response_store.put("resp_t1", {
        "model": "test",
        "input_items": [{"type": "message", "role": "user", "content": "My name is Zhang San."}],
        "response": {
            "id": "resp_t1", "model": "test",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello Zhang San."}]}],
        },
    })

    # Turn 2: "I am 25 years old" — references turn 1
    call_count = [0]
    async def complete(*args):
        call_count[0] += 1
        rid = f"resp_t{call_count[0] + 1}"
        return {"id": rid, "model": "test",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": f"reply {call_count[0]}"}]}],
                "status": "completed"}
    monkeypatch.setattr(api_server, "response_completion_generator", complete)

    req2 = ResponseRequest(model="test", input="I am 25 years old.",
                           previous_response_id="resp_t1", store=True)
    result2 = asyncio.run(api_server.create_response(req2, raw_request))
    assert result2.status_code == 200
    # Stored turn 2 input_items should be: t1_user + t1_assistant + t2_user
    stored_t2 = api_server.response_store.get("resp_t2")
    assert len(stored_t2["input_items"]) == 3
    assert stored_t2["input_items"][0]["content"] == "My name is Zhang San."

    # Turn 3: "How old am I?" — references turn 2
    req3 = ResponseRequest(model="test", input="How old am I?",
                           previous_response_id="resp_t2", store=True)
    result3 = asyncio.run(api_server.create_response(req3, raw_request))
    assert result3.status_code == 200
    # Turn 3 prompt should have: t1_user + t1_assistant + t2_user + t2_assistant + t3_user
    # = 5 messages
    stored_t3 = api_server.response_store.get("resp_t3")
    assert len(stored_t3["input_items"]) == 5
