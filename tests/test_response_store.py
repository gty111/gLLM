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
