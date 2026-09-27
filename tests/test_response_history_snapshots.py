"""Stored history must preserve file bytes and canonical checkpoint identity."""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from gllm.entrypoints import api_server, serving_responses
from gllm.entrypoints.protocol import ResponseRequest
from gllm.utils import StreamOutput
from tests.test_response_store import _mock_endpoint


class TextStream:
    def __init__(self):
        self.seq = SimpleNamespace(token_ids=[1, 2, 99], raw_prompt_len=2,
                                   ignore_eos=False, finish_tokens=[99], output_len=16)

    async def __aiter__(self):
        yield StreamOutput("Remembered.")


def endpoint(monkeypatch):
    seen, raw = _mock_endpoint(monkeypatch)

    async def add_requests(*args, **kwargs):
        return TextStream()

    monkeypatch.setattr(api_server.llm, "add_requests_async", add_requests)
    monkeypatch.setattr(api_server, "response_completion_generator", serving_responses.response_completion_generator)
    monkeypatch.setattr(api_server, "response_stream_generator", serving_responses.response_stream_generator)
    return seen, raw


async def consume(request, raw):
    response = await api_server.create_response(request, raw)
    assert response.status_code == 200, getattr(response, "body", None)
    if not request.stream:
        return json.loads(response.body)
    result = None
    async for frame in response.body_iterator:
        event = json.loads(next(line[6:] for line in frame.splitlines() if line.startswith("data: ")))
        if event["type"] == "response.completed":
            result = event["response"]
    assert result is not None
    return result


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_file_snapshot_survives_expiry_or_change(monkeypatch, streaming, changed):
    seen, raw = endpoint(monkeypatch)
    calls = []

    def download(url, param):
        calls.append(url)
        if len(calls) == 1:
            return b"Budget: 100", "text/plain", "budget.txt"
        if changed:
            return b"Budget: 200", "text/plain", "budget.txt"
        raise ValueError(param, "Expired signed URL")

    monkeypatch.setattr(serving_responses, "_download_file", download)
    part = {"type": "input_file", "file_url": "https://example.invalid/budget.txt"}
    request = ResponseRequest(model="test", store=True, stream=streaming, input=[{
        "role": "user", "content": [{"type": "input_text", "text": "Remember this."}, part],
    }])
    first = asyncio.run(consume(request, raw))
    assert "Budget: 100" in seen["messages"][0]["content"]
    assert request.input[0]["content"][1] == part  # no caller mutation
    stored = api_server.response_store.get(first["id"])
    snapshot = stored["input_items"][0]["content"][1]
    assert "file_url" not in snapshot
    assert snapshot["filename"] == "budget.txt"
    assert base64.b64decode(snapshot["file_data"].split(",", 1)[1]) == b"Budget: 100"
    frozen = json.dumps(stored)
    previous = first["id"]
    for store in (True, False):
        seen.clear()
        result = asyncio.run(consume(ResponseRequest(
            model="test", input="What is the budget?", previous_response_id=previous,
            store=store, stream=streaming,
        ), raw))
        assert "Budget: 100" in seen["messages"][0]["content"]
        assert "Budget: 200" not in seen["messages"][0]["content"]
        previous = result["id"]
    assert len(calls) == 1
    assert json.dumps(stored) == frozen


@pytest.mark.parametrize("streaming", [False, True])
def test_aliases_share_checkpoint_history(monkeypatch, streaming):
    _, raw = endpoint(monkeypatch)
    path = "/cache/models--org--test/snapshots/revision"
    api_server.llm.model_path = path
    previous = None
    for name in (path, "org/test", "revision"):
        result = asyncio.run(consume(ResponseRequest(
            model=name, input="hello", store=True, stream=streaming, previous_response_id=previous,
        ), raw))
        assert result["model"] == name
        assert api_server.response_store.get(result["id"])["model"] == path
        previous = result["id"]


def test_history_for_different_checkpoint_is_rejected(monkeypatch):
    _, raw = endpoint(monkeypatch)
    api_server.response_store.put("resp_other", {
        "model": "/other/checkpoint", "input_items": [], "response": {"output": []},
    })
    response = asyncio.run(api_server.create_response(ResponseRequest(
        model="test", input="hi", previous_response_id="resp_other",
    ), raw))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["param"] == "previous_response_id"


def test_store_false_keeps_file_url_and_does_not_persist(monkeypatch):
    _, raw = endpoint(monkeypatch)
    calls = []

    def download(*args):
        calls.append(args)
        return b"Budget: 100", "text/plain", "budget.txt"

    monkeypatch.setattr(serving_responses, "_download_file", download)
    request = ResponseRequest(model="test", store=False, input=[{"role": "user", "content": [
        {"type": "input_file", "file_url": "https://example.invalid/file"},
    ]}])
    result = asyncio.run(consume(request, raw))
    assert len(calls) == 1
    assert "file_url" in request.input[0]["content"][0]
    assert api_server.response_store.get(result["id"]) is None


@pytest.mark.parametrize("tool", [False, True])
def test_snapshot_preserves_filename_and_tool_item_identity(monkeypatch, tool):
    monkeypatch.setattr(serving_responses, "_download_file", lambda *args: (
        b"value", None, "download.txt",
    ))
    part = {"type": "input_file", "file_url": "https://example.invalid/file", "filename": "custom.csv"}
    item = ({"type": "function_call_output", "call_id": "c1", "output": [part]} if tool else
            {"role": "user", "content": [part]})
    request = ResponseRequest(model="test", input=[{"role": "user", "content": "Read"}, item])
    frozen = serving_responses.snapshot_response_files(request)
    assert serving_responses.response_input_to_messages(frozen)[-1]["content"] == "\n[File: custom.csv]\nvalue\n"
    if tool:
        assert frozen.input[-1]["call_id"] == "c1"
    assert "file_url" in part


@pytest.mark.parametrize("kind", ["message", "function_call_output", "custom_tool_call_output"])
def test_image_file_snapshot_replays_same_pixels(monkeypatch, kind):
    data = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(data, format="PNG")
    monkeypatch.setattr(serving_responses, "_download_file", lambda *args: (
        data.getvalue(), None, "original.png",
    ))
    parts = [{"type": "input_file", "file_url": "https://example.invalid/image"}]
    item = ({"role": "user", "content": parts} if kind == "message" else
            {"type": kind, "call_id": "view-1", "output": parts})
    request = ResponseRequest(model="test", input=[{"role": "user", "content": "Describe it."}, item])
    frozen = serving_responses.snapshot_response_files(request)
    monkeypatch.setattr(serving_responses, "_download_file", lambda *args: pytest.fail("Re-downloaded image"))
    image = serving_responses.make_chat_request(frozen).messages[-1]["content"][0]["image"]
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_tool_image_survives_endpoint_and_previous_response(monkeypatch, streaming, custom):
    from gllm.multimodal.mixin import MmMixin

    seen, raw = endpoint(monkeypatch)
    runner = api_server.llm.model_runner
    runner.use_mm = True

    def extract(messages):
        seen["media"] = MmMixin.extract_modify_mm(None, messages)
        return seen["media"]

    runner.extract_modify_mm = extract
    image_url = "data:image/png;base64,AAAA"
    call = {"type": "custom_tool_call" if custom else "function_call",
            "call_id": "view-1", "name": "view_image"}
    call.update({"input": "plot.png"} if custom else {"arguments": '{"path":"plot.png"}'})
    output = {"type": "custom_tool_call_output" if custom else "function_call_output",
              "call_id": "view-1", "output": [{"type": "input_image", "image_url": image_url}]}
    req = ResponseRequest(model="test", store=True, stream=streaming, input=[
        {"role": "user", "content": "Describe the image."}, call, output,
    ])
    first = asyncio.run(consume(req, raw))
    assert seen["media"] == {"image": [image_url], "video": []}
    assert seen["messages"][-1]["tool_call_id"] == "view-1"
    stored = api_server.response_store.get(first["id"])
    assert stored["input_items"][-1] == output
    frozen = json.dumps(stored)
    seen.clear()
    asyncio.run(consume(ResponseRequest(
        model="test", input="Describe it again.", previous_response_id=first["id"],
        stream=streaming, store=False,
    ), raw))
    assert seen["media"] == {"image": [image_url], "video": []}
    assert next(m for m in seen["messages"] if m["role"] == "tool")["tool_call_id"] == "view-1"
    assert json.dumps(stored) == frozen


def test_tool_image_is_rejected_for_text_only_model(monkeypatch):
    from gllm.multimodal.mixin import MmMixin

    _, raw = endpoint(monkeypatch)
    runner = api_server.llm.model_runner
    runner.use_mm = False
    runner.extract_modify_mm = lambda messages: MmMixin.extract_modify_mm(None, messages)
    response = asyncio.run(api_server.create_response(ResponseRequest(model="test", input=[
        {"role": "user", "content": "Describe it."},
        {"type": "function_call_output", "call_id": "view-1", "output": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]},
    ]), raw))
    assert response.status_code == 400
    assert "does not support image inputs" in json.loads(response.body)["error"]["message"]
