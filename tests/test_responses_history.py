"""Responses item boundaries must not become extra model conversation turns."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.serving_responses import (
    make_chat_request, response_completion_generator,
    response_input_to_messages, response_stream_generator,
)
from gllm.tokenizers.reasoning import ThinkParser
from gllm.tokenizers.tool_parsers import Qwen3ToolParser, normalize_chat_template_messages
from gllm.utils import StreamOutput


def message(role, text):
    return {"type": "message", "role": role, "content": text}


def call(call_id, *, custom=False, namespace=None):
    item = {"type": "custom_tool_call" if custom else "function_call",
            "call_id": call_id, "name": "run"}
    if namespace:
        item["namespace"] = namespace
    item.update({"input": "echo ok"} if custom else {"arguments": '{"cmd":"echo ok"}'})
    return item


def convert(items):
    return response_input_to_messages(ResponseRequest(model="test", input=items))


@pytest.mark.parametrize("text", [None, "", "Let me read the file."])
def test_text_and_multiple_calls_share_one_assistant_message(text):
    items = [message("user", "Inspect the file.")]
    if text is not None:
        items.append(message("assistant", text))
    # Reasoning items remain omitted, without creating artificial boundaries.
    items.extend([
        {"type": "reasoning", "summary": [], "content": [
            {"type": "reasoning_text", "text": "private reasoning"}]},
        call("read-1"), call("edit-1", custom=True, namespace="editor"),
        {"type": "function_call_output", "call_id": "read-1", "output": "file contents"},
        {"type": "custom_tool_call_output", "call_id": "edit-1", "output": "edited"},
    ])
    before = json.dumps(items)
    messages = convert(items)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
    assistant = messages[1]
    assert assistant["content"] == text
    assert [c["id"] for c in assistant["tool_calls"]] == ["read-1", "edit-1"]
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["run", "editor.run"]
    assert [json.loads(c["function"]["arguments"]) for c in assistant["tool_calls"]] == [
        {"cmd": "echo ok"}, {"input": "echo ok"},
    ]
    assert [m["tool_call_id"] for m in messages[2:]] == ["read-1", "edit-1"]
    assert "private reasoning" not in json.dumps(messages)
    assert json.dumps(items) == before


@pytest.mark.parametrize("boundary", [
    message("user", "Next request"), "Next request",
    message("developer", "Additional instruction"),
    {"type": "function_call_output", "call_id": "first", "output": "done"},
])
def test_calls_do_not_merge_across_other_roles(boundary):
    messages = convert([
        message("user", "Start"), message("assistant", "First action"),
        call("first"), boundary, call("second"),
    ])
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert [c["id"] for c in assistants[0]["tool_calls"]] == ["first"]
    assert [c["id"] for c in assistants[1]["tool_calls"]] == ["second"]


def test_distinct_assistant_text_messages_are_not_merged():
    messages = convert([
        message("user", "Start"), message("assistant", "First message"),
        message("assistant", "Second message"), call("read"),
    ])
    assert [m["content"] for m in messages] == ["Start", "First message", "Second message"]
    assert "tool_calls" not in messages[1]
    assert messages[2]["tool_calls"][0]["id"] == "read"


class Stream:
    def __init__(self, text):
        self.text = text
        self.seq = SimpleNamespace(token_ids=[1, 2, 99], raw_prompt_len=2,
                                   ignore_eos=False, finish_tokens=[99], output_len=256)

    async def __aiter__(self):
        # Include delimiter boundaries in the streaming round trip.
        for char in self.text:
            yield StreamOutput(char)


def roundtrip(streaming, custom, namespace):
    tool = {"type": "custom" if custom else "function", "name": "run"}
    if not custom:
        tool["parameters"] = {"type": "object", "properties": {"cmd": {"type": "string"}}}
    tools = ([{"type": "namespace", "name": namespace, "description": "Shell tools", "tools": [tool]}]
             if namespace else [tool])
    req = ResponseRequest(model="test", input="Inspect the file.", tools=tools)
    name = f"{namespace}.run" if namespace else "run"
    parameter = "input" if custom else "cmd"
    raw = ("<think>Plan.</think>Let me read the file.\n"
           f"<tool_call>\n<function={name}>\n<parameter={parameter}>echo ok</parameter>\n"
           "</function>\n</tool_call>")
    args = (Stream(raw), req, make_chat_request(req), Qwen3ToolParser(), ThinkParser())
    if streaming:
        async def collect():
            return [json.loads(wire.split("data: ", 1)[1])
                    async for wire in response_stream_generator(*args)]
        response = asyncio.run(collect())[-1]["response"]
    else:
        response = asyncio.run(response_completion_generator(*args))
    assert response["status"] == "completed"
    item = response["output"][-1]
    history = ResponseRequest(model="test", tools=tools, input=[
        message("user", "Inspect the file."), *response["output"],
        {"type": "custom_tool_call_output" if custom else "function_call_output",
         "call_id": item["call_id"], "output": "file contents"},
    ])
    chat = make_chat_request(history)
    # The request schema exposes lazy tool-call iterables; use the same
    # normalization as ModelRunner.encode before inspecting/rendering them.
    normalize_chat_template_messages(chat.messages)
    assert [m["role"] for m in chat.messages] == ["user", "assistant", "tool"]
    assistant = chat.messages[1]
    assert assistant["content"].strip() == "Let me read the file."
    assert assistant["tool_calls"][0]["id"] == chat.messages[2]["tool_call_id"]
    assert assistant["tool_calls"][0]["function"]["name"] == name
    assert assistant["tool_calls"][0]["function"]["arguments"] == {parameter: "echo ok"}
    return chat


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("namespace", [None, "shell"])
def test_generated_items_roundtrip_into_one_assistant_message(streaming, custom, namespace):
    roundtrip(streaming, custom, namespace)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    model_path = os.environ.get("GLLM_TEST_QWEN_TOKENIZER")
    if not model_path:
        pytest.skip("Set GLLM_TEST_QWEN_TOKENIZER to a local Qwen3.8 checkpoint")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


@pytest.mark.parametrize("custom", [False, True])
def test_qwen_template_has_no_end_marker_between_text_and_call(qwen_tokenizer, custom):
    chat = roundtrip(True, custom, "shell")
    normalize_chat_template_messages(chat.messages)
    rendered = qwen_tokenizer.apply_chat_template(
        chat.messages, tools=[t.model_dump(exclude_none=True) for t in chat.tools],
        tokenize=False, add_generation_prompt=True, reasoning_effort="medium",
    )
    start = rendered.index("Let me read the file.")
    end = rendered.index("<tool_response>", start)
    assistant = rendered[start:end]
    assert assistant.index("<tool_call>") < assistant.index("<|im_end|>")
    assert assistant.count("<|im_end|>") == 1
    assert "<function=shell.run>" in assistant
    assert "<|im_start|>assistant" not in assistant
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
