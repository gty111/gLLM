import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from openai.types.responses.response import Response
from pydantic import TypeAdapter

from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest
from gllm.entrypoints.serving_chat import chat_completion_stream_generator
from gllm.entrypoints.serving_responses import (
    make_chat_request,
    response_completion_generator,
    response_input_to_messages,
    response_stream_generator,
)
from gllm.tokenizers.tool_parsers import QwenToolParser
from gllm.utils import StreamOutput


class FakeStream:
    def __init__(self, text="hello", texts=None):
        self.items = [StreamOutput(part) for part in (texts or [text])]
        self.iterated = False
        self.seq = SimpleNamespace(
            token_ids=[10, 11, 12],
            raw_prompt_len=2,
            ignore_eos=False,
            finish_tokens=[99],
            output_len=8,
        )

    async def __aiter__(self):
        self.iterated = True
        for item in self.items:
            yield item


def _collect(generator):
    async def run():
        return [item async for item in generator]

    return asyncio.run(run())


def test_current_chat_tool_and_json_schema_request_shapes_validate():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "developer", "content": "Be concise"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": "auto",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        }
    )
    assert request.tools[0].function.strict is True
    assert request.response_format.type == "json_schema"


def test_legacy_functions_are_translated_to_tools():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [{"name": "ping"}],
            "function_call": {"name": "ping"},
        }
    )
    assert request.tools[0].function.name == "ping"
    assert request.tool_choice.function.name == "ping"


def test_chat_stream_has_stable_identity_role_and_separate_usage_chunk():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "request_id": "request-1",
        }
    )
    chunks = _collect(chat_completion_stream_generator(FakeStream(), request))
    payloads = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks[:-1]]
    assert {payload["id"] for payload in payloads} == {"chatcmpl-request-1"}
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["choices"] == []
    assert payloads[-1]["usage"]["total_tokens"] == 3
    assert chunks[-1] == "data: [DONE]\n\n"


def test_responses_nonstream_and_stream_parse_with_current_sdk():
    request = ResponseRequest.model_validate(
        {"model": "test", "input": "hello", "instructions": "Be concise"}
    )
    chat_request = make_chat_request(request)
    response = asyncio.run(
        response_completion_generator(FakeStream(), request, chat_request)
    )
    parsed = Response.model_validate(response)
    assert parsed.output_text == "hello"

    events = _collect(response_stream_generator(FakeStream(), request, chat_request))
    adapter = TypeAdapter(__import__("openai").types.responses.ResponseStreamEvent)
    parsed_events = []
    for raw in events:
        data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
        parsed_events.append(adapter.validate_json(data_line.removeprefix("data: ")))
    assert parsed_events[0].type == "response.created"
    assert parsed_events[-1].type == "response.completed"


def test_responses_stream_emits_before_generation_and_preserves_engine_deltas():
    request = ResponseRequest.model_validate(
        {"model": "test", "input": "hello", "stream": True}
    )
    chat_request = make_chat_request(request)
    stream = FakeStream(texts=["he", "llo"])

    async def run():
        generator = response_stream_generator(stream, request, chat_request)
        first = await generator.__anext__()
        assert stream.iterated is False
        return [first, *[event async for event in generator]]

    events = asyncio.run(run())
    payloads = [
        json.loads(
            next(line for line in raw.splitlines() if line.startswith("data: "))[
                len("data: ") :
            ]
        )
        for raw in events
    ]
    deltas = [
        payload["delta"]
        for payload in payloads
        if payload["type"] == "response.output_text.delta"
    ]
    assert deltas == ["he", "llo"]
    assert payloads[-1]["response"]["output"][0]["content"][0]["text"] == "hello"


def test_responses_stream_translates_native_tool_markup_without_leaking_it():
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": "calculate 1+2",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "calculate",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                }
            ],
        }
    )
    chat_request = make_chat_request(request)
    stream = FakeStream(
        texts=[
            "<tool_call>",
            '{"name":"calculate","arguments":{"expression":"1+2"}}',
            "</tool_call>",
        ]
    )
    events = _collect(
        response_stream_generator(
            stream, request, chat_request, tool_parser=QwenToolParser()
        )
    )
    payloads = [
        json.loads(
            next(line for line in raw.splitlines() if line.startswith("data: "))[
                len("data: ") :
            ]
        )
        for raw in events
    ]
    adapter = TypeAdapter(__import__("openai").types.responses.ResponseStreamEvent)
    for payload in payloads:
        adapter.validate_python(payload)
    event_types = [payload["type"] for payload in payloads]
    assert "response.output_text.delta" not in event_types
    assert "response.function_call_arguments.delta" in event_types
    function_call = payloads[-1]["response"]["output"][0]
    assert function_call["type"] == "function_call"
    assert function_call["name"] == "calculate"
    assert json.loads(function_call["arguments"]) == {"expression": "1+2"}


def test_stateless_response_tool_continuation_requires_original_user_input():
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "20",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="must include a user message"):
        make_chat_request(request)


def test_responses_input_image_uses_native_media_shape():
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is shown?"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAAA",
                            "detail": "auto",
                        },
                    ],
                }
            ],
        }
    )
    messages = response_input_to_messages(request)
    assert messages[0]["content"] == [
        {"type": "text", "text": "What is shown?"},
        {"type": "image", "image": "data:image/png;base64,AAAA"},
    ]
    assert all(part["type"] != "image_url" for part in messages[0]["content"])


def test_responses_inline_text_file_is_added_to_prompt():
    data = base64.b64encode("alpha,beta\n1,2\n".encode()).decode()
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "values.csv",
                            "file_data": data,
                        },
                        {"type": "input_text", "text": "Summarize it."},
                    ],
                }
            ],
        }
    )
    messages = response_input_to_messages(request)
    assert isinstance(messages[0]["content"], str)
    assert "[File: values.csv]" in messages[0]["content"]
    assert "alpha,beta" in messages[0]["content"]
    assert messages[0]["content"].endswith("Summarize it.")


def test_responses_file_id_fails_explicitly_until_files_api_exists():
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": "file_123"},
                        {"type": "input_text", "text": "Summarize it."},
                    ],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Files API"):
        response_input_to_messages(request)


def test_responses_pdf_is_explicitly_unsupported():
    encoded = base64.b64encode(b"%PDF-1.7\n").decode()
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "sample.pdf",
                            "file_data": encoded,
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="PDF file inputs are not supported"):
        response_input_to_messages(request)


def test_responses_audio_stays_explicitly_unsupported():
    request = ResponseRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": "AAAA"}}
                    ],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Audio input is not supported"):
        response_input_to_messages(request)
