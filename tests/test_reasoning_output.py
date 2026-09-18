import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest
from gllm.entrypoints.serving_chat import (
    chat_completion_generator, chat_completion_stream_generator,
)
from gllm.entrypoints.serving_responses import (
    make_chat_request, response_completion_generator, response_stream_generator,
)
from gllm.tokenizers.reasoning import ThinkParser, create_reasoning_parser
from gllm.tokenizers.tool_parsers import QwenToolParser, Qwen3ToolParser
from gllm.utils import StreamOutput


class Stream:
    def __init__(self, chunks, *, truncated=False):
        self.chunks = chunks
        self.seq = SimpleNamespace(
            token_ids=[1, 2, 3], raw_prompt_len=2, ignore_eos=False,
            finish_tokens=[99], output_len=1 if truncated else 128,
        )

    async def __aiter__(self):
        for chunk in self.chunks:
            yield StreamOutput(chunk)


def collect(generator):
    async def run():
        return [json.loads(next(l[6:] for l in raw.splitlines() if l.startswith("data: ")))
                async for raw in generator if "data: [DONE]" not in raw]
    return asyncio.run(run())


@pytest.mark.parametrize("prefilled", [False, True])
def test_every_chunk_boundary_and_literal_tags_in_answer(prefilled):
    thought = "Plan with <tool_call>fake</tool_call>."
    answer = "\nAnswer with literal <think> and </think> examples."
    text = ("" if prefilled else "<think>") + thought + "</think>" + answer
    for boundary in range(len(text) + 1):
        parser = ThinkParser(prefilled=prefilled)
        pieces = [parser.feed(text[:boundary]), parser.feed(text[boundary:]), parser.finish()]
        assert "".join(p[0] for p in pieces) == thought
        assert "".join(p[1] for p in pieces) == answer


@pytest.mark.parametrize("text", ["", "hello", "<", "<thi", " <thing>", "Use <think> literally."])
def test_nonreasoning_text_is_lossless(text):
    parser = ThinkParser()
    pieces = [parser.feed(char) for char in text] + [parser.finish()]
    assert "".join(p[0] for p in pieces) == ""
    assert "".join(p[1] for p in pieces) == text


@pytest.mark.parametrize("prefilled", [False, True])
@pytest.mark.parametrize("end", ["", "<", "</thi"])
def test_truncated_thought_never_becomes_answer(prefilled, end):
    text = ("" if prefilled else "<think>") + "unfinished" + end
    parser = ThinkParser(prefilled=prefilled)
    pieces = [parser.feed(char) for char in text] + [parser.finish()]
    assert "".join(p[0] for p in pieces) == "unfinished" + end
    assert not "".join(p[1] for p in pieces)


class Tokenizer:
    def __init__(self, suffix):
        self.suffix = suffix

    def convert_tokens_to_ids(self, marker):
        return {"<think>": 10, "</think>": 11}.get(marker)

    def decode(self, ids, **kwargs):
        return { (10,): "<think>", (11,): "</think>" }.get(tuple(ids), self.suffix)


@pytest.mark.parametrize("suffix,prefilled", [
    ("assistant\n<think>\n", True),
    ("assistant\n<think>\n\n</think>\n\n", False),
    ("user <think> example\nassistant\n", False),
])
def test_actual_prompt_controls_initial_state(suffix, prefilled):
    assert create_reasoning_parser(Tokenizer(suffix), [1, 2]).prefilled is prefilled
    assert create_reasoning_parser(None, [1, 2]) is None


def test_unknown_tokenizer_does_not_enable_parser():
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 0, decode=lambda *a, **k: "<unk>")
    assert create_reasoning_parser(tokenizer, [1, 2]) is None


def response_request(summary, tools):
    return ResponseRequest(
        model="test", input="hello", reasoning={"effort": "medium", "summary": summary},
        tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}] if tools else None,
    )


THOUGHT = 'Plan <tool_call>{"name":"fake","arguments":{}}</tool_call> carefully.'
TOOL = '<tool_call>{"name":"lookup","arguments":{"key":"ok"}}</tool_call>'
QWEN3_TOOL = '<tool_call>\n<function=lookup>\n<parameter=key>ok</parameter>\n</function>\n</tool_call>'


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prefilled", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("summary", [None, "none"])
@pytest.mark.parametrize("parser_class,tool_text", [(QwenToolParser, TOOL), (Qwen3ToolParser, QWEN3_TOOL)])
def test_responses_separate_reasoning_and_preserve_tools(streaming, prefilled, tools, summary, parser_class, tool_text):
    req = response_request(summary, tools)
    chat = make_chat_request(req)
    text = ("" if prefilled else "<think>") + THOUGHT + "</think>" + (tool_text if tools else "Hello!")
    parser = ThinkParser(prefilled=prefilled)
    stream = Stream(list(text))
    if streaming:
        events = collect(response_stream_generator(stream, req, chat, parser_class(), parser))
        adapter = TypeAdapter(ResponseStreamEvent)
        for event in events:
            adapter.validate_python(event)
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
        result = events[-1]["response"]
        assert events[-1]["type"] == "response.completed"
        assert "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta") == ("" if tools else "Hello!")
        assert "".join(e["delta"] for e in events if e["type"] == "response.reasoning_text.delta") == ("" if summary == "none" else THOUGHT)
        added = [e for e in events if e["type"] == "response.output_item.added"]
        done = [e for e in events if e["type"] == "response.output_item.done"]
        assert [e["output_index"] for e in added] == list(range(len(result["output"])))
        assert [e["item"]["id"] for e in added] == [e["item"]["id"] for e in done]
        assert [e["item"] for e in done] == result["output"]
    else:
        result = asyncio.run(response_completion_generator(stream, req, chat, parser_class(), parser))
    Response.model_validate(result)
    output = result["output"]
    if summary != "none":
        assert output[0]["type"] == "reasoning"
        assert output[0]["content"][0]["text"] == THOUGHT
        assert output[0]["summary"] == []
    else:
        assert THOUGHT not in json.dumps(result)
    if tools:
        assert output[-1]["type"] == "function_call"
        assert output[-1]["name"] == "lookup"
        assert json.loads(output[-1]["arguments"]) == {"key": "ok"}
    else:
        assert output[-1]["content"][0]["text"] == "Hello!"
    # Returned reasoning items must not make the next stateless request fail.
    continuation = make_chat_request(ResponseRequest(model="test", input=output + [{"role": "user", "content": "next"}]))
    assert continuation.messages[-1]["content"] == "next"
    assert THOUGHT not in continuation.model_dump_json()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("chunks", ["single", "characters"])
def test_chat_separates_reasoning_before_tool_parsing(streaming, tools, chunks):
    req = make_chat_request(response_request(None, tools))
    text = THOUGHT + "</think>" + ("Looking up." + TOOL if tools else "Hello!")
    stream = Stream([text] if chunks == "single" else list(text))
    parser = ThinkParser(prefilled=True)
    if streaming:
        events = collect(chat_completion_stream_generator(stream, req, QwenToolParser(), parser))
        deltas = [e["choices"][0]["delta"] for e in events]
        assert "".join(d.get("reasoning_content", "") for d in deltas) == THOUGHT
        content = "".join(d.get("content", "") for d in deltas)
        calls = [call for d in deltas for call in d.get("tool_calls", [])]
        if tools:
            assert calls[0]["function"]["name"] == "lookup"
            assert json.loads("".join(c["function"].get("arguments", "") for c in calls)) == {"key": "ok"}
        assert events[-1]["choices"][0]["finish_reason"] == ("tool_calls" if tools else "stop")
    else:
        result = asyncio.run(chat_completion_generator(stream, req, QwenToolParser(), parser))
        message = result.choices[0].message
        assert message.reasoning_content == THOUGHT
        content = message.content
        if tools:
            assert message.tool_calls[0].function.name == "lookup"
    assert content == ("Looking up." if tools else "Hello!")


@pytest.mark.parametrize("summary", [None, "none"])
def test_reasoning_only_length_limit(summary):
    req = response_request(summary, False)
    events = collect(response_stream_generator(
        Stream(["unfinished</thi"], truncated=True), req, make_chat_request(req),
        reasoning_parser=ThinkParser(prefilled=True),
    ))
    assert events[-1]["type"] == "response.incomplete"
    assert not any(e["type"] == "response.output_text.delta" for e in events)
    if summary != "none":
        assert events[-1]["response"]["output"][0]["status"] == "incomplete"


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("disagg", [False, True])
def test_routes_use_rendered_prompt_even_when_effort_is_none(monkeypatch, endpoint, streaming, disagg):
    from gllm.entrypoints import api_server

    async def add_requests(*args, **kwargs):
        return Stream(["private thought</think>Hello!"])

    runner = SimpleNamespace(
        tokenizer=Tokenizer("assistant\n<think>\n"), use_mm=True,
        extract_modify_mm=lambda messages: ["image"] if disagg else None,
        extract_mm_items_ordered=lambda messages: ["image"],
        encode=lambda *args, **kwargs: [1, 2],
        encode_skeleton=lambda *args, **kwargs: [1, 2],
    )
    monkeypatch.setattr(api_server, "llm", SimpleNamespace(
        model_path="test", model_runner=runner, is_disagg_lm=disagg,
        check_seq_length=lambda *args: True, add_requests_async=add_requests,
    ))
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    if endpoint == "chat":
        req = ChatCompletionRequest(model="test", messages=[], stream=streaming,
                                    reasoning_effort="none", chat_template_kwargs={"enable_thinking": True})
        response = asyncio.run(api_server.create_chat_completion(req, raw))
    else:
        req = ResponseRequest(model="test", input="hello", stream=streaming,
                              reasoning={"summary": "none"})
        response = asyncio.run(api_server.create_response(req, raw))
    if streaming:
        events = collect(response.body_iterator)
        if endpoint == "chat":
            assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "Hello!"
        else:
            assert "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta") == "Hello!"
    else:
        data = json.loads(response.body)
        if endpoint == "chat":
            assert data["choices"][0]["message"]["content"] == "Hello!"
        else:
            assert data["output"][0]["content"][0]["text"] == "Hello!"


@pytest.mark.parametrize("tail", ["<", "<tool_ca"])
def test_incomplete_tool_marker_is_flushed_as_literal_at_eof(tail):
    req = make_chat_request(response_request(None, True))
    events = collect(chat_completion_stream_generator(Stream(list("Literal " + tail)), req, QwenToolParser()))
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "Literal " + tail
