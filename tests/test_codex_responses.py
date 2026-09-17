"""Native Responses shapes observed with Codex CLI 0.154.0."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.responses.response import Response
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter, ValidationError

from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest
from gllm.entrypoints.response_tools import chat_tools
from gllm.entrypoints.serving_responses import (
    make_chat_request, response_completion_generator,
    response_input_to_messages, response_stream_generator,
)
from gllm.tokenizers.tool_parsers import Qwen3ToolParser, QwenToolParser
from gllm.utils import StreamOutput


class Stream:
    def __init__(self, name, arguments):
        self.text = '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'
        self.seq = SimpleNamespace(token_ids=[1, 2, 3], raw_prompt_len=2,
                                   ignore_eos=False, finish_tokens=[99], output_len=8)

    async def __aiter__(self):
        for part in (self.text[:15], self.text[15:]):
            yield StreamOutput(part)


def request(tools, **kwargs):
    return ResponseRequest(model="test", input="Use the tools.", tools=tools, **kwargs)


def complete(req, name, args):
    return asyncio.run(response_completion_generator(
        Stream(name, args), req, make_chat_request(req), QwenToolParser()))


def events(req, name, args):
    async def run():
        return [json.loads(next(line[6:] for line in raw.splitlines() if line.startswith("data: ")))
                async for raw in response_stream_generator(
                    Stream(name, args), req, make_chat_request(req), QwenToolParser())]
    result = asyncio.run(run())
    adapter = TypeAdapter(ResponseStreamEvent)
    for event in result:
        adapter.validate_python(event)
    return result


def custom_tool():
    return {"type": "custom", "name": "apply_patch", "description": "Edit files.",
            "format": {"type": "grammar", "syntax": "lark",
                       "definition": 'start: "*** Begin Patch" LF "*** End Patch" LF?\n%import common.LF\n'}}


def test_codex_metadata_is_accepted_but_never_added_to_prompt():
    req = request([], client_metadata={"session_id": "opaque-session"},
                  include=["reasoning.encrypted_content"], reasoning={"effort": "none"})
    assert "opaque-session" not in json.dumps(make_chat_request(req).model_dump())
    with pytest.raises(ValidationError):
        ResponseRequest(model="test", input="hello", invented_option=True)


def test_include_selector_validation(monkeypatch):
    from gllm.entrypoints import api_server
    monkeypatch.setattr(api_server, "llm", SimpleNamespace(model_path="test"))
    assert api_server._validate_response_capabilities(
        request([], include=["reasoning.encrypted_content"])) is None
    assert api_server._validate_response_capabilities(
        request([], include=["file_search_call.results"])).status_code == 400


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.parametrize("choice", [None, "none", "auto"])
def test_text_only_tool_choice_without_tools(tools, choice):
    kwargs = {} if tools is None else {"tools": tools}
    chat = ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "hello"}],
        tool_choice=choice, **kwargs)
    assert chat.tool_choice == choice
    response = ResponseRequest(model="test", input=[
        {"role": "user", "content": "Summarize the README."},
        {"type": "function_call", "call_id": "read-1", "name": "exec_command",
         "arguments": '{"cmd":"cat README.md"}'},
        {"type": "function_call_output", "call_id": "read-1", "output": "An inference engine."},
    ], tool_choice=choice, **kwargs)
    converted = make_chat_request(response)
    assert not converted.tools
    assert converted.tool_choice == choice
    assert converted.messages[-1]["content"] == "An inference engine."


@pytest.mark.parametrize("tools", [None, []])
@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "read"}}])
def test_forced_tool_choice_still_requires_tools(tools, choice):
    with pytest.raises(ValidationError, match="tools.*must be set"):
        ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hello"}],
            tools=tools, tool_choice=choice)


def test_codex_instructions_and_developer_message_share_one_template_role():
    req = ResponseRequest(model="test", instructions="Base instructions", input=[
        {"role": "developer", "content": [{"type": "input_text", "text": "Project instructions"}]},
        {"role": "user", "content": "Do the task"},
    ])
    messages = response_input_to_messages(req)
    assert messages == [{"role": "developer", "content": "Base instructions\n\nProject instructions"},
                        {"role": "user", "content": "Do the task"}]


@pytest.mark.parametrize("instructions", [None, "Base instructions"])
@pytest.mark.parametrize("role", ["developer", "system"])
def test_compacted_history_moves_instructions_before_conversation(instructions, role):
    history = [
        {"role": "user", "content": "Hello"},
        {"role": role, "content": [{"type": "input_text", "text": "Project instructions"}]},
        {"role": "user", "content": "Inspect the project"},
        {"type": "function_call", "call_id": "read-1", "name": "read",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "read-1", "output": "README contents"},
        {"role": "developer", "content": "Additional instructions"},
        {"role": "user", "content": "Summary of the previous work"},
    ]
    original = json.dumps(history)
    req = ResponseRequest(model="test", instructions=instructions, input=history)
    messages = response_input_to_messages(req)
    expected = ["Project instructions", "Additional instructions"]
    if instructions:
        expected.insert(0, instructions)
    assert messages[0] == {"role": "developer" if instructions else role,
                           "content": "\n\n".join(expected)}
    assert [m["role"] for m in messages[1:]] == ["user", "user", "assistant", "tool", "user"]
    assert messages[3]["tool_calls"][0]["id"] == messages[4]["tool_call_id"] == "read-1"
    assert json.dumps(history) == original


def test_model_template_errors_return_invalid_input(monkeypatch):
    from gllm.entrypoints import api_server
    from jinja2 import TemplateError

    def fail_encode(*args, **kwargs):
        raise TemplateError("Unsupported message for this model template")

    runner = SimpleNamespace(extract_modify_mm=lambda messages: None, encode=fail_encode)
    monkeypatch.setattr(api_server, "llm", SimpleNamespace(model_path="test", model_runner=runner))
    response = asyncio.run(api_server.create_response(request([]), SimpleNamespace()))
    assert response.status_code == 400
    error = json.loads(response.body)["error"]
    assert error["code"] == "invalid_input"
    assert "model template" in error["message"]


@pytest.mark.parametrize("streaming", [False, True])
def test_custom_tool_wire_output_and_stateless_continuation(streaming):
    patch = "*** Begin Patch\n*** End Patch\n"
    req = request([custom_tool()])
    if streaming:
        output = events(req, "apply_patch", {"input": patch})
        assert [e["sequence_number"] for e in output] == list(range(len(output)))
        assert any(e["type"] == "response.custom_tool_call_input.delta" and e["delta"] == patch for e in output)
        assert any(e["type"] == "response.custom_tool_call_input.done" and e["input"] == patch for e in output)
        assert not any(e["type"].startswith("response.function_call_arguments") for e in output)
        response = output[-1]["response"]
    else:
        response = complete(req, "apply_patch", {"input": patch})
    item = Response.model_validate(response).output[0]
    assert item.type == "custom_tool_call" and item.input == patch
    history = ResponseRequest(model="test", input=[
        {"role": "user", "content": "Edit the file."}, response["output"][0],
        {"type": "custom_tool_call_output", "call_id": item.call_id,
         "output": [{"type": "input_text", "text": "Success"}]},
    ])
    messages = response_input_to_messages(history)
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"input": patch}
    assert messages[2]["content"] == "Success"


@pytest.mark.parametrize("streaming", [False, True])
def test_namespace_preserves_same_leaf_names(streaming):
    tools = [{"type": "namespace", "name": ns, "description": "Lookup tools.", "tools": [
        {"type": "function", "name": "lookup", "parameters": {"type": "object"}}]} for ns in ("first", "second")]
    req = request(tools)
    assert [t.function.name for t in make_chat_request(req).tools] == ["first.lookup", "second.lookup"]
    response = (events(req, "second.lookup", {})[-1]["response"] if streaming
                else complete(req, "second.lookup", {}))
    item = response["output"][0]
    assert item["name"] == "lookup" and item["namespace"] == "second"
    history = ResponseRequest(model="test", input=[{"role": "user", "content": "lookup"}, item])
    assert response_input_to_messages(history)[1]["tool_calls"][0]["function"]["name"] == "second.lookup"


def test_custom_namespace_roundtrip():
    req = request([{"type": "namespace", "name": "editor", "description": "Editing tools.",
                    "tools": [custom_tool()]}])
    result = complete(req, "editor.apply_patch", {"input": "*** Begin Patch\n*** End Patch\n"})
    assert result["output"][0]["namespace"] == "editor"
    assert result["output"][0]["type"] == "custom_tool_call"


@pytest.mark.parametrize("streaming", [False, True])
def test_qwen38_xml_custom_tool_roundtrip(streaming):
    patch = "*** Begin Patch\n*** End Patch"
    req = request([{"type": "namespace", "name": "editor", "description": "Editing tools.",
                    "tools": [custom_tool()]}])
    stream = Stream("unused", {})
    stream.text = (
        "<tool_call>\n<function=editor.apply_patch>\n<parameter=input>\n"
        + patch + "\n</parameter>\n</function>\n</tool_call>"
    )

    async def run():
        if not streaming:
            return await response_completion_generator(
                stream, req, make_chat_request(req), Qwen3ToolParser())
        result = []
        adapter = TypeAdapter(ResponseStreamEvent)
        async for raw in response_stream_generator(
                stream, req, make_chat_request(req), Qwen3ToolParser()):
            payload = json.loads(next(line[6:] for line in raw.splitlines()
                                      if line.startswith("data: ")))
            adapter.validate_python(payload)
            result.append(payload)
        assert not any(e["type"] == "response.output_text.delta" for e in result)
        return result[-1]["response"]

    item = Response.model_validate(asyncio.run(run())).output[0]
    assert item.type == "custom_tool_call"
    assert item.name == "apply_patch" and item.namespace == "editor"
    assert item.input == patch


def test_codex_patch_grammar_accepts_empty_added_lines():
    tool = custom_tool()
    tool["format"]["definition"] = (
        'start: "*** Begin Patch" LF "*** Add File: " filename LF add_line+ "*** End Patch" LF?\n'
        'filename: /(.+)/\nadd_line: "+" /(.*)/ LF\n%import common.LF\n'
    )
    patch = "*** Begin Patch\n*** Add File: example.txt\n+hello\n+\n*** End Patch\n"
    result = complete(request([tool]), "apply_patch", {"input": patch})
    assert result["output"][0]["input"] == patch


@pytest.mark.parametrize("args", [{"input": "invalid patch"}, {"input": 12}, {}])
def test_invalid_custom_outputs_fail_without_publishing_a_tool_call(args):
    req = request([custom_tool()])
    with pytest.raises(ValueError):
        complete(req, "apply_patch", args)
    result = events(req, "apply_patch", args)
    assert result[-1]["type"] == "response.failed"
    assert result[-1]["response"]["error"]["code"] == "server_error"
    assert result[-1]["response"]["error"]["message"]
    assert not any(e["type"] == "response.output_item.added" for e in result)


@pytest.mark.parametrize("streaming", [False, True])
def test_undeclared_tools_are_never_published(streaming):
    req = request([custom_tool()])
    if streaming:
        result = events(req, "missing_tool", {})
        assert result[-1]["type"] == "response.failed"
        assert "undeclared tool" in result[-1]["response"]["error"]["message"]
        assert not any(e["type"] == "response.output_item.added" for e in result)
    else:
        with pytest.raises(ValueError, match="undeclared tool"):
            complete(req, "missing_tool", {})


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fmt,value", [
    ({"type": "text"}, "raw text\n含中文\n"),
    ({"type": "grammar", "syntax": "regex", "definition": r"[0-9]{3}"}, "123"),
])
def test_custom_text_and_regex_formats(streaming, fmt, value):
    req = request([{"type": "custom", "name": "send", "format": fmt}])
    response = (events(req, "send", {"input": value})[-1]["response"] if streaming
                else complete(req, "send", {"input": value}))
    assert response["output"][0]["input"] == value


def test_regex_custom_tools_require_a_full_match():
    req = request([{"type": "custom", "name": "send", "format": {
        "type": "grammar", "syntax": "regex", "definition": r"[0-9]{3}"}}])
    with pytest.raises(ValueError, match="does not match the grammar"):
        complete(req, "send", {"input": "123 trailing text"})


def test_ambiguous_namespace_names_and_invalid_grammars_are_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        chat_tools([{"type": "function", "name": "ns.lookup"},
                    {"type": "namespace", "name": "ns", "description": "Lookup tools.",
                     "tools": [{"type": "function", "name": "lookup"}]}])
    tool = custom_tool()
    tool["format"]["definition"] = "not a grammar"
    with pytest.raises(ValueError, match="Invalid custom tool grammar"):
        chat_tools([tool])


def test_namespace_requires_a_wire_compatible_description():
    with pytest.raises(ValueError, match="description string"):
        chat_tools([{"type": "namespace", "name": "ns", "tools": [custom_tool()]}])
