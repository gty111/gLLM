"""Client-executed Responses tool discovery and continuation."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response import Response
from pydantic import TypeAdapter

from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.response_tools import chat_tools, custom_tool_formats, output_tool_call, tool_specs
from gllm.entrypoints.serving_responses import (
    _previous_output_to_input_items, make_chat_request, response_completion_generator,
    response_input_to_messages, response_stream_generator,
)
from gllm.tokenizers.reasoning import ThinkParser
from gllm.tokenizers.tool_parsers import QwenToolParser
from gllm.utils import StreamOutput


def search():
    return {"type": "tool_search", "execution": "client", "description": "Find tools.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"], "additionalProperties": False}}


def namespace(custom=False):
    tool = ({"type": "custom", "name": "lookup", "format": {"type": "text"}} if custom else
            {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}})
    return {"type": "namespace", "name": "catalog", "description": "Catalog tools.",
            "tools": [{**tool, "defer_loading": True}]}


def history(custom=False):
    return [{"role": "user", "content": "Look up a document."},
            {"type": "tool_search_call", "id": "ts_1", "call_id": "call_search",
             "execution": "client", "status": "completed", "arguments": {"query": "documents"}},
            {"type": "tool_search_output", "id": "tso_1", "call_id": "call_search",
             "execution": "client", "status": "completed", "tools": [namespace(custom)]}]


class Stream:
    def __init__(self, name, arguments, reasoning=False):
        self.text = ("<think>Find the appropriate tool.</think>" if reasoning else "")
        self.text += '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'
        self.seq = SimpleNamespace(token_ids=[1, 2, 99], raw_prompt_len=2,
                                   ignore_eos=False, finish_tokens=[99], output_len=128)

    async def __aiter__(self):
        for start in range(0, len(self.text), 7):
            yield StreamOutput(self.text[start:start + 7])


def generate(request, name, arguments, streaming, reasoning=False):
    async def run():
        args = (Stream(name, arguments, reasoning), request, make_chat_request(request),
                QwenToolParser(), ThinkParser() if reasoning else None)
        if not streaming:
            result = await response_completion_generator(*args)
            Response.model_validate(result)
            return result, []
        events = [json.loads(wire.split('data: ', 1)[1]) async for wire in response_stream_generator(*args)]
        for event in events:
            TypeAdapter(ResponseStreamEvent).validate_python(event)
        assert [e['sequence_number'] for e in events] == list(range(len(events)))
        return events[-1]['response'], events
    return asyncio.run(run())


def test_search_tool_without_name_becomes_callable_function():
    tools = [search(), namespace()]
    frozen = copy.deepcopy(tools)
    native = chat_tools(tools)
    assert [t['function']['name'] for t in native] == ['tool_search']
    assert native[0]['function']['parameters'] == search()['parameters']
    assert tools == frozen


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('reasoning', [False, True])
def test_search_call_wire_protocol(streaming, reasoning):
    req = ResponseRequest(model='test', input='Find a document tool.', tools=[search()], reasoning={'summary': 'none'})
    response, events = generate(req, 'tool_search', {'query': 'documents'}, streaming, reasoning)
    assert response['status'] == 'completed'
    assert len(response['output']) == 1
    item = response['output'][0]
    assert item['type'] == 'tool_search_call' and item['execution'] == 'client'
    assert item['arguments'] == {'query': 'documents'} and item['call_id']
    assert 'name' not in item
    assert _previous_output_to_input_items(response) == [item]
    assert not any(e['type'].startswith('response.function_call_arguments') for e in events)


@pytest.mark.parametrize('custom', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_search_output_loads_tools_and_preserves_history(custom, streaming):
    items = history(custom)
    frozen = copy.deepcopy(items)
    req = ResponseRequest(model='test', input=items, tools=[search()])
    native = chat_tools(req.tools, req.input)
    assert [t['function']['name'] for t in native] == ['tool_search', 'catalog.lookup']
    messages = response_input_to_messages(req)
    assert [m['role'] for m in messages] == ['user', 'assistant', 'tool']
    assert messages[1]['tool_calls'][0]['id'] == messages[2]['tool_call_id'] == 'call_search'
    assert json.loads(messages[2]['content']) == {'available_tools': ['catalog.lookup']}
    if custom:
        assert custom_tool_formats(req.tools, req.input) == {'catalog.lookup': {'type': 'text'}}
    response, _ = generate(req, 'catalog.lookup', {'input': 'query'} if custom else {}, streaming)
    assert response['status'] == 'completed'
    item = response['output'][0]
    assert item['type'] == ('custom_tool_call' if custom else 'function_call')
    assert item['name'] == 'lookup' and item['namespace'] == 'catalog'
    assert items == frozen


def test_loaded_tools_work_without_repeating_request_tools():
    req = ResponseRequest(model='test', input=history())
    assert make_chat_request(req).tools[0].function.name == 'catalog.lookup'
    response, _ = generate(req, 'catalog.lookup', {}, False)
    assert response['tool_choice'] == 'auto'


def test_repeated_discovery_and_declared_deferred_tools_do_not_duplicate():
    items = history()
    items.append(copy.deepcopy(items[-1]))
    assert list(tool_specs([search(), namespace()], items)) == ['tool_search', 'catalog.lookup']


def test_conflicting_discovered_definition_is_rejected():
    declared = namespace()
    declared['tools'][0]['description'] = 'Different contract'
    with pytest.raises(ValueError, match='Conflicting tool definition'):
        tool_specs([search(), declared], history())


@pytest.mark.parametrize('tool', [
    {**search(), 'execution': 'server'}, {**search(), 'parameters': None},
    {'type': 'namespace', 'name': 'nested', 'description': '', 'tools': [search()]},
])
def test_unsupported_search_definitions_fail_before_inference(tool):
    with pytest.raises(ValueError):
        chat_tools([tool])


@pytest.mark.parametrize('tools', [[search(), search()],
    [search(), {'type': 'function', 'name': 'tool_search'}],
    [{'type': 'function', 'name': 'tool_search'}, search()]])
def test_search_name_collisions_are_rejected(tools):
    with pytest.raises(ValueError, match='Ambiguous'):
        chat_tools(tools)


@pytest.mark.parametrize('updates,param', [
    ({'execution': 'server'}, 'input.2'), ({'call_id': None}, 'input.2.call_id'),
    ({'status': 'in_progress'}, 'input.2.status'), ({'tools': {}}, 'input.2.tools'),
    ({'tools': [search()]}, 'input.2.tools.0'),
])
def test_invalid_search_output_has_precise_error(updates, param):
    items = history()
    items[-1].update(updates)
    with pytest.raises(ValueError) as error:
        make_chat_request(ResponseRequest(model='test', input=items, tools=[search()]))
    assert error.value.args[0] == param


def test_undiscovered_tools_cannot_be_published():
    call = SimpleNamespace(id='c1', function=SimpleNamespace(name='catalog.lookup', arguments='{}'))
    with pytest.raises(ValueError, match='undeclared tool'):
        output_tool_call(call, tool_specs([search(), namespace()]))
