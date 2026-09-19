"""Tool syntax mentioned in an answer must not silently truncate that answer."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter

from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.serving_chat import (
    chat_completion_generator, chat_completion_stream_generator,
)
from gllm.entrypoints.serving_responses import (
    make_chat_request, response_completion_generator, response_stream_generator,
)
from gllm.tokenizers.reasoning import ThinkParser
from gllm.tokenizers.tool_parsers import QwenToolParser, Qwen3ToolParser, ToolParseError
from gllm.utils import StreamOutput


JSON_CALL = '<tool_call>{"name":"run","arguments":{"cmd":"echo ok"}}</tool_call>'
XML_CALL = '<tool_call>\n<function=run><parameter=cmd>echo ok</parameter></function>\n</tool_call>'
PARSERS = [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)]
LITERAL = 'Plain answers or tool markers (`<function=...>`, `<tool_call>`). More explanation follows.'


def drain(parser, chunks):
    text, deltas = '', []
    for chunk in chunks:
        text += chunk
        while (delta := parser.process(text)) is not None:
            deltas.append(delta)
    while (delta := parser.process(text, final=True)) is not None:
        deltas.append(delta)
    return deltas


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('example', [
    LITERAL, 'Use <tool_call> as an opening marker. More explanation.',
    'Literal <tool_call>', 'Literal <tool_ca',
    '`{call}` More explanation.', '``example `{call}` `` More explanation.',
    '```xml\n{call}\n```\nMore explanation.', '~~~xml\n{call}\n~~~\nMore explanation.',
    '````xml\n```\n{call}\n```\n````\nMore explanation.',
    '```xml\n```not-a-closer\n{call}\n```\nMore explanation.',
    '\\<tool_call> literal marker. More explanation.',
])
def test_literal_examples_survive_every_chunk_boundary(parser_class, call, example):
    text = example.replace('{call}', call)
    content, calls = parser_class().parse(text)
    assert content == text
    assert calls == []
    for boundary in range(len(text) + 1):
        deltas = drain(parser_class().stream_parser(), [text[:boundary], text[boundary:]])
        assert ''.join(d.content or '' for d in deltas) == text
        assert not any(d.tool_calls for d in deltas)
    deltas = drain(parser_class().stream_parser(), list(text))
    assert ''.join(d.content or '' for d in deltas) == text


@pytest.mark.parametrize('parser_class,call', PARSERS)
def test_examples_real_calls_and_following_prose_keep_order(parser_class, call):
    before = f'Example: `{call}`\nNow run it:\n'
    text = before + call + '\nBetween calls.\n' + call + '\nAfter calls.'
    content, calls = parser_class().parse(text)
    assert content == before + '\nBetween calls.\n\nAfter calls.'
    assert [c.function.name for c in calls] == ['run', 'run']
    for chunks in ([text], list(text)):
        deltas = drain(parser_class().stream_parser(), chunks)
        assert ''.join(d.content or '' for d in deltas) == content
        streamed_calls = [c for d in deltas for c in d.tool_calls or []]
        assert [c.index for c in streamed_calls] == [0, 1]
        assert len({c.id for c in streamed_calls}) == 2
        assert [json.loads(c.function.arguments) for c in streamed_calls] == [
            {'cmd': 'echo ok'}, {'cmd': 'echo ok'},
        ]


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('argument', [
    'echo <tool_call> and </tool_call>',
    'echo </function> and </parameter> as literal examples',
])
def test_tool_argument_containing_marker_is_not_split(parser_class, call, argument):
    call = call.replace('echo ok', argument)
    content, calls = parser_class().parse(call)
    assert not content
    assert json.loads(calls[0].function.arguments)['cmd'] == argument
    deltas = drain(parser_class().stream_parser(), list(call))
    streamed_calls = [c for d in deltas for c in d.tool_calls or []]
    assert len(streamed_calls) == 1
    assert json.loads(streamed_calls[0].function.arguments)['cmd'] == argument


class Stream:
    def __init__(self, text, chunked, truncated=False):
        self.chunks = list(text) if chunked else [text]
        self.seq = SimpleNamespace(token_ids=[1, 2, 3], raw_prompt_len=2,
                                   ignore_eos=False, finish_tokens=[99],
                                   output_len=1 if truncated else 256)

    async def __aiter__(self):
        for chunk in self.chunks:
            yield StreamOutput(chunk)


def request():
    return ResponseRequest(model='test', input='Explain the tool syntax.', tools=[
        {'type': 'function', 'name': 'run', 'parameters': {
            'type': 'object', 'properties': {'cmd': {'type': 'string'}}}},
    ])


def collect(generator):
    async def run():
        return [json.loads(wire.split('data: ', 1)[1]) async for wire in generator
                if 'data: [DONE]' not in wire]
    return asyncio.run(run())


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('chunked', [False, True])
@pytest.mark.parametrize('with_call', [False, True])
def test_responses_items_and_text_are_lossless(parser_class, call, chunked, with_call):
    body = LITERAL + ('\n' + call + '\nAfter tool.' if with_call else '')
    req = request()
    args = lambda: (Stream('<think>Explain.</think>' + body, chunked), req,
                    make_chat_request(req), parser_class(), ThinkParser())
    events = collect(response_stream_generator(*args()))
    adapter = TypeAdapter(ResponseStreamEvent)
    for event in events:
        adapter.validate_python(event)
    assert events[-1]['type'] == 'response.completed'
    response = events[-1]['response']
    Response.model_validate(response)
    expected = LITERAL + ('\n\nAfter tool.' if with_call else '')
    assert ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta') == expected
    assert ''.join(p['text'] for i in response['output'] if i['type'] == 'message'
                   for p in i['content']) == expected
    added = [e['item']['id'] for e in events if e['type'] == 'response.output_item.added']
    done = [e['item']['id'] for e in events if e['type'] == 'response.output_item.done']
    assert added == done == [i['id'] for i in response['output']]
    if with_call:
        assert [i['type'] for i in response['output']] == [
            'reasoning', 'message', 'function_call', 'message',
        ]
    result = asyncio.run(response_completion_generator(*args()))
    assert result['status'] == 'completed'
    assert ''.join(p['text'] for i in result['output'] if i['type'] == 'message'
                   for p in i['content']) == expected


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('streaming', [False, True])
def test_chat_preserves_literal_examples(parser_class, call, streaming):
    req = make_chat_request(request())
    args = (Stream(LITERAL, True), req, parser_class())
    if streaming:
        events = collect(chat_completion_stream_generator(*args))
        text = ''.join(e['choices'][0]['delta'].get('content', '') for e in events)
        assert events[-1]['choices'][0]['finish_reason'] == 'stop'
    else:
        result = asyncio.run(chat_completion_generator(*args))
        text = result.choices[0].message.content
    assert text == LITERAL


@pytest.mark.parametrize('parser_class,body', [
    (QwenToolParser, '<tool_call>{"name":"run"'),
    (QwenToolParser, '<tool_call>{bad json}</tool_call>'),
    (QwenToolParser, '<tool_call>{"arguments":{}}</tool_call>'),
    (Qwen3ToolParser, '<tool_call><function=run><parameter=cmd>echo ok'),
    (Qwen3ToolParser, '<tool_call><function='),
    (Qwen3ToolParser, '<tool_call><function=run></function>'),
    (Qwen3ToolParser, '<tool_call><function=run>bad parameters</function></tool_call>'),
])
@pytest.mark.parametrize('truncated', [False, True])
def test_malformed_real_calls_do_not_report_success(parser_class, body, truncated):
    with pytest.raises(ToolParseError):
        parser_class().parse(body)
    req = request()
    args = lambda: (Stream('<think>Plan.</think>Before.\n' + body, True, truncated),
                    req, make_chat_request(req), parser_class(), ThinkParser())
    events = collect(response_stream_generator(*args()))
    expected = 'incomplete' if truncated else 'failed'
    assert events[-1]['type'] == 'response.' + expected
    result = asyncio.run(response_completion_generator(*args()))
    for response in (events[-1]['response'], result):
        Response.model_validate(response)
        assert response['status'] == expected
        if truncated:
            assert response['incomplete_details']['reason'] == 'max_output_tokens'
        else:
            assert response['error']['code'] == 'server_error'


@pytest.mark.parametrize('parser_class,call', PARSERS)
def test_chat_stream_reports_tool_parse_error(parser_class, call):
    body = call[:-len('</tool_call>')]
    events = collect(chat_completion_stream_generator(
        Stream(body, True), make_chat_request(request()), parser_class(),
    ))
    assert events[-1]['error']['code'] == 'invalid_tool_output'
    assert not any(e.get('choices', [{}])[0].get('finish_reason') == 'stop' for e in events)
