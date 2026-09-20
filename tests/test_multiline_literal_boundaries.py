"""A closing quote after blank lines must not leak reasoning or swallow calls."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.serving_chat import chat_completion_generator, chat_completion_stream_generator
from gllm.entrypoints.serving_responses import (
    make_chat_request, response_completion_generator, response_stream_generator,
)
from gllm.tokenizers.literals import LiteralScanner
from gllm.tokenizers.reasoning import ThinkParser
from gllm.tokenizers.tool_parsers import QwenToolParser, Qwen3ToolParser
from gllm.utils import StreamOutput


COMMAND = 'echo "quoted argument"'
JSON_CALL = '<tool_call>' + json.dumps({'name': 'run', 'arguments': {'cmd': COMMAND}}) + '</tool_call>'
XML_CALL = '<tool_call><function=run><parameter=cmd>' + COMMAND + '</parameter></function></tool_call>'
PARSERS = [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)]


def chunks(text):
    yield [text]
    yield list(text)
    for i in range(len(text) + 1):
        yield [text[:i], text[i:]]


def split(parts, native=False):
    p = ThinkParser(prefilled=True)
    reasoning, content = [], []
    # Annotate the reconstructed native markers independently of chunking.
    text = ''.join(parts)
    markers = [i for i in range(len(text)) if text.startswith('</think>', i)]
    offset = 0
    for part in parts:
        metadata = tuple((i-offset, '</think>') for i in markers
                         if offset <= i < offset+len(part)) if native else None
        r, c = p.feed(part, control_tokens=metadata)
        reasoning.append(r)
        content.append(c)
        offset += len(part)
    r, c = p.finish()
    return ''.join(reasoning)+r, ''.join(content)+c, p


@pytest.mark.parametrize('quote', ['"', "'"])
@pytest.mark.parametrize('gap', ['\n\n', '\n \t\n', '\r\n\r\n', '\n\n\n  '])
@pytest.mark.parametrize('native', [False, True])
def test_paired_quote_after_blank_lines_keeps_reasoning_private(quote, gap, native):
    thought = 'Example: ' + quote + '\n</think>' + gap + quote + '. Continue privately.'
    raw = thought + '</think>Answer.'
    for parts in chunks(raw):
        r, c, p = split(parts, native)
        assert (r, c, p.state) == (thought, 'Answer.', 'content')


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('native', [False, True])
def test_unmatched_outer_backtick_and_two_quote_fragments_before_real_call(parser_class, call, native):
    # The first quote closes after a blank line; the second quote is unmatched.
    # It must not borrow a quote from the real function's JSON/XML arguments.
    thought = 'The excludes list contains `["\n</think>\n\n", "\n\n<tool_call>\n\n'
    raw = thought + '</think>\n\n' + call
    for parts in chunks(raw):
        r, c, p = split(parts, native)
        assert (r, c, p.state) == (thought, '\n\n'+call, 'content')
        answer, calls = parser_class().parse(c)
        assert answer == '\n\n'
        assert len(calls) == 1
        assert json.loads(calls[0].function.arguments) == {'cmd': COMMAND}


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('quote', ['"', "'"])
def test_tool_examples_across_blank_line_then_real_quoted_call(parser_class, call, quote):
    literal_call = call.replace('\\', '\\\\').replace(quote, '\\'+quote)
    example = 'Example: ' + quote + literal_call + '\n\n' + quote + '\n\nRun:\n'
    raw = example + call
    for parts in chunks(raw):
        p = parser_class().stream_parser()
        total, content, calls = '', [], []
        for part in parts:
            total += part
            while (d := p.process(total)) is not None:
                content.append(d.content or '')
                calls.extend(d.tool_calls or [])
        while (d := p.process(total, final=True)) is not None:
            content.append(d.content or '')
            calls.extend(d.tool_calls or [])
        assert ''.join(content) == example
        assert len(calls) == 1
        assert json.loads(calls[0].function.arguments) == {'cmd': COMMAND}


@pytest.mark.parametrize('quote', ['"', "'"])
def test_quote_decision_waits_for_closer_without_leaking(quote):
    p = ThinkParser(prefilled=True)
    r, c = p.feed('Example: '+quote+'\n</think>\n\n')
    assert r == 'Example:' + ' '
    assert c == ''
    assert p.state == 'reasoning'
    assert p.feed(' \n  ') == ('', '')
    r, c = p.feed(quote+' Continue.</think>Answer.')
    assert r == quote+'\n</think>\n\n \n  '+quote+' Continue.'
    assert c == 'Answer.'


def test_blank_line_recovery_does_not_apply_to_closed_backtick_spans():
    # Retain paragraph recovery for Markdown backticks; changing every quote
    # type would let a stray backtick swallow the actual protocol boundary.
    raw = '`unmatched\n\nPlan.</think>Answer with `code`.'
    for parts in chunks(raw):
        r, c, p = split(parts)
        assert r == '`unmatched\n\nPlan.'
        assert c == 'Answer with `code`.'


def test_closing_quote_whitespace_lookahead_resumes_incrementally():
    class CountedText(str):
        reads = 0

        def __getitem__(self, key):
            type(self).reads += 1
            return super().__getitem__(key)

    scanner = LiteralScanner()
    text = '\"example\n\n'
    assert scanner.end(CountedText(text), 0) is None
    for _ in range(100):
        text += ' ' * 64
        assert scanner.end(CountedText(text), 0) is None
    text += '\"'
    assert scanner.end(CountedText(text), 0) == len(text)
    # Bound character work, not elapsed time or private cursor representation.
    # Rescanning all prior whitespace on each delta exceeds this by an order
    # of magnitude, while incremental lookahead reads each character once.
    assert CountedText.reads < 4 * len(text)


class Stream:
    def __init__(self, text, chunked, truncated=False):
        self.parts = list(text) if chunked else [text]
        self.seq = SimpleNamespace(token_ids=[1,2,3], raw_prompt_len=2,
            ignore_eos=False, finish_tokens=[99], output_len=1 if truncated else 10000)

    async def __aiter__(self):
        for part in self.parts:
            yield StreamOutput(part)


def collect(generator):
    async def run():
        return [json.loads(w.split('data: ',1)[1]) async for w in generator
                if 'data: [DONE]' not in w]
    return asyncio.run(run())


def request():
    return ResponseRequest(model='test', input='Run the command.', tools=[
        {'type':'function','name':'run','parameters':{'type':'object','properties':{'cmd':{'type':'string'}}}}])


@pytest.mark.parametrize('parser_class,call', PARSERS)
@pytest.mark.parametrize('chunked', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_responses_and_chat_keep_examples_in_reasoning_and_execute_real_call(parser_class, call, chunked, streaming):
    thought = 'The excludes list contains `["\n</think>\n\n", "\n\n<tool_call>\n\n'
    raw = '<think>'+thought+'</think>\n\n'+call
    req = request()
    args = lambda: (Stream(raw,chunked), req, make_chat_request(req), parser_class(), ThinkParser())
    if streaming:
        events = collect(response_stream_generator(*args()))
        for event in events:
            TypeAdapter(ResponseStreamEvent).validate_python(event)
        assert events[-1]['type'] == 'response.completed'
        result = events[-1]['response']
    else:
        result = asyncio.run(response_completion_generator(*args()))
    assert result['status'] == 'completed'
    assert [i['type'] for i in result['output']] == ['reasoning','message','function_call']
    assert result['output'][0]['content'][0]['text'] == thought
    assert result['output'][1]['content'][0]['text'] == '\n\n'
    assert json.loads(result['output'][2]['arguments']) == {'cmd':COMMAND}
    chat_args = (Stream(raw,chunked), make_chat_request(req), parser_class(), ThinkParser())
    if streaming:
        events = collect(chat_completion_stream_generator(*chat_args))
        assert events[-1]['choices'][0]['finish_reason'] == 'tool_calls'
        deltas = [e['choices'][0]['delta'] for e in events]
        assert ''.join(d.get('content','') for d in deltas) == '\n\n'
        assert ''.join(d.get('reasoning_content','') for d in deltas) == thought
        assert len([c for d in deltas for c in d.get('tool_calls',[])]) == 1
    else:
        result = asyncio.run(chat_completion_generator(*chat_args))
        assert result.choices[0].finish_reason == 'tool_calls'
        assert result.choices[0].message.reasoning_content == thought
        assert result.choices[0].message.content == '\n\n'
        assert len(result.choices[0].message.tool_calls) == 1


@pytest.mark.parametrize('truncated', [False, True])
def test_quote_closed_after_blank_lines_without_real_end_does_not_complete(truncated):
    raw = '<think>Example: "</think>\n\n"'
    req = request()
    events = collect(response_stream_generator(Stream(raw,True,truncated), req,
        make_chat_request(req), Qwen3ToolParser(), ThinkParser()))
    assert events[-1]['type'] == ('response.incomplete' if truncated else 'response.failed')
    assert not any(e['type']=='response.output_text.delta' for e in events)
