"""Cross-parser regressions: quoted protocol markers are data, not actions."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.serving_responses import make_chat_request, response_stream_generator
from gllm.runtime.sequence import GenerationSequence
from gllm.tokenizers.reasoning import ThinkParser, decode_stream_delta, reasoning_control_tokens
from gllm.tokenizers.tool_parsers import QwenToolParser, Qwen3ToolParser
from gllm.utils import StreamOutput


JSON_CALL = '<tool_call>{"name":"run","arguments":{"cmd":"echo ok"}}</tool_call>'
XML_CALL = '<tool_call><function=run><parameter=cmd>echo ok</parameter></function></tool_call>'


def split(text, chunks, prefilled=False):
    parser = ThinkParser(prefilled=prefilled)
    pieces = [parser.feed(c) for c in chunks] + [parser.finish()]
    return ''.join(r for r, _ in pieces), ''.join(c for _, c in pieces), parser


@pytest.mark.parametrize('literal', [
    '`</think>`', '``one ` </think> ``', '"</think>"', "'</think>'",
    '"escaped \\" quote and </think>"', '\\</think>',
    '```xml\n</think>\n```', '~~~xml\n</think>\n~~~',
    '````xml\n```\n</think>\n```\n````',
])
@pytest.mark.parametrize('prefilled', [False, True])
def test_reasoning_literals_at_every_boundary(literal, prefilled):
    thought = 'Discuss:\n' + literal + '\nKeep this private.'
    text = ('' if prefilled else '<think>') + thought + '</think>Answer.'
    for i in range(len(text) + 1):
        r, c, p = split(text, [text[:i], text[i:]], prefilled)
        assert (r, c, p.state) == (thought, 'Answer.', 'content')
    assert split(text, list(text), prefilled)[:2] == (thought, 'Answer.')


@pytest.mark.parametrize('tail', ['`</think>`', '"</think>"', '```\n</think>', '~~~\n</think>', '\\</think>'])
def test_eof_inside_literal_is_not_successful_reasoning_close(tail):
    text = '<think>Still thinking:\n' + tail
    r, c, parser = split(text, list(text))
    assert r == 'Still thinking:\n' + tail
    assert c == ''
    assert parser.state == 'reasoning'


def drain(parser, chunks):
    text, content, calls = '', [], []
    for chunk in chunks:
        text += chunk
        while (d := parser.process(text)) is not None:
            content.append(d.content or '')
            calls.extend(d.tool_calls or [])
    while (d := parser.process(text, final=True)) is not None:
        content.append(d.content or '')
        calls.extend(d.tool_calls or [])
    return ''.join(content), calls


@pytest.mark.parametrize('parser_class,call', [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)])
@pytest.mark.parametrize('prefix', ['` unmatched\n\nRun:\n', '`` unmatched\n \t\nRun:\n', "I'll run this:\n", '` unmatched '])
def test_unmatched_inline_delimiters_do_not_swallow_real_calls(parser_class, call, prefix):
    text = prefix + call
    for i in range(len(text) + 1):
        content, calls = drain(parser_class().stream_parser(), [text[:i], text[i:]])
        assert content == prefix
        assert [c.function.name for c in calls] == ['run']


@pytest.mark.parametrize('parser_class,call', [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)])
@pytest.mark.parametrize('wrapper', ['`{call}`', '"{call}"', "'{call}'", '```\n{call}', '~~~\n{call}'])
def test_examples_never_execute_even_at_eof(parser_class, call, wrapper):
    # JSON inside a quoted string must escape its quotes.
    value = call.replace('"', '\\"') if wrapper.startswith('"') else call
    text = wrapper.replace('{call}', value)
    content, calls = drain(parser_class().stream_parser(), list(text))
    assert content == text
    assert not calls


class Stream:
    def __init__(self, text, chunked):
        self.parts = list(text) if chunked else [text]
        self.seq = SimpleNamespace(token_ids=[1, 2, 3], raw_prompt_len=2,
            ignore_eos=False, finish_tokens=[99], output_len=10000)

    async def __aiter__(self):
        for text in self.parts:
            yield StreamOutput(text)


@pytest.mark.parametrize('chunked', [False, True])
@pytest.mark.parametrize('parser_class,call', [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)])
def test_reasoning_example_then_real_tool_is_not_a_completed_text_answer(chunked, parser_class, call):
    thought = 'The first token could be `</think>` in an example.\n\nContinue planning privately.'
    req = ResponseRequest(model='test', input='Run the command.', tools=[
        {'type':'function', 'name':'run', 'parameters':{'type':'object', 'properties':{'cmd':{'type':'string'}}}}])
    async def run():
        return [json.loads(w.split('data: ', 1)[1]) async for w in response_stream_generator(
            Stream('<think>' + thought + '</think>Run it.\n' + call, chunked),
            req, make_chat_request(req), parser_class(), ThinkParser())]
    events = asyncio.run(run())
    assert events[-1]['type'] == 'response.completed'
    output = events[-1]['response']['output']
    assert [i['type'] for i in output] == ['reasoning', 'message', 'function_call']
    assert output[0]['content'][0]['text'] == thought
    assert output[1]['content'][0]['text'] == 'Run it.\n'
    assert output[2]['name'] == 'run'
    assert json.loads(output[2]['arguments']) == {'cmd':'echo ok'}


def test_native_metadata_distinguishes_plain_marker_text():
    p = ThinkParser(prefilled=True)
    first = 'We wait until </think>, then run the grammar. '
    r, c = p.feed(first, control_tokens=())
    assert (r, c) == (first, '')
    r, c = p.feed('</think>Answer.', control_tokens=((0, '</think>'),))
    assert (r, c) == ('', 'Answer.')


def test_native_end_inside_code_is_literal_even_with_metadata():
    p = ThinkParser(prefilled=True)
    text = 'Discuss `</think>` first.\n</think>Answer.'
    first, last = text.index('</think>'), text.rindex('</think>')
    r, c = p.feed(text, control_tokens=((first, '</think>'), (last, '</think>')))
    assert (r, c) == (text[:last], 'Answer.')


class ToyTokenizer:
    """Native tags can be hidden, while ordinary bytes spell identical tags."""
    def convert_tokens_to_ids(self, text):
        return {'<think>':1000, '</think>':1001}.get(text)

    def decode(self, ids, skip_special_tokens=True):
        result = ''
        for token in ids:
            if token in (1000,1001):
                if not skip_special_tokens:
                    result += '<think>' if token == 1000 else '</think>'
            else:
                result += chr(token)
        return result


@pytest.mark.parametrize('batch_size', [1, 2, 4, 1000])
def test_mtp_control_offsets_and_hidden_special_tokens(batch_size):
    tok = ToyTokenizer()
    controls = reasoning_control_tokens(tok)
    thought = 'Literal </think>, still private.'
    tokens = [1000] + list(map(ord, thought)) + [1001] + list(map(ord, 'Answer.'))
    seq = GenerationSequence('test', [65], [], 1000)
    p = ThinkParser();r=[];c=[]
    for offset in range(0, len(tokens), batch_size):
        text, markers = decode_stream_delta(seq, tok, tokens[offset:offset+batch_size], controls)
        rr, cc = p.feed(text, control_tokens=markers);r.append(rr);c.append(cc)
    rr,cc=p.finish();r.append(rr);c.append(cc)
    assert ''.join(r) == thought
    assert ''.join(c) == 'Answer.'
    assert seq.token_ids == [65] + tokens


@pytest.fixture(scope='module')
def real_tokenizer():
    path = os.environ.get('GLLM_TEST_QWEN_TOKENIZER')
    if not path:
        pytest.skip('Set GLLM_TEST_QWEN_TOKENIZER to validate checkpoint tokenization.')
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize('batch_size', [1, 4, 10000])
def test_checkpoint_unicode_mtp_and_literal_native_tags(real_tokenizer, batch_size):
    tok = real_tokenizer
    thought = 'Discuss `</think>` with caf\u00e9 and \U0001f680.\nKeep planning.'
    answer = 'Ready: na\u00efve \U0001f680.'
    raw = '<think>' + thought + '</think>' + answer
    tokens = tok.encode(raw, add_special_tokens=False)
    seq = GenerationSequence('test', tok.encode('Prompt\n', add_special_tokens=False), [], 10000)
    p = ThinkParser();r=[];c=[]
    controls = reasoning_control_tokens(tok)
    assert controls
    for i in range(0, len(tokens), batch_size):
        text, markers = decode_stream_delta(seq, tok, tokens[i:i+batch_size], controls)
        rr, cc = p.feed(text, control_tokens=markers);r.append(rr);c.append(cc)
    rr,cc=p.finish();r.append(rr);c.append(cc)
    assert ''.join(r) == thought
    assert ''.join(c) == answer


@pytest.mark.parametrize('fence', ['```', '~~~'])
def test_fence_can_start_immediately_after_reasoning_opener(fence):
    thought = fence + '\n</think>\n' + fence + '\nPlan.'
    raw = '<think>' + thought + '</think>Answer.'
    assert split(raw, list(raw))[:2] == (thought, 'Answer.')


@pytest.mark.parametrize('parser_class,call', [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)])
def test_closed_multiline_code_and_quotes_preserve_examples_then_call(parser_class, call):
    example = '`example\n' + call + '\n`\n\n'
    example += '"Escaped quote: \\" and an ordinary marker <tool_call>."\n'
    raw = example + call
    content, calls = drain(parser_class().stream_parser(), list(raw))
    assert content == example
    assert len(calls) == 1


def test_literal_control_without_native_token_does_not_start_reasoning():
    p = ThinkParser()
    assert p.feed('<think>literal example</think>', control_tokens=()) == ('', '<think>literal example</think>')
    assert not p.started


def test_pending_utf8_is_not_lost_across_a_native_boundary():
    class ByteTokenizer(ToyTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            data = bytearray()
            for token in ids:
                if token in (1000, 1001):
                    if not skip_special_tokens:
                        data.extend(('<think>' if token == 1000 else '</think>').encode())
                else:
                    data.append(token)
            return data.decode('utf-8', errors='replace')
    tok = ByteTokenizer()
    seq = GenerationSequence('utf8', [65], [], 1000)
    controls = reasoning_control_tokens(tok)
    text, markers = decode_stream_delta(seq, tok, [0xC3], controls)
    assert text == ''
    text, markers = decode_stream_delta(seq, tok, [0xA9, 1001, 65], controls)
    assert text == '\u00e9</think>A'
    assert markers == ((1, '</think>'),)


def test_engine_keeps_control_metadata_logprobs_and_final_token_accounting():
    from gllm.engine.llm import LLM
    tok = ToyTokenizer()
    seq = GenerationSequence('test', [65], [99], 1000)
    emitted, finished = [], []
    engine = object.__new__(LLM)
    engine.model_runner = SimpleNamespace(tokenizer=tok)
    engine._reasoning_controls = reasoning_control_tokens(tok)
    engine.running_maps = {'test':seq}
    engine.async_streams = {'test':SimpleNamespace(put=emitted.append, finish=lambda:finished.append(True))}
    engine._make_logprob_entry = lambda token, value: {'token_id':token, 'value':value}
    engine._make_prompt_logprobs = lambda value:value
    engine.free_finish_ids = lambda ids:None
    package = SimpleNamespace(act_schedule_ids=['stale','test'],
        next_tokens=[[65],[1000,80,1001,65]], logprobs=[None,7],
        prompt_logprobs={'test':[1,2]}, free_ids=['test'])
    engine._apply_ipc_package(package)
    assert seq.token_ids == [65,1000,80,1001,65]
    assert len(emitted) == 1
    assert emitted[0].text == '<think>P</think>A'
    assert emitted[0].control_tokens == ((0,'<think>'),(8,'</think>'))
    assert emitted[0].logprob == {'token_id':65,'value':7}
    assert emitted[0].prompt_logprobs == [1,2]
    assert finished == [True]
    assert not engine.running_maps and not engine.async_streams


@pytest.mark.parametrize('budget_exhausted', [False, True])
def test_truncated_reasoning_fence_reports_failure_or_incomplete(budget_exhausted):
    req = ResponseRequest(model='test', input='Explain this marker.')
    source = Stream('<think>Example:\n```xml\n</think>', True)
    if budget_exhausted:
        source.seq.output_len = 1
    async def run():
        return [json.loads(w.split('data: ',1)[1]) async for w in response_stream_generator(
            source, req, make_chat_request(req), reasoning_parser=ThinkParser())]
    events = asyncio.run(run())
    assert events[-1]['type'] == ('response.incomplete' if budget_exhausted else 'response.failed')
    assert not any(e['type']=='response.output_text.delta' for e in events)


@pytest.mark.parametrize('parser_class,call', [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)])
def test_deterministic_random_partitions_preserve_combined_grammar(parser_class, call):
    import random
    randomizer = random.Random(137)
    thought = 'Contractions do not open quotes: it\'s safe.\n```xml\n</think>\n```\nUse `</think>` literally.'
    before = 'Example: `' + call + '`\n\n` unmatched\n\nRun it:\n'
    body = before + call + '\nFinished.'
    raw = '<think>' + thought + '</think>' + body
    for _ in range(30):
        boundaries = sorted({0, len(raw), *(randomizer.randrange(len(raw)+1) for _ in range(25))})
        chunks = [raw[a:b] for a,b in zip(boundaries,boundaries[1:])]
        p=ThinkParser();content=[];reasoning=[]
        for chunk in chunks:
            r,c=p.feed(chunk);reasoning.append(r);content.append(c)
        r,c=p.finish();reasoning.append(r);content.append(c)
        text,calls=drain(parser_class().stream_parser(),content)
        assert ''.join(reasoning) == thought
        assert text == before + '\nFinished.'
        assert len(calls) == 1


@pytest.mark.parametrize('parser_class,call', [(QwenToolParser, JSON_CALL), (Qwen3ToolParser, XML_CALL)])
@pytest.mark.parametrize('wrapper', ['> {call}\n', '   > {call}\n', '    {call}\n', '\t{call}\n', '<!-- {call} -->', '<!-- {call}'])
def test_block_examples_do_not_execute(parser_class, call, wrapper):
    example = wrapper.replace('{call}',call)
    for i in range(len(example)+1):
        text,calls=drain(parser_class().stream_parser(),[example[:i],example[i:]])
        assert text == example
        assert calls == []


@pytest.mark.parametrize('example', ['> </think>\n', '    </think>\n', '<!-- </think> -->\n'])
def test_reasoning_block_examples_do_not_end_thinking(example):
    thought = 'Example:\n'+example+'Continue.'
    raw = '<think>'+thought+'</think>Answer.'
    assert split(raw,list(raw))[:2] == (thought,'Answer.')
