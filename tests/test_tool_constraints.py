"""Tool and custom-input constraints, exercised without model inference."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import torch

import gllm.structured_output as so
from gllm.entrypoints.response_tools import chat_tools, custom_tool_formats, output_tool_call, tool_specs
from gllm.tokenizers.tool_parsers import Qwen3ToolParser
from gllm.tool_constraints import build_tool_tag
from test_structured_output import backend, emit, seq
from test_reasoning_guard import ByteTokenizer

PATCH_GRAMMAR = r'''start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?
filename: /(.+)/
add_line: "+" /(.*)/ LF -> line
change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF
%import common.LF
'''
PATCH = '*** Begin Patch\n*** Add File: example.py\n+print("hello")\n+\n*** End Patch\n'


def custom(definition=PATCH_GRAMMAR, syntax="lark", name="apply_patch"):
    return {"type": "custom", "name": name,
            "format": {"type": "grammar", "syntax": syntax, "definition": definition}}


def function():
    return {"type": "function", "function": {"name": "run", "parameters": {
        "type": "object", "properties": {"count": {"type": "integer", "enum": [1, 2]}},
        "required": ["count"], "additionalProperties": False}}}


def xml(value, name="apply_patch"):
    return f'<tool_call>\n<function={name}>\n<parameter=input>\n{value}\n</parameter>\n</function>\n</tool_call>'


def tag(tools=None, **kwargs):
    tools = tools or [custom()]
    return build_tool_tag(chat_tools(tools), "qwen3", custom_formats=custom_tool_formats(tools), **kwargs)


def matcher(spec):
    import xgrammar as xgr
    info = xgr.TokenizerInfo([chr(i) for i in range(128)] + ["<eos>", "<think>", "</think>"],
                            vocab_type=xgr.VocabType.RAW, stop_token_ids=[128])
    return xgr.GrammarMatcher(xgr.GrammarCompiler(info, max_threads=1).compile_structural_tag(spec))


@pytest.mark.parametrize("parser", ["qwen", "qwen3"])
def test_function_name_schema_and_complete_envelope(parser):
    spec = build_tool_tag([function()], parser)
    call = ('<tool_call>\n{"name": "run", "arguments": {"count": 1}}\n</tool_call>'
            if parser == "qwen" else '<tool_call>\n<function=run>\n<parameter=count>1</parameter>\n</function>\n</tool_call>')
    assert matcher(spec).accept_string("I will run it.\n" + call)
    assert matcher(spec).accept_string("An ordinary answer.")
    assert not matcher(spec).accept_string(call.replace("run", "unknown"))
    assert not matcher(spec).accept_string(call.replace("1", "9"))
    assert not matcher(spec).accept_string('<tool_call>{"name":"unknown"}')
    pending = matcher(spec)
    assert pending.accept_string(call[:-len('</tool_call>')])
    assert not pending.accept_token(128)
    complete = matcher(spec)
    assert complete.accept_string(call)
    assert complete.accept_token(128)


@pytest.mark.parametrize("patch", [PATCH,
    '*** Begin Patch\n*** Delete File: old.py\n*** End Patch',
    '*** Begin Patch\n*** Update File: old.py\n*** Move to: new.py\n@@\n-old\n+new\n*** End of File\n*** End Patch\n',
    PATCH.replace('print("hello")', 'print("\\\\\\\"</think><parameter=x></parameter></function></tool_call>")'),
])
def test_valid_patch_roundtrips_through_grammar_and_parser(patch):
    spec = tag()
    call = xml(patch)
    assert matcher(spec).accept_string(call)
    parser = Qwen3ToolParser(custom_formats=custom_tool_formats([custom()]))
    content, calls = parser.parse(call, chat_tools([custom()]))
    assert not content and len(calls) == 1
    item = output_tool_call(calls[0], tool_specs([custom()]))
    assert item["input"] == patch


@pytest.mark.parametrize("patch", [
    PATCH.replace("*** Begin Patch", "*** Begin Patching"),
    PATCH.replace('+print', 'print'),
    PATCH.replace('*** End Patch', '*** Done'),
    '```diff\n' + PATCH + '```',
    '{"input":' + json.dumps(PATCH) + '}',
])
def test_invalid_patch_rejected_during_generation(patch):
    assert not matcher(tag()).accept_string(xml(patch))


def test_patch_masks_illegal_token_before_sampling(backend):
    s = seq(so.StructuredOutput(None, tag=tag()))
    prefix = '<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** Add File: example.py\n'
    for token in map(ord, prefix):
        emit(backend, [s], [token])
    logits = torch.zeros(1, 131)
    logits[0, ord('p')] = 100  # Model would omit the required '+' without the mask.
    logits[0, ord('+')] = 10
    active = backend.mask(logits, [s])
    assert torch.isneginf(logits[0, ord('p')])
    assert torch.isneginf(logits[0, 128])
    assert logits.argmax(-1).item() == ord('+')
    assert active


def test_custom_regex_whitespace_and_namespace_preserved():
    tools = [{"type": "namespace", "name": "editor", "description": "Edits",
              "tools": [custom('start: "  value\\n "', name="raw")]}]
    value = "  value\n "
    call = xml(value, "editor.raw")
    assert matcher(tag(tools)).accept_string(call)
    parser = Qwen3ToolParser(custom_formats=custom_tool_formats(tools))
    item = output_tool_call(parser.parse(call, chat_tools(tools))[1][0], tool_specs(tools))
    assert item["input"] == value and item["namespace"] == "editor"
    spec = tag([custom(r"[a-z]{2}[0-9]{2}", "regex")])
    assert matcher(spec).accept_string(xml("ab12"))
    assert not matcher(spec).accept_string(xml("ab1x"))


def test_schema_and_tools_are_both_enforced():
    spec = tag(schema='{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}')
    assert matcher(spec).accept_string('{"ok":true}')
    assert matcher(spec).accept_string(xml(PATCH))
    assert matcher(spec).accept_string(xml(PATCH) + '\n' + xml(PATCH))
    assert matcher(spec).accept_string('\n\n' + xml(PATCH) + '\n')
    assert not matcher(spec).accept_string('{"ok":false}')
    assert not matcher(spec).accept_string('Unstructured answer')


def test_unsupported_custom_envelope_fails_closed():
    for parser in ("qwen", "kimi", None):
        with pytest.raises(ValueError, match="Qwen"):
            build_tool_tag(chat_tools([custom()]), parser, custom_formats=custom_tool_formats([custom()]))


def test_streaming_parser_preserves_input_at_every_character():
    from gllm.entrypoints.serving_responses import response_stream_generator, make_chat_request
    from gllm.entrypoints.protocol import ResponseRequest
    from gllm.utils import StreamOutput
    req = ResponseRequest(model="test", input="Edit", tools=[custom()])
    class Stream:
        seq = SimpleNamespace(token_ids=[1, 2], raw_prompt_len=1, ignore_eos=False,
                              finish_tokens=[99], output_len=1000)
        async def __aiter__(self):
            for char in xml(PATCH): yield StreamOutput(char)
    async def run():
        return [json.loads(next(line[6:] for line in event.splitlines() if line.startswith('data: ')))
                async for event in response_stream_generator(Stream(), req, make_chat_request(req), Qwen3ToolParser())]
    events = asyncio.run(run())
    assert events[-1]["type"] == "response.completed"
    calls = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert len(calls) == 1 and calls[0]["input"] == PATCH


def test_literal_complete_closing_sequence_inside_custom_input():
    value = 'first\n</parameter>\n</function>\n</tool_call>\nlast'
    tools = [custom('start: ' + json.dumps(value))]
    text = xml(value)
    assert matcher(tag(tools)).accept_string(text)
    parser = Qwen3ToolParser(custom_formats=custom_tool_formats(tools))
    item = output_tool_call(parser.parse(text, chat_tools(tools))[1][0], tool_specs(tools))
    assert item['input'] == value


def test_no_parallel_calls_after_first():
    spec = tag(parallel_tool_calls=False)
    assert matcher(spec).accept_string(xml(PATCH))
    assert not matcher(spec).accept_string(xml(PATCH) + '\n' + xml(PATCH))


def test_speculative_tool_mask_and_rewind(backend):
    s = seq(so.StructuredOutput(None, tag=tag()))
    prefix = '<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** Add File: x\n'
    for token in map(ord, prefix): emit(backend, [s], [token])
    logits = torch.zeros(3, 131)
    context = [1] + list(map(ord, prefix[:-1]))
    active = backend.mask_speculative(logits, [s], [context], [[ord('\n'), ord('p'), ord('!')]])
    assert torch.isneginf(logits[0, ord('p')])
    assert torch.isneginf(logits[0, 128])
    assert logits[0, ord('+')] == 0
    assert torch.isfinite(logits[1:]).all()
    backend.commit_speculative(active, [[ord('\n')]], [ord('+')])
    assert backend.states[s].history == list(map(ord, prefix))
    # A preemption rewinds to a point before entering the tool grammar.
    logits = torch.zeros(1, 131)
    backend.mask_speculative(logits, [s], [[1]], [[ord('H')]])
    assert backend.states[s].history == []
    assert torch.isfinite(logits[0, ord('i')])


def test_tool_grammar_keeps_literal_aware_reasoning_guard(monkeypatch):
    import xgrammar as xgr
    from gllm.runtime.sequence import GenerationSequence
    tok = ByteTokenizer()
    info = xgr.TokenizerInfo([chr(i) for i in range(256)] + ['<eos>', '<think>', '</think>'],
                            vocab_type=xgr.VocabType.RAW, stop_token_ids=[256])
    ctx = xgr.GrammarCompiler(info, max_threads=1)
    monkeypatch.setattr(so, 'compiler', lambda *args: ctx)
    spec = so.prepare_output(None, tok, 259, [256], tok.encode('<think>'),
                             tools=chat_tools([custom()]), parser_name='qwen3',
                             custom_formats=custom_tool_formats([custom()]))
    s = GenerationSequence(1, [1], [256], 1000, structured_output=spec)
    s.computed_token_num = 0
    s.to_compute_token_num = 1
    sampler = so.StructuredSampler(tok)
    reasoning = 'Example: "</think>". Still thinking.'
    def sample(token):
        logits = torch.zeros(1, 259)
        active = sampler.mask(logits, [s])
        assert not torch.isneginf(logits[0, token])
        sampler.record(active, torch.tensor([token]))
        s.computed_token_num += 1
    for token in tok.encode(reasoning): sample(token)
    sample(258)
    for token in tok.encode(xml(PATCH)): sample(token)
    sample(256)
    assert sampler.states[s].guard.parser.state == 'content'
    # Rewind inside reasoning, including a literal closing marker.
    state = sampler.states[s]
    state.synchronize(tok.encode(reasoning))
    mask = torch.full((1, (259 + 31) // 32), -1, dtype=torch.int32)
    state.fill_mask(mask, 0)
    assert not (int(mask[0, 256 // 32]) & (1 << (256 % 32)))
    assert state.guard.parser.state != 'content'


@pytest.mark.parametrize('endpoint', ['chat', 'responses'])
@pytest.mark.parametrize('choice', ['auto', 'none'])
def test_api_wiring_reaches_engine(monkeypatch, backend, endpoint, choice):
    from gllm.entrypoints import api_server
    from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest
    captured = []
    async def add_requests(*args, **kwargs):
        captured.append(kwargs['structured_output'])
        return SimpleNamespace()
    async def completed(*args):
        result = {'id': 'test', 'status': 'completed', 'output': []}
        return SimpleNamespace(model_dump=lambda **kwargs: result) if endpoint == 'chat' else result
    runner = SimpleNamespace(encode=lambda *a, **kw: [1], extract_modify_mm=lambda *a: None,
                             tokenizer=None, model_loader=SimpleNamespace(vocab_size=131))
    monkeypatch.setattr(api_server, 'llm', SimpleNamespace(
        model_path='test', model_runner=runner, finish_tokens=[128],
        check_seq_length=lambda *a: True, add_requests_async=add_requests))
    monkeypatch.setattr(api_server, 'tool_parser', Qwen3ToolParser())
    monkeypatch.setattr(api_server, 'chat_completion_generator', completed)
    monkeypatch.setattr(api_server, 'response_completion_generator', completed)
    raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    if endpoint == 'responses':
        request = ResponseRequest(model='test', input='Edit', tools=[custom()], tool_choice=choice, store=False)
        result = asyncio.run(api_server.create_response(request, raw))
    else:
        request = ChatCompletionRequest(model='test', messages=[{'role':'user','content':'Run'}],
                                        tools=[function()], tool_choice=choice)
        result = asyncio.run(api_server.create_chat_completion(request, raw))
    assert result.status_code == 200
    if choice == 'none': assert captured == [None]
    else:
        assert captured[0].tag
        if endpoint == 'responses':
            assert not matcher(captured[0].tag).accept_string(xml(PATCH.replace('+print', 'print')))


def test_grammar_error_reports_position_without_echoing_patch():
    from gllm.entrypoints.protocol import FunctionCall, ToolCall
    value = PATCH.replace('+print', 'print')
    with pytest.raises(ValueError, match=r'line 3, column 1') as error:
        output_tool_call(ToolCall(function=FunctionCall(name='apply_patch', arguments=json.dumps({'input':value}))),
                         tool_specs([custom()]))
    assert 'print(' not in str(error.value)


@pytest.mark.skipif(not os.environ.get('GLLM_TEST_QWEN_TOKENIZER'), reason='Real tokenizer path not provided')
def test_real_qwen_tokens_and_mask_roundtrip():
    import xgrammar as xgr
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ['GLLM_TEST_QWEN_TOKENIZER'], local_files_only=True)
    # Include multilingual text, JSON escapes and native marker token IDs.
    patch = PATCH.replace('hello', '\u4f60\u597d\U0001f600\\path').replace('+\n', '+literal </think> and <tool_call>\n')
    text = xml(patch)
    ctx = so.compiler(tokenizer, len(tokenizer), (tokenizer.eos_token_id,))
    compiled = ctx.compile_structural_tag(tag())
    m = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, len(tokenizer))
    for token in tokenizer.encode(text, add_special_tokens=False):
        m.fill_next_token_bitmask(mask)
        assert int(mask[0, token // 32]) & (1 << (token % 32)), tokenizer.decode([token])
        assert m.accept_token(token)
    assert m.accept_token(tokenizer.eos_token_id)
    parser = Qwen3ToolParser(custom_formats=custom_tool_formats([custom()]))
    item = output_tool_call(parser.parse(text, chat_tools([custom()]))[1][0], tool_specs([custom()]))
    assert item['input'] == patch


@pytest.mark.parametrize('value', [
    'echo <parameter=x>test',
    'print("</function></tool_call>")',
    'print("<tool_call><function=other>")',
])
def test_constrained_function_arguments_preserve_literal_tags(value):
    tools = [{'type':'function','function':{'name':'run','parameters':{
        'type':'object','properties':{'cmd':{'type':'string'}},
        'required':['cmd'],'additionalProperties':False}}}]
    text = '<tool_call>\n<function=run>\n<parameter=cmd>' + value + '</parameter>\n</function>\n</tool_call>'
    assert matcher(build_tool_tag(tools, 'qwen3')).accept_string(text)
    parser = Qwen3ToolParser().stream_parser(tools)
    result = []
    for end in range(1, len(text)+1):
        delta = parser.process(text[:end])
        if delta and delta.tool_calls: result.extend(delta.tool_calls)
    delta = parser.process(text, final=True)
    if delta and delta.tool_calls: result.extend(delta.tool_calls)
    assert len(result) == 1
    assert json.loads(result[0].function.arguments) == {'cmd':value}


@pytest.mark.parametrize('prefix', ['', '<think>', '<think>\n</think>\n'])
def test_tool_constraints_remain_active_for_all_thinking_modes(monkeypatch, prefix):
    import xgrammar as xgr
    tok = ByteTokenizer()
    info = xgr.TokenizerInfo([chr(i) for i in range(256)] + ['<eos>', '<think>', '</think>'],
                            vocab_type=xgr.VocabType.RAW, stop_token_ids=[256])
    ctx = xgr.GrammarCompiler(info, max_threads=1)
    monkeypatch.setattr(so, 'compiler', lambda *args: ctx)
    spec = so.prepare_output(None, tok, 259, [256], tok.encode(prefix),
                             tools=chat_tools([custom()]), parser_name='qwen3',
                             custom_formats=custom_tool_formats([custom()]))
    assert spec.tag is not None
    state = so._State(ctx.compile_structural_tag(spec.tag), spec, [256], tok)
    if spec.thinking:
        tokens = tok.encode('Ready.</think>')
    elif spec.allow_think_start:
        tokens = tok.encode('\n<think>Ready.</think>')
    else:
        tokens = []
    tokens += tok.encode(xml(PATCH)) + [256]
    for token in tokens:
        mask = torch.full((1, (259 + 31) // 32), -1, dtype=torch.int32)
        state.fill_mask(mask, 0)
        assert int(mask[0, token // 32]) & (1 << (token % 32))
        state.accept(token)
    assert state.matcher.is_terminated()


def test_tool_spec_survives_worker_serialization():
    import pickle
    from gllm.scheduling.distributed import DriverPayloadBuilder, FollowerSeqStore
    spec = so.StructuredOutput(None, tag=tag())
    request = seq(spec)
    payload = pickle.loads(pickle.dumps(DriverPayloadBuilder().build([request], [])))
    follower = FollowerSeqStore().apply_payload(payload)[0]
    assert follower.structured_output == spec


def test_json_object_and_tools_allow_arbitrary_object_properties():
    spec = tag(schema='{"type":"object"}')
    assert matcher(spec).accept_string('{"answer":42}')
