import json
from types import SimpleNamespace

import jsonschema
import pytest
import torch

import gllm.structured_output as so
from gllm.runtime.sequence import GenerationSequence


def title_schema(bounds=None, **extra):
    return {
        "type": "object",
        "properties": {"title": {"type": "string", **(bounds or {}), **extra}},
        "required": ["title"], "additionalProperties": False,
    }


def fmt(schema):
    return {"type": "json_schema", "json_schema": {
        "name": "title", "strict": True, "schema": schema,
    }}


@pytest.fixture(scope="module")
def byte_compiler():
    import xgrammar as xgr

    vocab = [bytes([i]) for i in range(256)] + [b"<eos>", b"abc", "\u4e2d\u6587".encode(), b"\\n"]
    info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, stop_token_ids=[256])
    return xgr.GrammarCompiler(info, max_threads=1)


@pytest.fixture
def backend(monkeypatch, byte_compiler):
    monkeypatch.setattr(so, "compiler", lambda *args: byte_compiler)
    return so.StructuredSampler(None)


def sequence(schema):
    seq = GenerationSequence(0, [1], [256], 256,
                             structured_output=so.StructuredOutput(so.normalize_format(fmt(schema))))
    seq.computed_token_num = 0
    seq.to_compute_token_num = 1
    return seq


def accepts(backend, schema, raw):
    seq = sequence(schema)
    for token in [*raw.encode(), 256]:
        logits = torch.zeros(1, 260)
        active = backend.mask(logits, [seq])
        if torch.isneginf(logits[0, token]):
            return False
        backend.record(active, torch.tensor([token]))
        seq.computed_token_num += 1
    return True


@pytest.mark.parametrize("bounds", [
    {"minLength": 1, "maxLength": 36}, {"minLength": 2, "maxLength": 2},
    {"maxLength": 0}, {"minLength": 2}, {"maxLength": 2},
])
@pytest.mark.parametrize("value", ["", "a", "ab", "abc", "\u4e2d", "\u4e2d\u6587", "\U0001f600", "a" * 36, "a" * 37])
def test_decoded_length_matches_jsonschema(backend, bounds, value):
    schema = title_schema(bounds)
    payload = {"title": value}
    expected = jsonschema.Draft202012Validator(schema).is_valid(payload)
    assert accepts(backend, schema, json.dumps(payload, ensure_ascii=False)) == expected


@pytest.mark.parametrize("value", ["\x00", "\t", "\x1f", r"\n", r"\\", r'\"', r"\u0041", r"\uD83D\uDE00"])
def test_string_edge_cases_follow_native_xgrammar(backend, byte_compiler, value):
    import xgrammar as xgr

    schema = title_schema({"minLength": 1, "maxLength": 2})
    encoded = so.normalize_format(fmt(schema))
    matcher = xgr.GrammarMatcher(byte_compiler.compile_json_schema(encoded, strict_mode=False))
    raw = '{"title":"' + value + '"}'
    expected = all(matcher.accept_token(token) for token in [*raw.encode(), 256])
    assert accepts(backend, schema, raw) == expected


def test_multi_character_token_boundary(backend):
    seq = sequence(title_schema({"maxLength": 2}))
    for token in b'{"title":"':
        logits = torch.zeros(1, 260)
        active = backend.mask(logits, [seq])
        assert not torch.isneginf(logits[0, token])
        backend.record(active, torch.tensor([token]))
        seq.computed_token_num += 1
    logits = torch.zeros(1, 260)
    backend.mask(logits, [seq])
    assert torch.isneginf(logits[0, 257])  # abc exceeds the bound in one token
    assert not torch.isneginf(logits[0, 258])  # Two Unicode characters fit the bound
    assert torch.isneginf(logits[0, 259])  # XGrammar 0.2.7 excludes bounded-string escapes


@pytest.mark.parametrize("bounds", [
    {"minLength": -1}, {"maxLength": -1}, {"maxLength": 1.5},
    {"minLength": True}, {"minLength": 4, "maxLength": 3},
])
def test_invalid_bounds_rejected(bounds):
    from gllm.entrypoints.api_server import _validate_output_format

    with pytest.raises(ValueError):
        so.normalize_format(fmt(title_schema(bounds)))
    error = _validate_output_format(fmt(title_schema(bounds)), "response_format")
    assert error.status_code == 400
    assert json.loads(error.body)["error"]["code"] == "invalid_output_format"


def test_untyped_length_rejected():
    schema = title_schema({"minLength": 1})
    del schema["properties"]["title"]["type"]
    with pytest.raises(ValueError, match="explicit string type"):
        so.normalize_format(fmt(schema))


def test_enum_cannot_bypass_length(backend):
    schema = title_schema({"minLength": 1, "maxLength": 2}, enum=["a", "\u4e2d\u6587"])
    assert accepts(backend, schema, '{"title":"\u4e2d\u6587"}')
    with pytest.raises(ValueError, match="enum/const"):
        so.normalize_format(fmt(title_schema({"maxLength": 1}, enum=["too long"])))


def test_nested_nullable_reference(backend):
    schema = {
        "type": "object", "properties": {"titles": {"type": "array", "items": {"$ref": "#/$defs/title"}}},
        "required": ["titles"], "additionalProperties": False,
        "$defs": {"title": {"type": ["string", "null"], "minLength": 1, "maxLength": 2}},
    }
    assert accepts(backend, schema, '{"titles":[null,"\u4e2d","ab"]}')
    assert not accepts(backend, schema, '{"titles":[""]}')
    assert not accepts(backend, schema, '{"titles":["abc"]}')


def test_codex_title_format_compiles_for_both_endpoints(monkeypatch, byte_compiler):
    from gllm.entrypoints import api_server
    from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest

    monkeypatch.setattr(api_server, "llm", SimpleNamespace(model_path="test"))
    monkeypatch.setattr(so, "compiler", lambda *args: byte_compiler)
    schema = title_schema({"minLength": 1, "maxLength": 36})
    chat = fmt(schema)
    response = {"type": "json_schema", **chat["json_schema"]}
    chat_request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "JSON"}],
                                         response_format=chat)
    response_request = ResponseRequest(model="test", input="JSON", text={"format": response})
    assert api_server._validate_chat_capabilities(chat_request) is None
    assert api_server._validate_response_capabilities(response_request) is None
    for value in (chat, response):
        spec = so.prepare_output(value, None, 260, [256], [1])
        assert json.loads(spec.schema) == schema


def test_speculative_length_boundary_rejects_draft_without_committing(backend):
    seq = sequence(title_schema({"minLength": 1, "maxLength": 2}))
    prefix = list(b'{"title":"')
    logits = torch.zeros(3, 260)
    # Two-character token fills the string; draft 'a' must be rejected.
    backend.mask_speculative(logits, [seq], [[1, *prefix]], [[258, ord('a'), ord('b')]])
    assert torch.isneginf(logits[0, ord('a')])
    assert not torch.isneginf(logits[0, ord('"')])
    assert torch.isfinite(logits[1:]).all()
    assert backend.states[seq].history == prefix
