"""Pre-tokenized Completions inputs bypass tokenization and fail safely."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gllm.entrypoints import api_server
from test_output_budget import FakeStream, make_engine


@pytest.fixture
def completion_client(monkeypatch):
    engine = make_engine(context=16)
    encoded, admitted = [], []
    def encode(text):
        assert isinstance(text, str)
        encoded.append(text)
        return [1, 2]
    engine.model_runner = SimpleNamespace(
        encode=encode, model_loader=SimpleNamespace(vocab_size=256),
        tokenizer=SimpleNamespace(vocab_size=128),
    )
    async def add_requests(raw, *args, **kwargs):
        kwargs.pop('dp_index', None)
        seq = engine.allocate_seq(*args, **kwargs)
        admitted.append(list(seq.token_ids))
        return FakeStream(seq)
    engine.add_requests_async = add_requests
    monkeypatch.setattr(api_server, 'llm', engine)
    with TestClient(api_server._build_app()) as client:
        yield client, encoded, admitted


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('tokens', [[0], [0, 255, 12], [9, 9, 9]])
def test_token_ids_reach_engine_unchanged(completion_client, streaming, tokens):
    client, encoded, admitted = completion_client
    response = client.post('/v1/completions', json=dict(
        model='test', prompt=tokens, max_tokens=1, stream=streaming,
        stream_options=dict(include_usage=True) if streaming else None,
    ))
    assert response.status_code == 200, response.text
    assert encoded == [] and admitted == [tokens]
    if streaming:
        assert 'data: [DONE]' in response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        usage = next(event['usage'] for event in events if event.get('usage'))
    else:
        usage = response.json()['usage']
    assert usage['prompt_tokens'] == len(tokens)
    assert usage['completion_tokens'] == 1


@pytest.mark.parametrize('prompt', [
    [], [-1], [256], [2**100], [True], [1.5], [1.0], [1, 'two'],
    [[1, 2], [3]], ['first', 'second'], [[1]], ['single'],
])
def test_invalid_or_batched_prompts_return_400_without_enqueue(completion_client, prompt):
    client, encoded, admitted = completion_client
    response = client.post('/v1/completions', json=dict(model='test', prompt=prompt, max_tokens=1))
    assert response.status_code == 400, response.text
    assert response.json()['error']['type'] == 'invalid_request_error'
    assert response.json()['error']['param'].startswith('prompt')
    assert not admitted and not encoded


def test_token_array_obeys_context_limit(completion_client):
    client, encoded, admitted = completion_client
    response = client.post('/v1/completions', json=dict(model='test', prompt=[1]*16, max_tokens=1))
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'context_length_exceeded'
    assert not admitted and not encoded


def test_text_prompt_keeps_existing_encoding(completion_client):
    client, encoded, admitted = completion_client
    response = client.post('/v1/completions', json=dict(model='test', prompt='Hello', max_tokens=1))
    assert response.status_code == 200
    assert encoded == ['Hello'] and admitted == [[1, 2]]
