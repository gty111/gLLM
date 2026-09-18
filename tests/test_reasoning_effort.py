"""Reasoning controls must reach both normal and disaggregated tokenization."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from gllm.entrypoints import api_server
from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("disagg", [False, True])
@pytest.mark.parametrize("effort", [None, "none", "minimal", "low", "medium", "high", "xhigh"])
def test_effort_reaches_tokenizer(monkeypatch, endpoint, disagg, effort):
    captured = []

    def encode(messages, **kwargs):
        captured.append(kwargs.get("chat_template_kwargs"))
        return [1, 2]

    runner = SimpleNamespace(
        extract_modify_mm=lambda messages: ["image"] if disagg else None,
        extract_mm_items_ordered=lambda messages: ["image"],
        encode=encode,
        encode_skeleton=encode,
        use_mm=True,
    )
    monkeypatch.setattr(api_server, "llm", SimpleNamespace(
        model_path="test", model_runner=runner, is_disagg_lm=disagg,
        # Stop after tokenization: this regression requires no inference engine.
        check_seq_length=lambda *args: False,
    ))
    if endpoint == "chat":
        req = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": "hello"}],
            reasoning_effort=effort,
        )
        handler = api_server.create_chat_completion
    else:
        req = ResponseRequest(
            model="test", input="hello",
            reasoning={"effort": effort} if effort is not None else None,
        )
        handler = api_server.create_response
    response = asyncio.run(handler(req, SimpleNamespace()))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["code"] == "context_length_exceeded"
    expected = None
    if effort == "none":
        expected = {"enable_thinking": False, "thinking": False}
    elif effort is not None:
        expected = {"reasoning_effort": effort}
    assert captured == [expected]


@pytest.mark.parametrize("effort", [None, "none", "medium"])
def test_explicit_template_overrides_are_preserved_without_mutation(effort):
    overrides = {"reasoning_effort": "low", "enable_thinking": True,
                 "thinking": True, "custom_option": "keep"}
    req = ChatCompletionRequest(
        model="test", messages=[], reasoning_effort=effort,
        chat_template_kwargs=dict(overrides),
    )
    actual = api_server._chat_template_kwargs(req)
    assert actual == overrides
    assert req.chat_template_kwargs == overrides
    assert actual is not req.chat_template_kwargs
