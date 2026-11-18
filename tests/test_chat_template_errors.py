"""Explicit template input rejection is a 400, not an internal server error."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from jinja2 import Environment, TemplateError, TemplateSyntaxError, UndefinedError

from gllm.entrypoints import api_server


@pytest.fixture
def frontend(monkeypatch):
    seen = []
    state = {"disagg": False, "error": None, "extract_error": None}
    template = Environment().from_string(
        '{% if reasoning_effort is defined and reasoning_effort not in '
        '["low", "medium", "xhigh"] %}'
        '{{ raise_exception("Unexpected reasoning effort " ~ reasoning_effort ~ '
        '". Supported types are xhigh (default), medium, and low.") }}'
        '{% endif %}'
        '{% if reject_roles | default(false) %}'
        '{{ raise_exception("Conversation roles must alternate user/assistant.") }}'
        '{% endif %}OK'
    )

    def reject(message):
        raise TemplateError(message)

    def encode(messages, **kwargs):
        options = kwargs.get("chat_template_kwargs") or {}
        seen.append(dict(options))
        if state["error"] is not None:
            raise state["error"]
        template.render(raise_exception=reject, **options)
        return [1, 2]

    def extract(messages):
        if state["extract_error"] is not None:
            raise state["extract_error"]
        return {"image": ["image"]} if state["disagg"] else None

    runner = SimpleNamespace(extract_modify_mm=extract, use_mm=True,
                             extract_mm_items_ordered=lambda messages: [("image", "image")],
                             encode=encode, encode_skeleton=encode)
    llm = SimpleNamespace(model_path="test", model_runner=runner,
                          is_disagg_lm=False, check_seq_length=lambda *args: False)
    monkeypatch.setattr(api_server, "llm", llm)
    return TestClient(api_server._build_app(), raise_server_exceptions=False), state, seen, llm


def payload(endpoint, **kwargs):
    if endpoint == "chat":
        return {"model": "test", "messages": [{"role": "user", "content": "Hello"}], **kwargs}
    return {"model": "test", "input": "Hello", **kwargs}


def url(endpoint):
    return "/v1/chat/completions" if endpoint == "chat" else "/v1/responses"


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("disagg", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("effort", ["minimal", "high"])
def test_unsupported_effort_is_a_clear_400(frontend, endpoint, disagg, streaming, effort):
    client, state, seen, llm = frontend
    state["disagg"] = llm.is_disagg_lm = disagg
    options = {"reasoning_effort": effort} if endpoint == "chat" else {"reasoning": {"effort": effort}}
    response = client.post(url(endpoint), json=payload(endpoint, stream=streaming, **options))
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_chat_template_input"
    assert f"Unexpected reasoning effort {effort}" in error["message"]
    assert "xhigh (default), medium, and low" in error["message"]
    assert error["param"] is None  # Jinja does not identify the offending field.
    assert seen == [{"reasoning_effort": effort}]  # No silent effort remapping.


def test_other_explicit_template_rejections_are_also_client_errors(frontend):
    client, _, _, _ = frontend
    response = client.post(url("chat"), json=payload(
        "chat", chat_template_kwargs={"reject_roles": True}))
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Conversation roles must alternate user/assistant."


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("error", [
    RuntimeError("internal failure"), AttributeError("implementation bug"),
    TemplateSyntaxError("broken template", 1), UndefinedError("missing template variable"),
])
def test_internal_errors_remain_500(frontend, endpoint, error):
    client, state, _, _ = frontend
    state["error"] = error
    response = client.post(url(endpoint), json=payload(endpoint))
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["message"] == "Internal server error"


def test_template_error_outside_encoding_is_not_misclassified(frontend):
    client, state, _, _ = frontend
    state["extract_error"] = TemplateError("unexpected extraction failure")
    response = client.post(url("chat"), json=payload("chat"))
    assert response.status_code == 500


@pytest.mark.parametrize("disagg", [False, True])
@pytest.mark.parametrize("options", [None, {"reasoning_effort": "low"},
                                     {"reasoning_effort": "medium"}, {"reasoning_effort": "xhigh"},
                                     {"enable_thinking": False, "thinking": False}])
def test_supported_options_still_tokenize(frontend, disagg, options):
    _, state, seen, llm = frontend
    state["disagg"] = llm.is_disagg_lm = disagg
    tokens, media, items = asyncio.run(api_server._tokenize_messages(
        [{"role": "user", "content": "Hello"}], None, options))
    assert tokens == [1, 2]
    assert media is None
    assert items == ([("image", "image")] if disagg else None)
    assert seen == [options or {}]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("effort", ["minimal", "high"])
def test_local_checkpoint_rejection_is_a_400(frontend, endpoint, effort):
    import os
    from transformers import AutoProcessor

    path = os.environ.get("GLLM_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("Set GLLM_TEST_QWEN_TOKENIZER to a local Qwen checkpoint")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    client, _, _, llm = frontend

    def encode(messages, **kwargs):
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **(kwargs.get("chat_template_kwargs") or {}),
        )
        raise AssertionError("The checkpoint must reject the unsupported effort")

    llm.model_runner.encode = encode
    options = {"reasoning_effort": effort} if endpoint == "chat" else {"reasoning": {"effort": effort}}
    response = client.post(url(endpoint), json=payload(endpoint, **options))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_chat_template_input"
    assert f"Unexpected reasoning effort {effort}" in response.json()["error"]["message"]
