"""Optional assistant content must survive validation and media extraction."""

import copy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gllm.entrypoints import api_server
from gllm.entrypoints.protocol import ChatCompletionRequest
from gllm.multimodal.mixin import MmMixin
from gllm.tokenizers.mixin import TokenizerMixin
from gllm.utils import StreamOutput


FUNCTION_CALL = {"name": "lookup", "arguments": '{"key":"value"}'}
TOOL_CALL = {"id": "call-1", "type": "function", "function": FUNCTION_CALL}


class Runner(MmMixin, TokenizerMixin):
    def __init__(self):
        self.seen = None
        self.use_mm = True
        self.is_kimi_mm = False
        self._deepseek_encoder_variant = None
        self.tokenizer = SimpleNamespace(
            apply_chat_template=self.render, convert_tokens_to_ids=lambda marker: None,
        )
        self.processor = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: [self.render(messages, **kwargs)]
        )

    def render(self, messages, **kwargs):
        self.seen = copy.deepcopy(messages)
        assert all(isinstance(m["content"], (str, list)) for m in messages)
        return [1, 2]


class Stream:
    seq = SimpleNamespace(token_ids=[1, 2, 3], raw_prompt_len=2,
                          ignore_eos=False, finish_tokens=[99], output_len=16)

    async def __aiter__(self):
        yield StreamOutput("OK")


@pytest.fixture
def client(monkeypatch):
    runner = Runner()

    async def add_requests(*args, **kwargs):
        return Stream()

    monkeypatch.setattr(api_server, "llm", SimpleNamespace(
        model_path="test", model_runner=runner, check_seq_length=lambda *args: True,
        add_requests_async=add_requests,
    ))
    monkeypatch.setattr(api_server, "tool_parser", None)
    return TestClient(api_server._build_app()), runner


@pytest.mark.parametrize("call", [{"tool_calls": [TOOL_CALL]}, {"function_call": FUNCTION_CALL}])
@pytest.mark.parametrize("content", [{}, {"content": None}, {"content": ""}])
@pytest.mark.parametrize("stream", [False, True])
def test_http_tool_history_without_text_reaches_generation(client, call, content, stream):
    http, runner = client
    assistant = {"role": "assistant", **copy.deepcopy(call), **content}
    payload = {"model": "test", "messages": [
        {"role": "user", "content": "Look it up."}, assistant,
        {"role": "tool", "tool_call_id": "call-1", "content": "Found."},
    ], "stream": stream}
    original = copy.deepcopy(payload)
    response = http.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert "OK" in response.text
    assert runner.seen[1]["content"] == ""
    key = next(iter(call))
    assert key in runner.seen[1]
    assert payload == original


@pytest.mark.parametrize("role,content", [
    (role, content)
    for role in ("user", "system", "developer", "tool", "function", "custom", "assistant")
    for content in ({}, {"content": None})
    if not (content and role in ("assistant", "function"))
])
def test_missing_required_content_is_an_openai_400(client, role, content):
    http, runner = client
    response = http.post("/v1/chat/completions", json={"model": "test", "messages": [
        {"role": "user", "content": "Start."}, {"role": role, **content},
    ]})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "messages.1.content"
    assert error["code"] == "invalid_request"
    assert runner.seen is None


@pytest.mark.parametrize("content", [None, "", []])
def test_nullable_assistant_and_empty_content_remain_supported(content):
    messages = [{"role": "assistant", "content": content}]
    request = ChatCompletionRequest(model="test", messages=messages)
    actual = request.messages[0]["content"]
    assert (list(actual) if isinstance(content, list) else actual) == content


def test_media_extraction_skips_absent_content_and_preserves_order():
    messages = [
        {"role": "assistant", "tool_calls": [copy.deepcopy(TOOL_CALL)]},
        {"role": "assistant", "content": None, "tool_calls": [copy.deepcopy(TOOL_CALL)]},
        {"role": "tool", "tool_call_id": "call-1", "content": [
            {"type": "text", "text": "Before"},
            {"type": "image_url", "image_url": {"url": "image-1"}},
            {"type": "video_url", "video_url": "video-1"},
            {"type": "image_url", "image_url": "image-2"},
        ]},
    ]
    request = ChatCompletionRequest(model="test", messages=messages)
    assert MmMixin.extract_modify_mm(None, request.messages) == {
        "image": ["image-1", "image-2"], "video": ["video-1"],
    }
    assert MmMixin.extract_mm_items_ordered(None, request.messages) == [
        ("image", "image-1"), ("video", "video-1"), ("image", "image-2"),
    ]
    assert list(request.messages[0]["tool_calls"])[0]["id"] == "call-1"
