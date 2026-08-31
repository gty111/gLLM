from pathlib import Path

import gllm.tokenizers.deepseek_official as official
from gllm.tokenizers.deepseek_official import (
    apply_deepseek_chat_template,
    load_deepseek_encoder,
)


class _LegacyEncoder:
    def __init__(self):
        self.call = None

    def encode_messages(self, messages, thinking_mode, drop_thinking):
        self.call = (messages, thinking_mode, drop_thinking)
        return "rendered"


class _Tokenizer:
    def __init__(self):
        self.call = None

    def encode(self, prompt, **kwargs):
        self.call = (prompt, kwargs)
        return [1, 2, 3]


def test_official_encoder_adapter_supports_legacy_signature_and_tools():
    encoder = _LegacyEncoder()
    tokenizer = _Tokenizer()
    tools = [{"type": "function", "function": {"name": "f"}}]
    ids = apply_deepseek_chat_template(
        encoder,
        [{"role": "user", "content": "hello"}],
        tokenizer,
        tools=tools,
        thinking=True,
        reasoning_effort="high",
    )
    messages, mode, drop = encoder.call
    assert messages[0] == {"role": "system", "tools": tools}
    assert messages[1]["content"] == "hello"
    assert (mode, drop) == ("thinking", True)
    assert ids == [1, 2, 3]
    assert tokenizer.call == ("rendered", {"add_special_tokens": False})


def test_load_deepseek_encoder_uses_checkpoint_variant(tmp_path: Path):
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode, drop_thinking=True):\n"
        "    return thinking_mode\n",
        encoding="utf-8",
    )
    module = load_deepseek_encoder(str(tmp_path), "dsv4")
    assert module is not None
    assert module.encode_messages([], "chat") == "chat"
    assert load_deepseek_encoder(str(tmp_path), "dsv4") is module


def test_load_deepseek_encoder_resolves_huggingface_cache(tmp_path, monkeypatch):
    source = tmp_path / "encoding_dsv4.py"
    source.write_text(
        "def encode_messages(messages, thinking_mode, drop_thinking=True):\n"
        "    return thinking_mode\n",
        encoding="utf-8",
    )
    model_id = "deepseek-ai/test-v4"
    official._ENCODER_CACHE.pop((model_id, "dsv4"), None)
    monkeypatch.setattr(
        official,
        "try_to_load_from_cache",
        lambda repo_id, filename: str(source),
    )
    monkeypatch.setattr(
        official,
        "hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached encoder must not access the network")
        ),
    )

    module = load_deepseek_encoder(model_id, "dsv4")

    assert module is not None
    assert module.encode_messages([], "chat") == "chat"
