"""Adapters for checkpoint-bundled DeepSeek message encoders."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from typing import Any, Optional

from huggingface_hub import hf_hub_download, try_to_load_from_cache


_ENCODER_CACHE: dict[tuple[str, str], Optional[Any]] = {}


def load_deepseek_encoder(model_path: str, variant: str) -> Optional[Any]:
    """Load the checkpoint's ``encoding/encoding_<variant>.py`` module.

    ``model_path`` may be either a local checkpoint directory or a Hugging Face
    repository id.  Prefer the local HF cache so normal offline serving does
    not perform a network request; download only the small encoder source when
    the checkpoint was addressed by repo id and that file is not cached yet.
    """
    key = (model_path, variant)
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    relative_path = f"encoding/encoding_{variant}.py"
    path = os.path.join(model_path, relative_path)
    if not os.path.isfile(path) and not os.path.isdir(model_path):
        cached = try_to_load_from_cache(model_path, relative_path)
        if isinstance(cached, str):
            path = cached
        else:
            try:
                path = hf_hub_download(model_path, relative_path)
            except Exception:
                path = ""
    module: Optional[Any] = None
    if os.path.isfile(path):
        try:
            spec = importlib.util.spec_from_file_location(
                f"gllm_{variant}_encoding", path
            )
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                module = None
            else:
                spec.loader.exec_module(module)
                if not hasattr(module, "encode_messages"):
                    module = None
        except Exception:
            module = None
    _ENCODER_CACHE[key] = module
    return module


def _flatten_content(content):
    """Collapse OpenAI structured content parts into the encoder's plain text.

    The bundled DeepSeek encoders join a message's rendered parts with
    ``"\n\n".join(...)`` and therefore require ``content`` to be a string.
    Modern clients (the OpenAI SDK, vLLM's benchmark client, anything sending
    multi-part input) send ``[{"type": "text", "text": ...}, ...]`` instead,
    which reaches the encoder as a list and dies with a bare
    ``TypeError: sequence item 0: expected str instance, list found``.
    Flatten text parts here; a non-text part (image/audio) is not something a
    text-only DeepSeek encoder can render, so say so explicitly.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            raise ValueError(
                f"unsupported DeepSeek chat content part: {type(part).__name__}"
            )
        kind = part.get("type", "text")
        if kind not in ("text", "input_text"):
            raise ValueError(
                f"DeepSeek chat encoders render text only, got a {kind!r} part"
            )
        parts.append(part.get("text") or "")
    return "".join(parts)


def apply_deepseek_chat_template(
    encoder: Any,
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    tokenize: bool = True,
    **kwargs: Any,
):
    """Render OpenAI messages through a bundled DeepSeek encoder."""
    thinking = bool(
        kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)
    )
    thinking_mode = "thinking" if thinking else "chat"
    normalized = []
    for message in messages:
        if hasattr(message, "model_dump"):
            normalized.append(
                message.model_dump(mode="json", exclude_none=True)
            )
        else:
            normalized.append(json.loads(json.dumps(message, default=list)))
        if "content" in normalized[-1]:
            normalized[-1]["content"] = _flatten_content(
                normalized[-1]["content"]
            )
    if tools:
        normalized.insert(0, {"role": "system", "tools": tools})
    drop_thinking = bool(normalized) and normalized[-1].get("role") == "user"
    encode_kwargs = {
        "thinking_mode": thinking_mode,
        "drop_thinking": drop_thinking,
    }
    # V4 accepts this extension, while older V3.2 encoders do not. Inspect the
    # bundled function rather than branching on a model name.
    if "reasoning_effort" in inspect.signature(
        encoder.encode_messages
    ).parameters:
        encode_kwargs["reasoning_effort"] = kwargs.get("reasoning_effort")
    prompt = encoder.encode_messages(normalized, **encode_kwargs)
    if not tokenize:
        return prompt
    token_kwargs = {
        key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs
    }
    return tokenizer.encode(prompt, add_special_tokens=False, **token_kwargs)


__all__ = ["apply_deepseek_chat_template", "load_deepseek_encoder"]
