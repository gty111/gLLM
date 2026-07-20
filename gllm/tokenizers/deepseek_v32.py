"""DeepSeek-V3.2 chat encoding via the model's own official encoder.

The V3.2 checkpoint does NOT ship a usable ``chat_template`` (neither in
``tokenizer_config.json`` nor a root ``chat_template.jinja`` from upstream).
Instead it bundles the reference message encoder at
``<model_path>/encoding/encoding_dsv32.py`` -- the same file vLLM vendors as
``vllm/tokenizers/deepseek_v32_encoding.py`` and drives via
``--tokenizer-mode deepseek_v32``. It renders the DeepSeek DSML prompt format
(``<｜User｜>...<｜Assistant｜>``, ``<think>`` gating, ``<｜DSML｜invoke>`` tool
calls) that a hand-written Jinja template cannot express.

This module loads that official encoder at runtime (zero-maintenance: it always
tracks whatever the checkpoint ships) and adapts gLLM's OpenAI-style call site to
it, mirroring vLLM's ``get_deepseek_v32_tokenizer`` glue. When the encoder file is
absent we return ``None`` so the caller falls back to ``apply_chat_template``.
"""

import importlib.util
import json
import os
from typing import Any, Optional

# model_path -> loaded encoder module (or None if it couldn't be loaded).
_ENCODER_CACHE: dict[str, Optional[Any]] = {}


def load_dsv32_encoder(model_path: str) -> Optional[Any]:
    """Dynamically import ``<model_path>/encoding/encoding_dsv32.py``.

    Returns the module (exposing ``encode_messages`` and
    ``parse_message_from_completion_text``) or ``None`` if the file is missing
    or fails to import -- callers then fall back to the Jinja chat template.
    """
    if model_path in _ENCODER_CACHE:
        return _ENCODER_CACHE[model_path]

    enc_path = os.path.join(model_path, "encoding", "encoding_dsv32.py")
    module: Optional[Any] = None
    if os.path.isfile(enc_path):
        try:
            spec = importlib.util.spec_from_file_location(
                "gllm_dsv32_encoding", enc_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Sanity check the API surface we rely on.
            if not hasattr(module, "encode_messages"):
                module = None
        except Exception:
            module = None

    _ENCODER_CACHE[model_path] = module
    return module


def apply_dsv32_chat_template(
    encoder: Any,
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    tokenize: bool = True,
    **kwargs: Any,
):
    """Render ``messages`` with the official DeepSeek-V3.2 encoder.

    Mirrors vLLM's ``_DeepseekV32Tokenizer.apply_chat_template``
    (``vllm/tokenizers/deepseek_v32.py``):

    - ``thinking`` / ``enable_thinking`` (from the request's
      ``chat_template_kwargs``) select ``thinking_mode`` (``"thinking"`` vs the
      default ``"chat"``).
    - When ``tools`` are supplied, they are attached to a leading ``system``
      message so the encoder renders the tool-declaration block.
    - Historical reasoning is dropped once a new ``user`` message is introduced.

    The encoder already emits the ``<｜begin▁of▁sentence｜>`` BOS, so when we
    tokenize we pass ``add_special_tokens=False`` to avoid a duplicate BOS.
    Returns a token-id list when ``tokenize`` else the prompt string.
    """
    thinking = bool(kwargs.get("thinking", False) or kwargs.get("enable_thinking", False))
    thinking_mode = "thinking" if thinking else "chat"

    # Normalize messages to plain JSON-native dicts. Requests arrive as OpenAI
    # TypedDicts, but nested fields (e.g. ``tool_calls``) can be a pydantic
    # ``ValidatorIterator`` -- not a list -- which the encoder's ``len(...)`` and
    # iteration choke on, and which isn't deep-copyable. A JSON round-trip
    # (``default=list`` materializes any lazy iterator) fully realizes them.
    norm: list[dict[str, Any]] = []
    for m in messages:
        if hasattr(m, "model_dump"):
            norm.append(m.model_dump(mode="json", exclude_none=True))
        else:
            norm.append(json.loads(json.dumps(m, default=list)))
    messages = norm
    if tools:
        messages.insert(0, {"role": "system", "tools": tools})

    # Reasoning from prior turns is only kept mid-thinking; a fresh user turn
    # resets it (matches vLLM's drop_thinking heuristic).
    drop_thinking = bool(messages) and messages[-1].get("role") == "user"

    prompt_str = encoder.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
    )

    if not tokenize:
        return prompt_str

    tok_kwargs = {k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs}
    return tokenizer.encode(prompt_str, add_special_tokens=False, **tok_kwargs)
