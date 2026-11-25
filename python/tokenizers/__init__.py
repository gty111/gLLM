"""Tokenizer-side helpers for offline chat encoding (``LLM`` / ``ModelRunner``).

- :mod:`gllm.tokenizers.chat_messages` -- OpenAI message normalization for
  model-native chat templates.
- :mod:`gllm.tokenizers.deepseek_official` -- DeepSeek official message
  encoder (loaded from the checkpoint's ``encoding/`` dir).
- :mod:`gllm.tokenizers.reasoning` -- ``<think>`` handling used by the
  structured-output sampler.
"""

from gllm.tokenizers.chat_messages import (
    normalize_chat_template_messages,
    normalize_chat_template_tool_arguments,
)

__all__ = [
    "normalize_chat_template_tool_arguments",
    "normalize_chat_template_messages",
]
