"""Tokenizer-side logic: chat-template encoding and tool-call parsing.

- :mod:`gllm.tokenizers.deepseek_v32` -- DeepSeek-V3.2 official message encoder
  (loaded from the checkpoint's ``encoding/`` dir; replaces a hand-written Jinja
  chat template).
- :mod:`gllm.tokenizers.tool_parsers` -- model-native tool-call markup parsers
  (Qwen / Kimi / DeepSeek) that turn generated text into structured
  ``tool_calls``.
"""

from gllm.tokenizers.deepseek_v32 import (
    apply_dsv32_chat_template,
    load_dsv32_encoder,
)
from gllm.tokenizers.tool_parsers import ToolParser, get_tool_parser

__all__ = [
    "apply_dsv32_chat_template",
    "load_dsv32_encoder",
    "ToolParser",
    "get_tool_parser",
]
