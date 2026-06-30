"""Model-native tool-call parsers.

The chat server lets a model emit tool calls in its own native markup (Qwen's
Hermes-style ``<tool_call>{...}</tool_call>`` JSON blocks, Qwen3.5's
``<tool_call><function=..><parameter=..>`` XML form, Kimi's
``<|tool_call_*|>`` markers, ...). These parsers turn that raw decoded text
into the structured ``tool_calls`` of the OpenAI schema so clients (and
function-calling benchmarks such as BFCL) can consume them uniformly.

Two entry points are used by ``serving_chat.py``:

* ``ToolParser.parse(full_text) -> (content, tool_calls)`` for non-streaming
  responses: ``content`` is the assistant text with the tool-call markup
  removed (``None``/empty when the whole reply was a tool call), and
  ``tool_calls`` is a list of :class:`ToolCall`.
* ``ToolParser.stream_parser()`` returns a stateful object exposing
  ``process(full_text) -> DeltaMessage | None`` (called with the *cumulative*
  text on every streamed step) and ``has_tool_calls() -> bool``.

``get_tool_parser(architecture, name)`` resolves which parser to use: an
explicit ``name`` ("qwen"/"kimi") wins, otherwise it is auto-detected from the
model architecture string. Returns ``None`` for unknown models, in which case
the server leaves the raw text in ``content`` untouched.
"""

import json
import re
from typing import List, Optional, Tuple

from gllm.entrypoints.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)


def _dump_arguments(arguments) -> str:
    """Tool-call arguments are always serialized as a JSON *string* in the
    OpenAI schema; pass through strings, JSON-encode everything else."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


class ToolParser:
    """Base class. Subclasses set ``name`` and implement :meth:`parse` and
    :meth:`stream_parser`."""

    name: str = "base"

    def parse(self, full_text: str) -> Tuple[Optional[str], List[ToolCall]]:
        raise NotImplementedError

    def stream_parser(self) -> "StreamToolParser":
        raise NotImplementedError


class StreamToolParser:
    """Incremental counterpart of :meth:`ToolParser.parse`.

    ``process`` is fed the cumulative decoded text on each step and returns the
    delta to emit (plain content before any tool call, then one tool call per
    completed block) or ``None`` when there is nothing new to send yet.
    """

    def __init__(self, parser: ToolParser) -> None:
        self._parser = parser
        self._content_emitted = 0
        self._tool_calls_emitted = 0

    def has_tool_calls(self) -> bool:
        return self._tool_calls_emitted > 0

    def process(self, full_text: str) -> Optional[DeltaMessage]:
        head = self._parser.content_prefix(full_text)

        # 1) Stream the leading natural-language content (the part before the
        #    first tool-call marker) as it grows.
        if self._content_emitted < len(head):
            new = head[self._content_emitted :]
            self._content_emitted = len(head)
            if new:
                return DeltaMessage(content=new)

        # 2) Emit the next fully-formed tool call, if one is now available.
        calls = self._parser.parse(full_text)[1]
        if len(calls) > self._tool_calls_emitted:
            idx = self._tool_calls_emitted
            call = calls[idx]
            self._tool_calls_emitted += 1
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=idx,
                        id=call.id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=call.function.name,
                            arguments=call.function.arguments,
                        ),
                    )
                ]
            )
        return None


class QwenToolParser(ToolParser):
    """Qwen / Hermes style: zero or more ``<tool_call>{json}</tool_call>``
    blocks, optionally preceded by natural-language content."""

    name = "qwen"
    _START = "<tool_call>"
    _BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def content_prefix(self, full_text: str) -> str:
        return full_text.split(self._START, 1)[0]

    def parse(self, full_text: str) -> Tuple[Optional[str], List[ToolCall]]:
        if self._START not in full_text:
            return full_text, []

        tool_calls: List[ToolCall] = []
        for block in self._BLOCK_RE.findall(full_text):
            block = block.strip()
            if not block:
                continue
            try:
                obj = json.loads(block)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if not name:
                continue
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=_dump_arguments(obj.get("arguments", {})),
                    )
                )
            )

        content = self.content_prefix(full_text).strip() or None
        return content, tool_calls

    def stream_parser(self) -> StreamToolParser:
        return StreamToolParser(self)


class Qwen3ToolParser(ToolParser):
    """Qwen3.5 XML-style tool calls. Each call renders as::

        <tool_call>
        <function=NAME>
        <parameter=ARG>
        VALUE
        </parameter>
        ...
        </function>
        </tool_call>

    The chat template emits scalar arg values as bare text (``10``, ``units``,
    ``true``) and dict/list values as JSON, so on parse we ``json.loads`` each
    value and fall back to the raw string -- ints/floats/bools/arrays/objects
    round-trip while plain strings stay strings.
    """

    name = "qwen3"
    _START = "<tool_call>"
    _FUNC_RE = re.compile(
        r"<function=(?P<name>[^>\n]+)>(?P<body>.*?)</function>", re.DOTALL
    )
    _PARAM_RE = re.compile(
        r"<parameter=(?P<key>[^>\n]+)>\n?(?P<val>.*?)\n?</parameter>", re.DOTALL
    )

    def content_prefix(self, full_text: str) -> str:
        return full_text.split(self._START, 1)[0]

    @staticmethod
    def _coerce(raw: str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    def parse(self, full_text: str) -> Tuple[Optional[str], List[ToolCall]]:
        if self._START not in full_text:
            return full_text, []

        tool_calls: List[ToolCall] = []
        # ``<function=..>`` blocks only ever appear inside ``<tool_call>``, so
        # scanning the whole text is safe and also tolerates a missing/garbled
        # closing ``</tool_call>`` tag.
        for fm in self._FUNC_RE.finditer(full_text):
            name = fm.group("name").strip()
            if not name:
                continue
            args = {}
            for pm in self._PARAM_RE.finditer(fm.group("body")):
                key = pm.group("key").strip()
                if key:
                    args[key] = self._coerce(pm.group("val"))
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name, arguments=_dump_arguments(args)
                    )
                )
            )

        content = self.content_prefix(full_text).strip() or None
        return content, tool_calls

    def stream_parser(self) -> StreamToolParser:
        return StreamToolParser(self)


class KimiToolParser(ToolParser):
    """Kimi K2 style: tool calls are wrapped in a section and each call is

    ``<|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>{json}<|tool_call_end|>``
    """

    name = "kimi"
    _SECTION_START = "<|tool_calls_section_begin|>"
    _CALL_RE = re.compile(
        r"<\|tool_call_begin\|>\s*(?P<fid>[^\s<]+?)\s*"
        r"<\|tool_call_argument_begin\|>\s*(?P<args>.*?)\s*<\|tool_call_end\|>",
        re.DOTALL,
    )

    def content_prefix(self, full_text: str) -> str:
        return full_text.split(self._SECTION_START, 1)[0]

    @staticmethod
    def _name_from_id(fid: str) -> str:
        # Ids look like "functions.get_weather:0"; strip the index and the
        # leading "functions." namespace if present.
        fid = fid.split(":", 1)[0]
        if fid.startswith("functions."):
            fid = fid[len("functions.") :]
        return fid

    def parse(self, full_text: str) -> Tuple[Optional[str], List[ToolCall]]:
        if self._SECTION_START not in full_text:
            return full_text, []

        tool_calls: List[ToolCall] = []
        for m in self._CALL_RE.finditer(full_text):
            name = self._name_from_id(m.group("fid"))
            if not name:
                continue
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=_dump_arguments(m.group("args").strip()),
                    )
                )
            )

        content = self.content_prefix(full_text).strip() or None
        return content, tool_calls

    def stream_parser(self) -> StreamToolParser:
        return StreamToolParser(self)


def _qwen_parser_for_arch(architecture: Optional[str]) -> ToolParser:
    """Pick the right Qwen markup for the architecture: Qwen3.5 switched from
    the Hermes JSON ``<tool_call>{...}</tool_call>`` form to the
    ``<function=..><parameter=..>`` XML form; older Qwen stays on Hermes."""
    arch = (architecture or "").lower()
    if "qwen3_5" in arch or "qwen3.5" in arch:
        return Qwen3ToolParser()
    return QwenToolParser()


# Explicit ``--tool-call-parser`` names. The qwen-family names defer to the
# architecture to pick the markup variant; "hermes"/"qwen3" force a variant.
_AVAILABLE_NAMES = ("qwen", "qwen2", "qwen2.5", "qwen3", "qwen3.5", "hermes", "kimi")


def get_tool_parser(
    architecture: Optional[str] = None, name: Optional[str] = None
) -> Optional[ToolParser]:
    """Resolve the tool-call parser.

    An explicit ``name`` takes precedence; otherwise the parser is auto-detected
    from the model ``architecture`` string. The qwen-family names ("qwen",
    "qwen2", "qwen3", ...) resolve to the Hermes or XML variant based on the
    architecture; "hermes" forces Hermes and "qwen3.5" forces XML. Returns
    ``None`` for unknown models (raw text then passes through as ``content``).
    """
    if name:
        n = name.lower()
        if n in ("qwen", "qwen2", "qwen2.5", "qwen3"):
            return _qwen_parser_for_arch(architecture)
        if n == "hermes":
            return QwenToolParser()
        if n in ("qwen3.5", "qwen3_5", "qwen_xml"):
            return Qwen3ToolParser()
        if n == "kimi":
            return KimiToolParser()
        raise ValueError(
            f"Unknown tool-call parser '{name}'. Available: {list(_AVAILABLE_NAMES)}"
        )

    if architecture:
        arch = architecture.lower()
        if "qwen" in arch:
            return _qwen_parser_for_arch(architecture)
        if "kimi" in arch:
            return KimiToolParser()

    return None
