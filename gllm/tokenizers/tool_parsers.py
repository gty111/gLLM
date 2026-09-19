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
import math
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from gllm.entrypoints.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)


def normalize_chat_template_tool_arguments(messages) -> None:
    """Convert OpenAI wire-format tool arguments for chat templates.

    OpenAI assistant messages carry ``function.arguments`` as a JSON string,
    while several Hugging Face chat templates (including Qwen3.5/3.8) iterate
    the arguments as a mapping. Requests have already passed protocol
    validation by this point, so decode valid JSON objects in place before
    rendering. Invalid or non-object JSON is left untouched rather than
    changing its meaning.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if tool_calls is None or isinstance(tool_calls, (str, bytes, dict)):
            continue
        if not isinstance(tool_calls, list):
            try:
                tool_calls = list(tool_calls)
            except TypeError:
                continue
            message["tool_calls"] = tool_calls
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                function["arguments"] = parsed


def normalize_chat_template_messages(messages) -> None:
    """Normalize modern OpenAI messages for model-native chat templates.

    ``developer`` is the current OpenAI instruction role, while many local
    templates still only recognize the equivalent legacy ``system`` role.
    Tool-call arguments also need conversion from their JSON wire string to the
    mapping expected by several templates.
    """
    normalize_chat_template_tool_arguments(messages)
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "developer":
            message["role"] = "system"


def _dump_arguments(arguments) -> str:
    """Tool-call arguments are always serialized as a JSON *string* in the
    OpenAI schema; pass through strings, JSON-encode everything else."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


# ── Schema-aware argument coercion ──────────────────────────────────────────
# XML tool-call markup (Qwen3.5) carries no type information -- every
# ``<parameter>`` value is raw text. To emit correctly-typed ``tool_calls`` we
# coerce each value to the type declared in the tool's JSON schema: string-typed
# params stay strings, while integer/number/boolean/array/object params are
# converted. Schema-driven conversion is required for native tool calls: a
# consumer can turn ``"4"`` into an int when
# the schema says integer, but a spurious ``json.loads`` that turns a string
# ``"4"`` into ``int`` unconditionally breaks string-typed params (e.g. BFCL's
# Java/JS categories, where every value is a string).

_TYPE_ALIASES: Dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "long": "integer",
    "short": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
    "tuple": "array",
}


def _extract_types_from_schema(schema: Any) -> List[str]:
    """All possible JSON-Schema type strings for a property (handles ``type``
    as str/list, ``enum`` inference and ``anyOf``/``oneOf``/``allOf``).
    Defaults to ``["string"]`` when nothing can be determined."""
    if not isinstance(schema, dict):
        return ["string"]
    types: set = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        for v in enum:
            if v is None:
                types.add("null")
            elif isinstance(v, bool):
                types.add("boolean")
            elif isinstance(v, int):
                types.add("integer")
            elif isinstance(v, float):
                types.add("number")
            elif isinstance(v, str):
                types.add("string")
            elif isinstance(v, list):
                types.add("array")
            elif isinstance(v, dict):
                types.add("object")
    for field in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(field)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_types_from_schema(choice))
    return list(types) if types else ["string"]


def _json_finite(obj: Any) -> bool:
    """JSON has no inf/nan; reject them so we never emit invalid JSON."""
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, list):
        return all(_json_finite(x) for x in obj)
    if isinstance(obj, dict):
        return all(_json_finite(v) for v in obj.values())
    return True


def _coerce_to_schema_type(value: str, schema_types) -> Any:
    """Best-effort coercion of a raw string to a JSON-Schema type, trying
    ``null > integer > number > boolean > object > array > string`` and
    falling back to the original string when nothing fits."""
    if isinstance(schema_types, str):
        schema_types = [schema_types]
    normalized = {
        _TYPE_ALIASES.get(k, k) for t in schema_types for k in [str(t).strip().lower()]
    }
    for candidate in (
        "null",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
        "string",
    ):
        if candidate not in normalized:
            continue
        if candidate == "null":
            if value.strip().lower() == "null":
                return None
            continue
        if candidate == "string":
            return value
        if candidate == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate == "number":
            try:
                val = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(val):
                continue
            return val if val != int(val) else int(val)
        if candidate == "boolean":
            low = value.strip().lower()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
            continue
        if candidate in ("object", "array"):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _json_finite(parsed):
                return parsed
            continue
    # No declared type matched cleanly; try a plain JSON parse, else keep string.
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    return parsed if _json_finite(parsed) else value


def _coerce_value(value: Any, schema: Any) -> Any:
    """Coerce a parsed value against its schema (recursing into object
    ``properties`` and array ``items``)."""
    if not isinstance(schema, dict):
        return value
    if isinstance(value, str):
        return _coerce_to_schema_type(value, _extract_types_from_schema(schema))
    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for k in list(value):
                if isinstance(props.get(k), dict):
                    value[k] = _coerce_value(value[k], props[k])
        return value
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                value[i] = _coerce_value(item, items)
        return value
    return value


def _properties_for(tools, func_name: str) -> Optional[Dict[str, Any]]:
    """Find the ``parameters.properties`` schema map for ``func_name`` in the
    request's ``tools`` (accepts pydantic tool params or plain dicts)."""
    if not tools:
        return None
    for tool in tools:
        fn = getattr(tool, "function", None)
        if fn is None and isinstance(tool, dict):
            fn = tool.get("function")
        if fn is None:
            continue
        name = getattr(fn, "name", None)
        params = getattr(fn, "parameters", None)
        if isinstance(fn, dict):
            name = fn.get("name")
            params = fn.get("parameters")
        if name != func_name:
            continue
        if isinstance(params, dict):
            props = params.get("properties")
            return props if isinstance(props, dict) else None
        return None
    return None


def _coerce_args(args: Dict[str, Any], tools, func_name: str) -> Dict[str, Any]:
    """Type-correct an all-string XML arg dict using the tool schema. A no-op
    when no schema is available (values then stay strings)."""
    props = _properties_for(tools, func_name)
    if not props:
        return args
    for key in list(args):
        schema = props.get(key)
        if isinstance(schema, dict):
            args[key] = _coerce_value(args[key], schema)
    return args


class ToolParser:
    """Base class. Subclasses set ``name`` and implement :meth:`parse` and
    :meth:`stream_parser`."""

    name: str = "base"

    def parse(self, full_text: str, tools=None) -> Tuple[Optional[str], List[ToolCall]]:
        raise NotImplementedError

    def stream_parser(self, tools=None) -> "StreamToolParser":
        raise NotImplementedError


class StreamToolParser:
    """Incremental counterpart of :meth:`ToolParser.parse`.

    ``process`` is fed the cumulative decoded text on each step and returns the
    delta to emit (plain content before any tool call, then one tool call per
    completed block) or ``None`` when there is nothing new to send yet.
    """

    def __init__(self, parser: ToolParser, tools=None) -> None:
        self._parser = parser
        self._tools = tools
        self._content_emitted = 0
        self._tool_calls_emitted = 0

    def has_tool_calls(self) -> bool:
        return self._tool_calls_emitted > 0

    def process(self, full_text: str, *, final: bool = False) -> Optional[DeltaMessage]:
        head = self._parser.content_prefix(full_text)
        # The Qwen opening marker can share a chunk with </think>, or arrive
        # in pieces. Do not publish a prefix before knowing whether it is a
        # tool call. At EOF an unfinished prefix is ordinary literal text.
        if not final and isinstance(self._parser, (QwenToolParser, Qwen3ToolParser)):
            marker = self._parser._START
            if marker not in full_text:
                for size in range(min(len(head), len(marker) - 1), 0, -1):
                    if head.endswith(marker[:size]):
                        head = head[:-size]
                        break

        # 1) Stream the leading natural-language content (the part before the
        #    first tool-call marker) as it grows.
        if self._content_emitted < len(head):
            new = head[self._content_emitted :]
            self._content_emitted = len(head)
            if new:
                return DeltaMessage(content=new)

        # 2) Emit the next fully-formed tool call, if one is now available.
        calls = self._parser.parse(full_text, self._tools)[1]
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


class ToolParseError(ValueError):
    """A model began a native tool call but did not produce a valid call."""


class QwenStreamToolParser(StreamToolParser):
    """Consume Qwen markup while preserving literal text and Markdown code.

    Only an opening marker followed by the parser's call syntax commits to a
    call. A marker in code, or followed by ordinary prose, remains literal.
    Keep the cursor at ambiguous chunk boundaries so already-emitted text is
    never retracted. Completed calls are decoded once, with stable ids.
    """

    def __init__(self, parser, tools=None):
        super().__init__(parser, tools)
        self._pos = 0
        self._code = None
        self._fenced = False
        self._pending = deque()
        self._error = None

    def process(self, full_text: str, *, final: bool = False) -> Optional[DeltaMessage]:
        if self._error is not None:
            if final:
                raise self._error
            return None
        if self._pending:
            call = self._pending.popleft()
            index = self._tool_calls_emitted
            self._tool_calls_emitted += 1
            return DeltaMessage(tool_calls=[DeltaToolCall(
                index=index, id=call.id, type="function",
                function=DeltaFunctionCall(name=call.function.name,
                                          arguments=call.function.arguments),
            )])
        text = []
        marker = self._parser._START
        while self._pos < len(full_text):
            pos = self._pos
            char = full_text[pos]
            # Escaped Markdown punctuation cannot open a code span or call.
            if self._code is None and char == "\\":
                if pos + 1 == len(full_text) and not final:
                    break
                end = min(pos + 2, len(full_text))
                text.append(full_text[pos:end])
                self._pos = end
                continue
            if char in "`~":
                end = pos + 1
                while end < len(full_text) and full_text[end] == char:
                    end += 1
                if end == len(full_text) and not final:
                    break  # The delimiter run may continue in the next chunk.
                size = end - pos
                indent = full_text[full_text.rfind("\n", 0, pos) + 1:pos]
                line_start = len(indent) <= 3 and not indent.strip()
                if self._code is not None:
                    code_char, code_size = self._code
                    closes = char == code_char and (
                        size == code_size if not self._fenced
                        else size >= code_size and line_start
                    )
                    if closes and self._fenced:
                        line_end = full_text.find("\n", end)
                        rest = full_text[end:line_end if line_end >= 0 else None]
                        if line_end < 0 and not final and not rest.strip():
                            break  # A fence closer allows only trailing whitespace.
                        closes = not rest.strip()
                    if closes:
                        self._code = None
                        self._fenced = False
                elif line_start and size >= 3:
                    self._code = (char, size)
                    self._fenced = True
                elif char == "`":
                    self._code = (char, size)
                text.append(full_text[pos:end])
                self._pos = end
                continue
            if self._code is None and char == "<":
                tail = full_text[pos:]
                if not final and marker.startswith(tail):
                    break
                if tail.startswith(marker):
                    if text:
                        break  # Publish preceding prose before inspecting the call.
                    body_start = pos + len(marker)
                    while body_start < len(full_text) and full_text[body_start].isspace():
                        body_start += 1
                    body = full_text[body_start:]
                    opener = self._parser._CALL_START
                    if not body or opener.startswith(body):
                        if not final:
                            break
                        if body:
                            raise ToolParseError("Generation ended before the tool call was closed.")
                    if body.startswith(opener):
                        block = self._parser.call_block(full_text, body_start)
                        if block is None:
                            if not final:
                                break
                            raise ToolParseError("Generation ended before the tool call was closed.")
                        payload, end = block
                        try:
                            calls = self._parser.decode_calls(payload, self._tools)
                        except ToolParseError as exc:
                            # Consume the engine stream before reporting failure,
                            # preserving usage and the actual budget finish reason.
                            self._error = exc
                            if final:
                                raise
                            return None
                        self._pending.extend(calls)
                        self._pos = end
                        return self.process(full_text, final=final)
                    # No call structure follows this marker: emit it literally.
                    text.append(marker)
                    self._pos = pos + len(marker)
                    continue
            text.append(char)
            self._pos += 1
        return DeltaMessage(content="".join(text)) if text else None


def _parse_qwen_text(parser, full_text, tools):
    stream = QwenStreamToolParser(parser, tools)
    content, calls = [], []
    while (delta := stream.process(full_text, final=True)) is not None:
        if delta.content:
            content.append(delta.content)
        for call in delta.tool_calls or []:
            calls.append(ToolCall(id=call.id, function=FunctionCall(
                name=call.function.name, arguments=call.function.arguments,
            )))
    return "".join(content) or None, calls


class QwenToolParser(ToolParser):
    """Qwen / Hermes style: zero or more ``<tool_call>{json}</tool_call>``
    blocks, optionally preceded by natural-language content."""

    name = "qwen"
    _START = "<tool_call>"
    _CALL_START = "{"

    def parse(self, full_text: str, tools=None) -> Tuple[Optional[str], List[ToolCall]]:
        return _parse_qwen_text(self, full_text, tools)

    def call_block(self, text, start):
        # A closing tag inside a JSON string is part of an argument, not the
        # boundary of the call. Decode the JSON before looking for the tag.
        try:
            _, size = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
        end = start + size
        closing = re.match(r"\s*</tool_call>", text[end:])
        return (text[start:end], end + closing.end()) if closing else None

    def decode_calls(self, block, tools):
        obj = json.loads(block)
        name = obj.get("name") if isinstance(obj, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ToolParseError("Generated tool call has no valid function name.")
        return [ToolCall(function=FunctionCall(
            name=name, arguments=_dump_arguments(obj.get("arguments", {})),
        ))]

    def stream_parser(self, tools=None) -> StreamToolParser:
        return QwenStreamToolParser(self, tools)


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

    Each ``<parameter>`` value comes out of the XML as raw text (no type
    information). We type-correct it against the tool's JSON schema via
    :func:`_coerce_args`: ``string`` params stay strings, while
    ``integer``/``number``/``boolean``/``array``/``object`` params are coerced
    to their real types. When no schema is supplied the values stay strings.

    Doing schema-*less* ``json.loads`` on every value (turning ``"4"`` into
    ``int`` and ``"true"`` into ``bool`` unconditionally) is wrong -- it breaks
    string-typed params (e.g. BFCL's Java/JS categories, where every value is a
    string). Keeping *everything* a string is equally wrong -- it breaks numeric
    Python params. The schema is the only reliable signal.
    """

    name = "qwen3"
    _START = "<tool_call>"
    _CALL_START = "<function="
    _FUNC_RE = re.compile(
        r"<function=(?P<name>[^>\n]+)>(?P<body>.*?)</function>"
        r"(?=\s*(?:<function=|\Z))", re.DOTALL
    )
    # Tolerate a missing/garbled closing ``</parameter>``: a value runs until
    # its ``</parameter>``, the next ``<parameter=``, or the end of the function
    # body (Qwen sometimes drops the final closing tag).
    _PARAM_RE = re.compile(
        r"<parameter=(?P<key>[^>\n]+)>"
        r"(?P<val>.*?)"
        r"(?:</parameter>(?=\s*(?:<parameter=|\Z))|(?=<parameter=)|\Z)",
        re.DOTALL,
    )

    def parse(self, full_text: str, tools=None) -> Tuple[Optional[str], List[ToolCall]]:
        return _parse_qwen_text(self, full_text, tools)

    def call_block(self, text, start):
        closing = re.search(r"</function>\s*</tool_call>", text[start:])
        if closing is None:
            return None
        end = start + closing.start() + len("</function>")
        return text[start:end], start + closing.end()

    def decode_calls(self, block, tools):
        tool_calls: List[ToolCall] = []
        consumed = 0
        for fm in self._FUNC_RE.finditer(block):
            if block[consumed:fm.start()].strip():
                raise ToolParseError("Generated tool call contains malformed function markup.")
            name = fm.group("name").strip()
            if not name:
                raise ToolParseError("Generated tool call has no valid function name.")
            args = {}
            body = fm.group("body")
            parameter_end = 0
            for pm in self._PARAM_RE.finditer(body):
                if body[parameter_end:pm.start()].strip():
                    raise ToolParseError("Generated tool call contains malformed parameter markup.")
                key = pm.group("key").strip()
                if key:
                    args[key] = pm.group("val").strip()
                parameter_end = pm.end()
            if body[parameter_end:].strip():
                raise ToolParseError("Generated tool call contains malformed parameter markup.")
            # Values come out of the XML as raw strings; type-correct them
            # against the tool schema (string params stay strings).
            args = _coerce_args(args, tools, name)
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(name=name, arguments=_dump_arguments(args))
                )
            )
            consumed = fm.end()
        if not tool_calls or block[consumed:].strip():
            raise ToolParseError("Generated tool call contains malformed function markup.")
        return tool_calls

    def stream_parser(self, tools=None) -> StreamToolParser:
        return QwenStreamToolParser(self, tools)


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

    def parse(self, full_text: str, tools=None) -> Tuple[Optional[str], List[ToolCall]]:
        if self._SECTION_START not in full_text:
            return full_text, []

        tool_calls: List[ToolCall] = []
        for m in self._CALL_RE.finditer(full_text):
            name = self._name_from_id(m.group("fid"))
            if not name:
                continue
            # Kimi emits a JSON argument blob; pass it through unchanged.
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

    def stream_parser(self, tools=None) -> StreamToolParser:
        return StreamToolParser(self, tools)


class DeepSeekToolParser(ToolParser):
    """DeepSeek-V3.2/V4 DSML tool calls.

    V3.2 uses a ``<｜DSML｜function_calls>`` outer block while V4 uses
    ``<｜DSML｜tool_calls>``.  Both contain ``invoke`` elements with typed
    ``parameter`` children::

        <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜function_calls>

    Parsing and typing are delegated to the model's own reference decoder
    (``encoding_dsv32.parse_message_from_completion_text``) when available, so
    the OpenAI ``arguments`` JSON follows the checkpoint's native encoding.
    ``encoder`` is injected by the server (loaded from the checkpoint's
    ``encoding/`` dir). When it's absent we fall back to a lenient regex that
    extracts names + parameters without the reference typing.
    """

    name = "deepseek"
    _DSML = "｜DSML｜"
    # The model emits the block with the ``｜DSML｜`` special token, but some
    # checkpoints/decodes drop it and produce the plain ``<function_calls>`` form.
    # Detect and parse BOTH: treat the ``｜DSML｜`` prefix as optional everywhere.
    _BLOCK_TAGS = ("function_calls", "tool_calls")
    _INVOKE_RE = re.compile(
        r"<(?:｜DSML｜)?invoke\s+name=\"(?P<name>[^\"]+)\">(?P<body>.*?)"
        r"</(?:｜DSML｜)?invoke>",
        re.DOTALL,
    )
    _PARAM_RE = re.compile(
        r"<(?:｜DSML｜)?parameter\s+name=\"(?P<key>[^\"]+)\"\s+"
        r"string=\"(?P<is_str>true|false)\">(?P<value>.*?)</(?:｜DSML｜)?parameter>",
        re.DOTALL,
    )

    def __init__(self, encoder=None) -> None:
        self._encoder = encoder

    def _block_start(self, full_text: str) -> int:
        """Index of the tool-call block (DSML or plain form), or -1."""
        positions = [
            i
            for tag in self._BLOCK_TAGS
            for marker in (f"<{self._DSML}{tag}", f"<{tag}")
            if (i := full_text.find(marker)) != -1
        ]
        return min(positions, default=-1)

    def content_prefix(self, full_text: str) -> str:
        i = self._block_start(full_text)
        if i != -1:
            return full_text[:i]

        # Streaming can split a DSML marker across decoded token chunks. Keep
        # a trailing prefix such as ``<｜DSML｜tool_c`` buffered until it either
        # becomes a complete marker or is proven to be ordinary content.
        markers = tuple(
            marker
            for tag in self._BLOCK_TAGS
            for marker in (f"<{self._DSML}{tag}", f"<{tag}")
        )
        withheld = 0
        for marker in markers:
            for size in range(min(len(full_text), len(marker) - 1), 0, -1):
                if full_text.endswith(marker[:size]):
                    withheld = max(withheld, size)
                    break
        return full_text[:-withheld] if withheld else full_text

    def _parse_official(self, full_text: str):
        """Use the checkpoint's reference decoder; returns tool_calls or raises.

        The reference parser requires the assistant text to terminate with the
        EOS token and rejects trailing content, so append EOS when the streamed
        text (stop token stripped) lacks it. It also requires the ``｜DSML｜``
        special token, so re-insert it if the model emitted the plain form.
        """
        eos = "<｜end▁of▁sentence｜>"
        text = full_text
        if self._DSML not in text:
            # Restore the DSML token the strict decoder expects.
            for tag in (*self._BLOCK_TAGS, "invoke", "parameter"):
                text = text.replace(f"<{tag}", f"<{self._DSML}{tag}")
                text = text.replace(f"</{tag}", f"</{self._DSML}{tag}")
        text = text if text.endswith(eos) else text + eos
        parsed = self._encoder.parse_message_from_completion_text(
            text, thinking_mode="chat"
        )
        calls: List[ToolCall] = []
        for tc in parsed.get("tool_calls") or []:
            fn = tc["function"]
            calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=fn["name"],
                        arguments=_dump_arguments(fn["arguments"]),
                    )
                )
            )
        return parsed.get("content") or None, calls

    def _parse_regex(self, full_text: str):
        """Encoder-free fallback: extract name + params via regex (no reference
        typing -- string params stay strings, others JSON-decoded best-effort)."""
        calls: List[ToolCall] = []
        for m in self._INVOKE_RE.finditer(full_text):
            args: Dict[str, Any] = {}
            for pm in self._PARAM_RE.finditer(m.group("body")):
                key, is_str, value = (
                    pm.group("key"),
                    pm.group("is_str"),
                    pm.group("value"),
                )
                if is_str == "true":
                    args[key] = value
                else:
                    try:
                        args[key] = json.loads(value)
                    except (TypeError, ValueError):
                        args[key] = value
            calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=m.group("name"),
                        arguments=_dump_arguments(args),
                    )
                )
            )
        return self.content_prefix(full_text).strip() or None, calls

    def parse(self, full_text: str, tools=None) -> Tuple[Optional[str], List[ToolCall]]:
        if self._block_start(full_text) == -1:
            return full_text, []

        if self._encoder is not None:
            try:
                return self._parse_official(full_text)
            except Exception:
                # Reference parser is strict about formatting; fall back to the
                # lenient regex rather than dropping the tool call entirely.
                pass
        return self._parse_regex(full_text)

    def stream_parser(self, tools=None) -> StreamToolParser:
        return StreamToolParser(self, tools)


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
_AVAILABLE_NAMES = (
    "qwen",
    "qwen2",
    "qwen2.5",
    "qwen3",
    "qwen3.5",
    "hermes",
    "kimi",
    "deepseek",
)


def get_tool_parser(
    architecture: Optional[str] = None,
    name: Optional[str] = None,
    encoder=None,
) -> Optional[ToolParser]:
    """Resolve the tool-call parser.

    An explicit ``name`` takes precedence; otherwise the parser is auto-detected
    from the model ``architecture`` string. The qwen-family names ("qwen",
    "qwen2", "qwen3", ...) resolve to the Hermes or XML variant based on the
    architecture; "hermes" forces Hermes and "qwen3.5" forces XML. ``encoder``
    (the DeepSeek reference message decoder, loaded from the checkpoint) is
    injected into :class:`DeepSeekToolParser` for exact-typed parsing. Returns
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
        if n == "deepseek":
            return DeepSeekToolParser(encoder=encoder)
        raise ValueError(
            f"Unknown tool-call parser '{name}'. Available: {list(_AVAILABLE_NAMES)}"
        )

    if architecture:
        arch = architecture.lower()
        if "qwen" in arch:
            return _qwen_parser_for_arch(architecture)
        if "kimi" in arch:
            return KimiToolParser()
        if any(
            marker in arch
            for marker in ("deepseekv32", "deepseek_v32", "deepseekv4", "deepseek_v4")
        ):
            return DeepSeekToolParser(encoder=encoder)

    return None
