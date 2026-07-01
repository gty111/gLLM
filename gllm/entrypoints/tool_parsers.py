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
from typing import Any, Dict, List, Optional, Tuple

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


# ── Schema-aware argument coercion ──────────────────────────────────────────
# XML tool-call markup (Qwen3.5) carries no type information -- every
# ``<parameter>`` value is raw text. To emit correctly-typed ``tool_calls`` we
# coerce each value to the type declared in the tool's JSON schema: string-typed
# params stay strings, while integer/number/boolean/array/object params are
# converted. This mirrors the reference Qwen3 parser (vLLM) and is what makes
# native tool-calling scores match: a consumer can turn ``"4"`` into an int when
# the schema says integer, but a spurious ``json.loads`` that turns a string
# ``"4"`` into ``int`` unconditionally breaks string-typed params (e.g. BFCL's
# Java/JS categories, where every value is a string).

_TYPE_ALIASES: Dict[str, str] = {
    "str": "string", "text": "string", "varchar": "string", "char": "string",
    "enum": "string", "int": "integer", "int32": "integer", "int64": "integer",
    "uint": "integer", "long": "integer", "short": "integer", "float": "number",
    "float32": "number", "float64": "number", "double": "number",
    "bool": "boolean", "dict": "object", "arr": "array", "list": "array",
    "sequence": "array", "tuple": "array",
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
        _TYPE_ALIASES.get(k, k)
        for t in schema_types
        for k in [str(t).strip().lower()]
    }
    for candidate in ("null", "integer", "number", "boolean", "object", "array", "string"):
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

    def parse(
        self, full_text: str, tools=None
    ) -> Tuple[Optional[str], List[ToolCall]]:
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


class QwenToolParser(ToolParser):
    """Qwen / Hermes style: zero or more ``<tool_call>{json}</tool_call>``
    blocks, optionally preceded by natural-language content."""

    name = "qwen"
    _START = "<tool_call>"
    _BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def content_prefix(self, full_text: str) -> str:
        return full_text.split(self._START, 1)[0]

    def parse(
        self, full_text: str, tools=None
    ) -> Tuple[Optional[str], List[ToolCall]]:
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
            # Hermes JSON already carries native types; no schema coercion.
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

    def stream_parser(self, tools=None) -> StreamToolParser:
        return StreamToolParser(self, tools)


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
    to their real types. This mirrors the reference Qwen3 parser (vLLM). When no
    schema is supplied the values simply stay strings.

    Doing schema-*less* ``json.loads`` on every value (turning ``"4"`` into
    ``int`` and ``"true"`` into ``bool`` unconditionally) is wrong -- it breaks
    string-typed params (e.g. BFCL's Java/JS categories, where every value is a
    string). Keeping *everything* a string is equally wrong -- it breaks numeric
    Python params. The schema is the only reliable signal.
    """

    name = "qwen3"
    _START = "<tool_call>"
    _FUNC_RE = re.compile(
        r"<function=(?P<name>[^>\n]+)>(?P<body>.*?)</function>", re.DOTALL
    )
    # Tolerate a missing/garbled closing ``</parameter>``: a value runs until
    # its ``</parameter>``, the next ``<parameter=``, or the end of the function
    # body (Qwen sometimes drops the final closing tag).
    _PARAM_RE = re.compile(
        r"<parameter=(?P<key>[^>\n]+)>"
        r"(?P<val>.*?)"
        r"(?:</parameter>|(?=<parameter=)|\Z)",
        re.DOTALL,
    )

    def content_prefix(self, full_text: str) -> str:
        return full_text.split(self._START, 1)[0]

    def parse(
        self, full_text: str, tools=None
    ) -> Tuple[Optional[str], List[ToolCall]]:
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
                    args[key] = pm.group("val").strip()
            # Values come out of the XML as raw strings; type-correct them
            # against the tool schema (string params stay strings).
            args = _coerce_args(args, tools, name)
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name, arguments=_dump_arguments(args)
                    )
                )
            )

        content = self.content_prefix(full_text).strip() or None
        return content, tool_calls

    def stream_parser(self, tools=None) -> StreamToolParser:
        return StreamToolParser(self, tools)


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

    def parse(
        self, full_text: str, tools=None
    ) -> Tuple[Optional[str], List[ToolCall]]:
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
