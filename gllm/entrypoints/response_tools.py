"""Responses tool types represented through model-native function calling.

Namespaces remain distinct on the wire. Custom tools use a single string
argument internally; callers still receive custom_tool_call items and events.
Grammar formats also constrain decoding for supported model envelopes; the
final check remains a defense against parser and backend mismatches.
"""

import json
import re
from functools import lru_cache

from gllm.utils import random_uuid

TOOL_SEARCH_NAME = "tool_search"


def validate_client_search_item(item, param):
    if item.get("execution") != "client":
        raise ValueError(param, "Only client-executed tool search is supported.")
    if not isinstance(item.get("call_id"), str) or not item["call_id"]:
        raise ValueError(f"{param}.call_id", "Tool search items require a nonempty call_id.")
    if item.get("status", "completed") != "completed":
        raise ValueError(f"{param}.status", "Tool search history must be completed.")


def search_output_specs(item, param):
    validate_client_search_item(item, param)
    tools = item.get("tools")
    if not isinstance(tools, list):
        raise ValueError(f"{param}.tools", "Tool search output must contain a tools array.")
    return _tool_specs(tools, f"{param}.tools", loaded=True)


@lru_cache(maxsize=64)
def _grammar(syntax, definition):
    if syntax == "regex":
        return re.compile(definition).fullmatch
    if syntax == "lark":
        from lark import Lark

        # The Responses Lark dialect permits empty line regexes (used by
        # Codex apply_patch). Python Lark forbids zero-width lexer terminals;
        # express the same language as a nonempty token plus an empty rule.
        tokens = r'"(?:\\.|[^"\\])*"|#[^\n]*|//[^\n]*|/(?:\\.|[^/\\\n])+/[imslux]*'
        definition = re.sub(
            tokens,
            lambda match: {"/(.*)/": "/(.+)/?", "/.*/": "/.+/?"}.get(match[0], match[0]),
            definition,
        )
        parser = Lark(definition, parser="earley")

        def matches(value):
            from lark import UnexpectedInput

            try:
                parser.parse(value)
                return True
            except UnexpectedInput:
                return False

        def explain(value):
            from lark import UnexpectedInput

            try:
                parser.parse(value)
            except UnexpectedInput as exc:
                if getattr(exc, "line", -1) > 0:
                    return f"line {exc.line}, column {exc.column}"
                return "end of input"
            return "unknown position"

        matches.explain = explain

        return matches
    raise ValueError("Unsupported custom tool grammar syntax.")


def _tool_specs(tools, root="tools", *, loaded=False):
    result = {}

    def add(tool, param, namespace=None):
        if not isinstance(tool, dict):
            raise ValueError(param, "Tools must be objects.")
        kind = tool.get("type")
        if kind == "tool_search":
            if namespace is not None or loaded:
                raise ValueError(param, "tool_search must be a top-level request tool.")
            if tool.get("execution") != "client":
                raise ValueError(param, "Only client-executed tool search is supported.")
            parameters = tool.get("parameters")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError(f"{param}.parameters", "Client tool search requires an object argument schema.")
            if TOOL_SEARCH_NAME in result:
                raise ValueError(param, f"Ambiguous tool name: {TOOL_SEARCH_NAME}.")
            result[TOOL_SEARCH_NAME] = (tool, None)
            return
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(param, "Tool names must be nonempty strings.")
        if kind == "namespace" and namespace is None:
            if not isinstance(tool.get("description"), str):
                raise ValueError(param, "A namespace must contain a description string.")
            children = tool.get("tools")
            if not isinstance(children, list):
                raise ValueError(param, "A namespace must contain a tools array.")
            for index, child in enumerate(children):
                add(child, f"{param}.tools.{index}", name)
            return
        if kind not in ("function", "custom"):
            raise ValueError(param, "Only function/custom tools and their namespaces are supported.")
        native_name = f"{namespace}.{name}" if namespace else name
        if native_name in result:
            raise ValueError(param, f"Ambiguous tool name: {native_name}.")
        if kind == "custom":
            fmt = tool.get("format") or {"type": "text"}
            if not isinstance(fmt, dict) or fmt.get("type") not in ("text", "grammar"):
                raise ValueError(param, "Unsupported custom tool format.")
            if fmt["type"] == "grammar":
                try:
                    _grammar(fmt["syntax"], fmt["definition"])
                except Exception as exc:
                    raise ValueError(param, f"Invalid custom tool grammar: {exc}") from exc
        result[native_name] = (tool, namespace)

    for index, tool in enumerate(tools or []):
        add(tool, f"{root}.{index}")
    return result


def tool_specs(tools, input_items=None):
    """Resolve callable tools, including discoveries replayed in Responses history."""
    catalog = _tool_specs(tools)
    result = {name: spec for name, spec in catalog.items()
              if not spec[0].get("defer_loading", False)}
    for index, item in enumerate(input_items if isinstance(input_items, list) else []):
        if not isinstance(item, dict) or item.get("type") != "tool_search_output":
            continue
        for name, spec in search_output_specs(item, f"input.{index}").items():
            previous = catalog.get(name)
            if previous is not None:
                # A discovered tool may also be declared as deferred or replayed
                # multiple times, but conflicting schemas cannot share a name.
                old = {k: v for k, v in previous[0].items() if k != "defer_loading"}
                new = {k: v for k, v in spec[0].items() if k != "defer_loading"}
                if old != new or previous[1] != spec[1]:
                    raise ValueError(f"input.{index}.tools", f"Conflicting tool definition: {name}.")
            catalog[name] = spec
            result[name] = spec
    return result


def chat_tools(tools, input_items=None):
    translated = []
    for name, (tool, namespace) in tool_specs(tools, input_items).items():
        description = tool.get("description") or ""
        parameters = tool.get("parameters")
        if tool["type"] == "custom":
            description += (
                "\nPass the complete raw tool input as the string argument `input`."
                " Do not wrap the string contents in another JSON object."
            )
            fmt = tool.get("format") or {}
            if fmt.get("type") == "grammar":
                description += f"\nThe input must match this {fmt['syntax']} grammar:\n{fmt['definition']}"
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
        translated.append({
            "type": "function",
            "function": {
                "name": name, "description": description,
                "parameters": parameters, "strict": tool.get("strict"),
            },
        })
    return translated or None


def custom_tool_formats(tools, input_items=None):
    return {name: tool.get("format") or {"type": "text"}
            for name, (tool, _) in tool_specs(tools, input_items).items() if tool["type"] == "custom"}


def bind_custom_parser(parser, tools, input_items=None):
    from gllm.tokenizers.tool_parsers import Qwen3ToolParser

    formats = custom_tool_formats(tools, input_items)
    if isinstance(parser, Qwen3ToolParser) and formats:
        return Qwen3ToolParser(custom_formats=formats)
    return parser


def output_tool_call(tool_call, specs):
    function = tool_call.function
    name = function.name
    if name not in specs:
        raise ValueError(f"Model returned an undeclared tool: {name}.")
    tool, namespace = specs[name]
    if tool["type"] == "tool_search":
        try:
            arguments = json.loads(function.arguments or "")
        except (ValueError, TypeError) as exc:
            raise ValueError("Tool search arguments must be a JSON object.") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool search arguments must be a JSON object.")
        return {"id": f"ts_{random_uuid()}", "type": "tool_search_call",
                "call_id": tool_call.id or f"call_{random_uuid()}",
                "execution": "client", "status": "completed", "arguments": arguments}
    item = {"id": f"fc_{random_uuid()}", "call_id": tool_call.id or f"call_{random_uuid()}",
            "name": tool["name"]}
    if namespace:
        item["namespace"] = namespace
    arguments = function.arguments or ""
    if tool["type"] == "custom":
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError) as exc:
            raise ValueError("Custom tool arguments must contain a string input.") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("input"), str):
            raise ValueError("Custom tool arguments must contain a string input.")
        value = parsed["input"]
        fmt = tool.get("format") or {}
        if fmt.get("type") == "grammar" and not _grammar(fmt["syntax"], fmt["definition"])(value):
            validator = _grammar(fmt["syntax"], fmt["definition"])
            detail = validator.explain(value) if hasattr(validator, "explain") else "regex mismatch"
            raise ValueError(
                f"Generated input does not match the grammar for {name}: {detail} "
                f"(input length {len(value)} characters)."
            )
        item.update(type="custom_tool_call", input=value)
    else:
        item.update(type="function_call", arguments=arguments, status="completed")
    return item
