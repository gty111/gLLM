"""Responses tool types represented through model-native function calling.

Namespaces remain distinct on the wire. Custom tools use a single string
argument internally; callers still receive custom_tool_call items and events.
Grammar formats are validated before publishing a call, not during decoding.
"""

import json
import re
from functools import lru_cache

from gllm.utils import random_uuid


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

        return matches
    raise ValueError("Unsupported custom tool grammar syntax.")


def tool_specs(tools):
    """Map model-facing names to (wire definition, namespace)."""
    result = {}

    def add(tool, param, namespace=None):
        if not isinstance(tool, dict):
            raise ValueError(param, "Tools must be objects.")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(param, "Tool names must be nonempty strings.")
        kind = tool.get("type")
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
        add(tool, f"tools.{index}")
    return result


def chat_tools(tools):
    translated = []
    for name, (tool, namespace) in tool_specs(tools).items():
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


def output_tool_call(tool_call, specs):
    function = tool_call.function
    name = function.name
    if name not in specs:
        raise ValueError(f"Model returned an undeclared tool: {name}.")
    tool, namespace = specs[name]
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
            raise ValueError(f"Generated input does not match the grammar for {name}.")
        item.update(type="custom_tool_call", input=value)
    else:
        item.update(type="function_call", arguments=arguments, status="completed")
    return item
