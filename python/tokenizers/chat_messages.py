"""OpenAI chat messages -> the shape model-native chat templates expect."""

import json


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


def normalize_chat_template_messages(messages, tool_arguments: bool = True) -> None:
    """Normalize modern OpenAI messages for model-native chat templates.

    ``developer`` is the current OpenAI instruction role, while many local
    templates still only recognize the equivalent legacy ``system`` role.
    Tool-call arguments also need conversion from their JSON wire string to the
    mapping expected by several templates; pass ``tool_arguments=False`` for
    encoders that take the wire string as is.
    """
    if tool_arguments:
        normalize_chat_template_tool_arguments(messages)
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "developer":
            message["role"] = "system"
