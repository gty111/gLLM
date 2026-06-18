import argparse
import json
import re

from openai import OpenAI

parser = argparse.ArgumentParser(description="Chat client")
parser.add_argument("--num-tokens", type=int, default=2048)
parser.add_argument("--port", type=int)
parser.add_argument(
    "--thinking",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable the model's <think> reasoning block (default: off). "
    "Toggle at runtime with '\\think' / '\\nothink'.",
)
parser.add_argument(
    "--tools",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Expose a demo function-calling toolset (get_weather, calculate) so "
    "the model can emit tool calls. Toggle at runtime with '\\tools' / "
    "'\\notools'.",
)
args = parser.parse_args()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://0.0.0.0:{args.port}/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

messages = []
# Reasoning models gate "thinking" via different chat-template variable names
# (Qwen3/3.5 read ``enable_thinking``; Kimi-K2.5 reads ``thinking``), so we send
# both -- the template silently ignores whichever it doesn't define.
thinking = args.thinking
use_tools = args.tools


def build_chat_template_kwargs(enabled: bool) -> dict:
    return {"thinking": enabled, "enable_thinking": enabled}


# --- Demo toolset --------------------------------------------------------
# OpenAI-style tool schemas advertised to the model via the ``tools`` request
# field. The server's chat template renders them into the model's native
# tool-declaration block so the model knows what it may call.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Beijing"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '2 * (3 + 4)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    """Mock local execution of a tool call -> JSON string result."""
    if name == "get_weather":
        city = arguments.get("city", "unknown")
        unit = arguments.get("unit", "celsius")
        temp = 23 if unit == "celsius" else 73
        return json.dumps({"city": city, "temperature": temp, "unit": unit, "sky": "sunny"})
    if name == "calculate":
        expr = arguments.get("expression", "")
        try:
            # Tiny safe-ish eval for the demo: digits/operators/space/parens only.
            if re.fullmatch(r"[0-9+\-*/(). ]+", expr):
                return json.dumps({"expression": expr, "result": eval(expr)})  # noqa: S307
            return json.dumps({"error": "unsupported characters in expression"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"unknown tool {name}"})


def stream_reply(msgs):
    """Stream one completion. Prints assistant text live and reassembles any
    streamed tool-call deltas (the server emits OpenAI-style incremental
    ``tool_calls``: a name in the first chunk, then ``arguments`` fragments,
    tied together by ``index``).

    Returns ``(content, tool_calls)`` where ``tool_calls`` is a list of
    ``{"id", "name", "arguments"(str)}`` dicts (empty if the model just chatted).
    """
    kwargs = dict(
        messages=msgs,
        model=model,
        stream=True,
        max_tokens=args.num_tokens,
        extra_body={"chat_template_kwargs": build_chat_template_kwargs(thinking)},
    )
    if use_tools:
        kwargs["tools"] = TOOLS
    chat_completion = client.chat.completions.create(**kwargs)

    content = ""
    # index -> {"id", "name", "arguments"}
    acc_calls: dict = {}
    print()
    for chunk in chat_completion:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)
        for tc in (delta.tool_calls or []):
            slot = acc_calls.setdefault(
                tc.index, {"id": None, "name": "", "arguments": ""}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    print()
    print()
    tool_calls = [acc_calls[i] for i in sorted(acc_calls)]
    return content, tool_calls


print(
    "\nWelcome to the chatbot!\n"
    "Type '\\exit' to exit the chatbot.\n"
    "Type '\\clear' to clear the chatbot's history.\n"
    "Type '\\think' to enable thinking, '\\nothink' to disable it.\n"
    "Type '\\tools' to enable tool calling, '\\notools' to disable it.\n"
    "\nAvailable demo tools (used when tool calling is ON):\n"
    + "".join(
        f"  - {t['function']['name']}: {t['function']['description']}\n"
        for t in TOOLS
    )
    + "  e.g. \"What's the weather in Beijing?\" or \"Calculate (3 + 4) * 5\"\n"
    f"\nThinking is currently {'ON' if thinking else 'OFF'}; "
    f"tools are {'ON' if use_tools else 'OFF'}.\n"
)

while True:
    prompt = input(">>> ")
    if prompt == "\\exit":
        break
    elif prompt == "\\clear":
        messages = []
        continue
    elif prompt == "\\think":
        thinking = True
        print("[thinking ON]\n")
        continue
    elif prompt == "\\nothink":
        thinking = False
        print("[thinking OFF]\n")
        continue
    elif prompt == "\\tools":
        use_tools = True
        print("[tools ON]\n")
        continue
    elif prompt == "\\notools":
        use_tools = False
        print("[tools OFF]\n")
        continue

    messages.append({"role": "user", "content": prompt})
    content, tool_calls = stream_reply(messages)

    # Tool-calling loop: while the model requests tools, execute them locally,
    # append the call + results to the history, and let the model continue.
    # Cap the loop so a model that keeps re-calling can't spin forever.
    for _ in range(5):
        if not tool_calls:
            # Plain assistant turn: record and stop.
            messages.append({"role": "assistant", "content": content})
            break

        # Record the assistant's tool-call turn in OpenAI format.
        assistant_msg = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": c["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for i, c in enumerate(tool_calls)
            ],
        }
        messages.append(assistant_msg)

        # Execute each call and append its result as a ``tool`` message.
        for c in tool_calls:
            try:
                arguments = json.loads(c["arguments"]) if c["arguments"] else {}
            except json.JSONDecodeError:
                arguments = {}
            result = run_tool(c["name"], arguments)
            print(f"[tool {c['name']}({arguments}) -> {result}]\n")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"] or f"call_{c['name']}",
                    "content": result,
                }
            )

        # Let the model continue with the tool results.
        content, tool_calls = stream_reply(messages)
