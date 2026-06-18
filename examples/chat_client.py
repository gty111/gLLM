import argparse

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


def build_chat_template_kwargs(enabled: bool) -> dict:
    return {"thinking": enabled, "enable_thinking": enabled}


print(
    "\nWelcome to the chatbot!\n"
    "Type '\\exit' to exit the chatbot.\n"
    "Type '\\clear' to clear the chatbot's history.\n"
    "Type '\\think' to enable thinking, '\\nothink' to disable it.\n"
    f"Thinking is currently {'ON' if thinking else 'OFF'}.\n"
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
    messages.append({"role": "user", "content": prompt})
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        max_tokens=args.num_tokens,
        extra_body={"chat_template_kwargs": build_chat_template_kwargs(thinking)},
    )
    reply = ""
    print()
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content or ""
        reply += content
        print(content, end="", flush=True)
    print()
    print()
    messages.append({"role": "assistant", "content": reply})
