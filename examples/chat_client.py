import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description='Chat client')
parser.add_argument("--num-tokens", type=int, default=2048)
parser.add_argument("--port", type=int)
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

print("\nWelcome to the chatbot!\n"
      "Type '\\exit' to exit the chatbot.\n"
      "Type '\\clear' to clear the chatbot's history.\n")

while True:
    prompt = input('>>> ')
    if prompt == '\\exit':
        break
    elif prompt == '\\clear':
        messages = []
    messages.append({'role': 'user', 'content': prompt})
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        max_tokens=args.num_tokens,
    )
    reply = ''
    print()
    for i in chat_completion:
        reply += i.choices[0].delta.content
        print(i.choices[0].delta.content, end='', flush=True)
    print()
    print()
    messages.append({'role': 'assistant', 'content': reply})
