import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description='Chat client')
parser.add_argument("--stream",action="store_true")
parser.add_argument("--port",type=int)
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

# Completion API
completion = client.completions.create(
    model=model,
    prompt="A robot may not injure a human being",
    stream=args.stream,
    max_tokens=128)

print("Completion results:")
if args.stream:
    for i in completion:
        print(i.choices[0].text, end='', flush=True)
    print()
else:
    print(completion)
