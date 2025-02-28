import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description='Chat client')
parser.add_argument("--stream",action="store_true")
parser.add_argument("--port",type=int)
args = parser.parse_args()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = f"http://localhost:{args.port}/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Can you tell a fairy tale?"
    }],
    model=model,
    stream=args.stream,
    max_tokens = 1024
)

if args.stream:
    print("Chat completion results:")
    for i in chat_completion:
        print(i.choices[0].delta.content,end='',flush=True)
    print()
else:
    print(chat_completion)