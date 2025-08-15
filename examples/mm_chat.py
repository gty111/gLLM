# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An example showing how to use vLLM to serve multimodal models
and run online serving with OpenAI client.

Launch the vLLM server with the following command:

(single image inference with Llava)
vllm serve llava-hf/llava-1.5-7b-hf

(multi-image inference with Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
    --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt '{"image":2}'

(audio inference with Ultravox)
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
    --max-model-len 4096 --trust-remote-code

run the script with
python openai_chat_completion_client_for_multimodal.py --chat-type audio
"""

import base64

import requests
from openai import OpenAI
from openai import APIConnectionError, OpenAI
from openai.pagination import SyncPage
from openai.types.model import Model
from argparse import ArgumentParser

def get_first_model(client: OpenAI) -> str:
    """
    Get the first model from the vLLM server.
    """
    try:
        models: SyncPage[Model] = client.models.list()
    except APIConnectionError as e:
        raise RuntimeError(
            "Failed to get the list of models from the vLLM server at "
            f"{client.base_url} with API key {client.api_key}. Check\n"
            "1. the server is running\n"
            "2. the server URL is correct\n"
            "3. the API key is correct"
        ) from e

    if len(models.data) == 0:
        raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")

    return models.data[0].id


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


# Text-only inference
def run_text_only(model: str, client) -> None:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion.choices[0].message.content
    print("Chat completion output:", result)


# Single-image input inference
def run_single_image(model: str, client) -> None:
    ## Use image url in the payload
    image_url = 'https://2026.eurosys.org/img/EuroSys-2026-logo.png'
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述下这个图片?"},
                    {
                        "type": "image_url",
                        "image_url": {'url': image_url},
                    },
                ],
            }
        ],
        model=model,
        max_tokens=512,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:", result)

# Multi-image input inference
def run_multi_image(model: str, client) -> None:
    image1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    image2 = "https://2026.eurosys.org/img/EuroSys-2026-logo.png"
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {'url':image1},
                    },
                    {
                        "type": "image_url",
                        "image_url": {'url':image2},
                    },
                ],
            }
        ],
        model=model,
        max_tokens=1024,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output:", result)


# Video input inference
def run_video(model: str, client) -> None:
    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
    video_base64 = encode_base64_content_from_url(video_url)

    ## Use video url in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:", result)

    ## Use base64 encoded video in the payload
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded image:", result)



example_function_map = {
    "text-only": run_text_only,
    "single-image": run_single_image,
    "multi-image": run_multi_image,
    "video": run_video,
}


def parse_args():
    parser = ArgumentParser(
        description="Demo on using OpenAI client for online serving with "
        "multimodal language models served with vLLM."
    )
    parser.add_argument(
        "--chat-type",
        "-c",
        type=str,
        default="single-image",
        choices=list(example_function_map.keys()),
        help="Conversation type with multimodal data.",
    )
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=8000
    )
    return parser.parse_args()


def main(args) -> None:
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    chat_type = args.chat_type
    model = get_first_model(client)
    example_function_map[chat_type](model, client)


if __name__ == "__main__":
    args = parse_args()
    main(args)
