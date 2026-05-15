from gllm.async_llm_engine import AsyncStream
from gllm.entrypoints.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)

# Common stop strings that should be stripped from model output
_STOP_STRINGS = ["<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"]


def _strip_stop_strings(text: str) -> str:
    for s in _STOP_STRINGS:
        if text.endswith(s):
            text = text[: -len(s)]
    return text


async def chat_completion_generator(
    stream: AsyncStream, request: ChatCompletionRequest
):
    full_text = ""
    async for text in stream:
        full_text += text
    full_text = _strip_stop_strings(full_text)
    choice_data = ChatCompletionResponseChoice(
        index=0, message=ChatMessage(role="assistant", content=full_text)
    )
    response = ChatCompletionResponse(
        choices=[choice_data], usage=UsageInfo(), model=request.model
    )
    return response


async def chat_completion_stream_generator(
    stream: AsyncStream, request: ChatCompletionRequest
):
    async for text in stream:
        # Strip stop strings if they appear at the end of a chunk
        text = _strip_stop_strings(text)
        if not text:
            continue
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=text)
        )
        chunk = ChatCompletionStreamResponse(choices=[choice_data], model=request.model)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"
