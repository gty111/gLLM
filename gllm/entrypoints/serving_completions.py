from gllm.async_llm_engine import AsyncStream
from gllm.entrypoints.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
)


async def completion_generator(stream: AsyncStream, request: CompletionRequest):
    full_text = ""
    async for text in stream:
        full_text += text
    choice_data = CompletionResponseChoice(index=0, text=full_text)
    completion = CompletionResponse(
        choices=[choice_data], model=request.model, usage=UsageInfo()
    )
    return completion


async def completion_stream_generator(stream: AsyncStream, request: CompletionRequest):
    async for text in stream:
        choice_data = CompletionResponseStreamChoice(index=0, text=text)
        chunk = CompletionStreamResponse(choices=[choice_data], model=request.model)
        data = chunk.model_dump_json(exclude_unset=False)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"
