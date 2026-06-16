from gllm.async_llm_engine import AsyncStream
from gllm.entrypoints.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
)
from gllm.utils import build_usage, get_finish_reason


async def chat_completion_generator(
    stream: AsyncStream, request: ChatCompletionRequest
):
    full_text = ""
    async for text in stream:
        full_text += text
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=full_text),
        finish_reason=get_finish_reason(stream.seq),
    )
    response = ChatCompletionResponse(
        choices=[choice_data],
        usage=build_usage(stream.seq),
        model=request.model,
    )
    return response


async def chat_completion_stream_generator(
    stream: AsyncStream, request: ChatCompletionRequest
):
    async for text in stream:
        if not text:
            continue
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=text)
        )
        chunk = ChatCompletionStreamResponse(choices=[choice_data], model=request.model)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"

    # Final chunk: empty delta carrying the finish_reason, mirroring the OpenAI
    # streaming protocol. Usage is attached when the client opted in via
    # ``stream_options.include_usage`` (default on in our schema).
    final_choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason=get_finish_reason(stream.seq),
    )
    include_usage = (
        request.stream_options is None
        or request.stream_options.include_usage
    )
    final_chunk = ChatCompletionStreamResponse(
        choices=[final_choice],
        model=request.model,
        usage=build_usage(stream.seq) if include_usage else None,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n"
    yield "data: [DONE]\n\n"
