from gllm.async_llm_engine import AsyncStream
from gllm.entrypoints.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from gllm.utils import build_usage, get_finish_reason


async def completion_generator(stream: AsyncStream, request: CompletionRequest):
    full_text = ""
    async for text in stream:
        full_text += text
    choice_data = CompletionResponseChoice(
        index=0, text=full_text, finish_reason=get_finish_reason(stream.seq)
    )
    completion = CompletionResponse(
        choices=[choice_data], model=request.model, usage=build_usage(stream.seq)
    )
    return completion


async def completion_stream_generator(stream: AsyncStream, request: CompletionRequest):
    async for text in stream:
        if not text:
            continue
        choice_data = CompletionResponseStreamChoice(index=0, text=text)
        chunk = CompletionStreamResponse(choices=[choice_data], model=request.model)
        data = chunk.model_dump_json(exclude_unset=False)
        yield f"data: {data}\n\n"

    # Final chunk: empty text carrying the finish_reason; usage attached when
    # the client opted in via ``stream_options.include_usage`` (default on).
    final_choice = CompletionResponseStreamChoice(
        index=0, text="", finish_reason=get_finish_reason(stream.seq)
    )
    include_usage = (
        request.stream_options is None
        or request.stream_options.include_usage
    )
    final_chunk = CompletionStreamResponse(
        choices=[final_choice],
        model=request.model,
        usage=build_usage(stream.seq) if include_usage else None,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_unset=False)}\n\n"
    yield "data: [DONE]\n\n"
