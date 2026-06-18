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
from gllm.entrypoints.tool_parsers import ToolParser
from gllm.utils import build_usage, get_finish_reason


async def chat_completion_generator(
    stream: AsyncStream,
    request: ChatCompletionRequest,
    tool_parser: ToolParser = None,
):
    full_text = ""
    async for text in stream:
        full_text += text

    # Parse model-native tool-call markup into structured ``tool_calls`` when a
    # parser is available AND the request actually offered tools. Without a
    # parser (unknown model) or tools, the raw text passes through as content.
    content = full_text
    tool_calls = []
    if tool_parser is not None and request.tools:
        parsed_content, tool_calls = tool_parser.parse(full_text)
        content = parsed_content if parsed_content is not None else ""

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant", content=content, tool_calls=tool_calls
        ),
        # OpenAI sets finish_reason="tool_calls" when the model chose to call a
        # tool; clients (and benchmarks) branch on this.
        finish_reason=(
            "tool_calls" if tool_calls else get_finish_reason(stream.seq)
        ),
    )
    response = ChatCompletionResponse(
        choices=[choice_data],
        usage=build_usage(stream.seq),
        model=request.model,
    )
    return response


async def chat_completion_stream_generator(
    stream: AsyncStream,
    request: ChatCompletionRequest,
    tool_parser: ToolParser = None,
):
    # When a parser is available and the request offered tools, run the
    # incremental streaming tool-call parser: it accumulates the full text and
    # emits content fragments + tool-call name/argument fragments (tied by
    # ``index``), mirroring vLLM / SGLang. Otherwise stream raw text deltas.
    streaming = tool_parser is not None and bool(request.tools)
    sp = tool_parser.stream_parser() if streaming else None
    full_text = ""

    async for text in stream:
        if not text:
            continue
        if streaming:
            full_text += text
            delta = sp.process(full_text)
            if delta is None:
                continue
            choice_data = ChatCompletionResponseStreamChoice(index=0, delta=delta)
        else:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=text)
            )
        chunk = ChatCompletionStreamResponse(choices=[choice_data], model=request.model)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f"data: {data}\n\n"

    # Final chunk: empty delta carrying the finish_reason, mirroring the OpenAI
    # streaming protocol. Usage is attached when the client opted in via
    # ``stream_options.include_usage`` (default on in our schema).
    final_reason = get_finish_reason(stream.seq)
    if streaming and sp.has_tool_calls():
        final_reason = "tool_calls"
    final_choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason=final_reason,
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
