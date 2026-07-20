from gllm.async_llm_engine import AsyncStream
from gllm.entrypoints.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
)
from gllm.tokenizers.tool_parsers import ToolParser
from gllm.utils import build_usage, get_finish_reason


def _token_str(entry, as_token_ids):
    return f"token_id:{entry['token_id']}" if as_token_ids else entry["token"]


def _build_chat_logprobs(entries, as_token_ids):
    """Assemble an OpenAI ``ChatCompletionLogProbs`` from per-token entries.

    ``as_token_ids`` mirrors the request's ``return_tokens_as_token_ids``: when
    set, every token field is rendered as ``token_id:<id>`` so non-UTF-8 tokens
    stay JSON-safe.
    """
    content = []
    for e in entries:
        top = [
            ChatCompletionLogProb(
                token=_token_str(t, as_token_ids),
                logprob=t["logprob"],
                bytes=t["bytes"],
            )
            for t in e["top_logprobs"]
        ]
        content.append(
            ChatCompletionLogProbsContent(
                token=_token_str(e, as_token_ids),
                logprob=e["logprob"],
                bytes=e["bytes"],
                top_logprobs=top,
            )
        )
    return ChatCompletionLogProbs(content=content)


async def chat_completion_generator(
    stream: AsyncStream,
    request: ChatCompletionRequest,
    tool_parser: ToolParser = None,
):
    full_text = ""
    entries = []
    prompt_logprobs = None
    async for item in stream:
        full_text += item.text
        if item.logprob is not None:
            entries.append(item.logprob)
        if item.prompt_logprobs is not None:
            prompt_logprobs = item.prompt_logprobs

    # Parse model-native tool-call markup into structured ``tool_calls`` when a
    # parser is available AND the request actually offered tools. Without a
    # parser (unknown model) or tools, the raw text passes through as content.
    content = full_text
    tool_calls = []
    if tool_parser is not None and request.tools:
        parsed_content, tool_calls = tool_parser.parse(full_text, request.tools)
        content = parsed_content if parsed_content is not None else ""

    logprobs = None
    if entries:
        logprobs = _build_chat_logprobs(
            entries, bool(request.return_tokens_as_token_ids)
        )

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant", content=content, tool_calls=tool_calls
        ),
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
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
    sp = tool_parser.stream_parser(request.tools) if streaming else None
    full_text = ""
    as_token_ids = bool(request.return_tokens_as_token_ids)

    async for item in stream:
        text = item.text
        # Keep chunks that carry text OR a logprob (a multi-byte token can
        # produce an empty text delta whose logprob must still be reported) OR
        # the one-shot prompt_logprobs payload.
        if not text and item.logprob is None and item.prompt_logprobs is None:
            continue
        logprobs = None
        if item.logprob is not None:
            logprobs = _build_chat_logprobs([item.logprob], as_token_ids)
        prompt_logprobs = item.prompt_logprobs
        if streaming:
            full_text += text
            delta = sp.process(full_text)
            if delta is None and logprobs is None and prompt_logprobs is None:
                continue
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta or DeltaMessage(),
                logprobs=logprobs,
                prompt_logprobs=prompt_logprobs,
            )
        else:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=text),
                logprobs=logprobs,
                prompt_logprobs=prompt_logprobs,
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
