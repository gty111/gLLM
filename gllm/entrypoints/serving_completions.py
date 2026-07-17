from gllm.async_llm_engine import AsyncStream
from gllm.entrypoints.protocol import (
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from gllm.utils import build_usage, get_finish_reason


def _build_completion_logprobs(entries, text_offset_start=0):
    """Assemble an OpenAI ``CompletionLogProbs`` from per-token entries.

    Each entry is the dict produced by ``LLM._make_logprob_entry``. The
    ``top_logprobs`` map always includes the sampled token (mirroring OpenAI,
    which reports the chosen token even when it is outside the top-k).
    """
    lp = CompletionLogProbs()
    offset = text_offset_start
    for e in entries:
        lp.tokens.append(e["token"])
        lp.token_logprobs.append(e["logprob"])
        top = {t["token"]: t["logprob"] for t in e["top_logprobs"]}
        top.setdefault(e["token"], e["logprob"])
        lp.top_logprobs.append(top)
        lp.text_offset.append(offset)
        offset += len(e["token"])
    return lp


async def completion_generator(stream: AsyncStream, request: CompletionRequest):
    full_text = ""
    entries = []
    prompt_logprobs = None
    async for item in stream:
        full_text += item.text
        if item.logprob is not None:
            entries.append(item.logprob)
        if item.prompt_logprobs is not None:
            prompt_logprobs = item.prompt_logprobs
    logprobs = _build_completion_logprobs(entries) if entries else None
    choice_data = CompletionResponseChoice(
        index=0,
        text=full_text,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        finish_reason=get_finish_reason(stream.seq),
    )
    completion = CompletionResponse(
        choices=[choice_data], model=request.model, usage=build_usage(stream.seq)
    )
    return completion


async def completion_stream_generator(stream: AsyncStream, request: CompletionRequest):
    text_offset = 0
    async for item in stream:
        # Keep chunks that carry text OR a logprob (a multi-byte token can
        # produce an empty text delta whose logprob must still be reported) OR
        # the one-shot prompt_logprobs payload.
        if not item.text and item.logprob is None and item.prompt_logprobs is None:
            continue
        logprobs = None
        if item.logprob is not None:
            logprobs = _build_completion_logprobs([item.logprob], text_offset)
        text_offset += len(item.text)
        choice_data = CompletionResponseStreamChoice(
            index=0,
            text=item.text,
            logprobs=logprobs,
            prompt_logprobs=item.prompt_logprobs,
        )
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
