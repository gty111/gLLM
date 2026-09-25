import argparse
import asyncio
import json
import os
import traceback
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import fastapi
import uvicorn
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import TemplateError
from logger import logger

from gllm.engine.async_llm import AsyncLLM
from gllm.runtime.sequence import RequestCapacityError
from gllm.entrypoints import cli_args
from gllm.observability.metrics import get_frontend_metrics, set_frontend_model_name
from gllm.entrypoints.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorDetail,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    ResponseRequest,
)
from gllm.entrypoints.serving_chat import (
    chat_completion_generator,
    chat_completion_stream_generator,
)
from gllm.entrypoints.serving_completions import (
    completion_generator,
    completion_stream_generator,
)
from gllm.entrypoints.serving_responses import (
    _previous_output_to_input_items,
    make_chat_request,
    response_completion_generator,
    response_stream_generator,
    snapshot_response_files,
)
from gllm.entrypoints.response_store import ResponseStore
from gllm.tokenizers.tool_parsers import ToolParseError, get_tool_parser
from gllm.tokenizers.reasoning import create_reasoning_parser
from gllm.utils import find_free_ports, make_async

router = APIRouter()

llm: AsyncLLM = None
# Resolved once at startup (see ``run`` / ``__main__``): turns model-native
# tool-call markup into structured ``tool_calls``. ``None`` => model has no
# known tool-call format, raw text passes through as content.
tool_parser = None
response_store = ResponseStore()


def _abort_stream(stream):
    abort = getattr(stream, "abort", None)
    if abort is not None:
        abort()


class RequestStreamingResponse(StreamingResponse):
    """Own engine cancellation even while the consumer is blocked in send()."""

    def __init__(self, content, stream, on_complete=None):
        async def with_errors():
            try:
                async for chunk in content:
                    yield chunk
            except RequestCapacityError as exc:
                error = {"type": "error", "error": {
                    "type": "server_error", "code": "cache_capacity_exceeded",
                    "message": str(exc),
                }}
                yield "event: error\ndata: " + json.dumps(error) + "\n\n"
        super().__init__(content=with_errors(), media_type="text/event-stream")
        self.engine_stream = stream
        self._on_complete = on_complete

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Covers disconnect, send failure, cancellation, and early parser
            # termination. A normally finished stream makes this a no-op.
            _abort_stream(self.engine_stream)
            if self._on_complete is not None:
                try:
                    self._on_complete()
                except Exception:
                    pass


def _openai_error(
    message: str,
    status_code: int = 400,
    *,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
):
    body = ErrorResponse(
        error=ErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _metrics_request_key(request_id: str) -> str:
    return f"{id(llm)}:{request_id}"


def _metrics_begin(request_id: str, method: str, *, streaming: bool,
                   prompt_tokens: int) -> None:
    """Register a request with the frontend metrics registry."""
    m = get_frontend_metrics()
    if m.enabled:
        m.begin_request(
            _metrics_request_key(request_id), method,
            streaming=streaming, prompt_tokens=prompt_tokens,
        )


def _metrics_finish(request_id: str, *, error: bool = False,
                    finish_reason: Optional[str] = None) -> None:
    """Drop a request from the registry; observes E2E/ITL/token histograms."""
    m = get_frontend_metrics()
    if m.enabled:
        m.finish_request(
            _metrics_request_key(request_id), error=error,
            finish_reason=finish_reason,
        )


def _metrics_track_stream(request_id: str, stream):
    """Bind an AsyncStream to its request so stream events update metrics.

    Called from the entrypoint right after ``add_requests_async`` returns.
    Wraps ``AsyncStream.put`` so that each token delta observed by the
    frontend triggers a ``first_token`` / ``token`` observation in
    :class:`FrontendMetrics`.  Idempotent: a stream can be wrapped at most
    once, so wrapping multiple times is safe.
    """
    m = get_frontend_metrics()
    if not m.enabled or stream is None:
        return
    key = _metrics_request_key(request_id)
    if getattr(stream, "_metrics_wrapped", False):
        return
    orig_put = stream.put

    def patched_put(item):
        orig_put(item)
        # StreamOutput carries ``.text``; error / StopAsyncIteration items do not.
        if not hasattr(item, "text") or isinstance(item, BaseException):
            return
        m.first_token(key)
        m.token(key, 1)

    stream.put = patched_put
    stream._metrics_wrapped = True


def _served_model_ids():
    """Return stable aliases for the one checkpoint loaded by this server."""
    model_path = str(getattr(llm, "model_path", ""))
    ids = {model_path}
    parts = Path(model_path).parts
    for part in parts:
        if part.startswith("models--"):
            ids.add(part.removeprefix("models--").replace("--", "/"))
    if model_path:
        ids.add(Path(model_path).name)
    return {model_id for model_id in ids if model_id}


def _validate_model(model: str):
    if model not in _served_model_ids():
        return _openai_error(
            f"The model `{model}` does not exist or is not loaded by this server.",
            404,
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )
    return None


def _public_model_id():
    aliases = [
        model_id
        for model_id in _served_model_ids()
        if not model_id.startswith("/") and "/" in model_id
    ]
    return min(aliases, key=len) if aliases else str(llm.model_path)


def _unsupported(param: str, detail: Optional[str] = None):
    message = detail or f"The parameter `{param}` is not supported by this server."
    return _openai_error(message, param=param, code="unsupported_parameter")


def _validate_output_format(fmt, param, tools=None, ignore_eos=False):
    from gllm.structured_output import normalize_format

    try:
        schema = normalize_format(fmt)
        if (schema is not None or tools) and ignore_eos:
            raise ValueError("Structured output does not support ignore_eos.")
    except (ValueError, TypeError, RecursionError) as exc:
        return _openai_error(str(exc), param=param, code="invalid_output_format")
    return None


async def _prepare_output_format(fmt, token_ids, *, tools=None, custom_formats=None,
                                 parallel_tool_calls=True):
    from gllm.structured_output import prepare_output

    runner = getattr(llm, "model_runner", None)
    return await make_async(prepare_output)(
        fmt, getattr(runner, "tokenizer", None),
        getattr(getattr(runner, "model_loader", None), "vocab_size", 0),
        getattr(llm, "finish_tokens", ()), token_ids,
        tools=tools, parser_name=getattr(tool_parser, "name", None),
        custom_formats=custom_formats, parallel_tool_calls=parallel_tool_calls,
    )


def _validate_chat_capabilities(request: ChatCompletionRequest):
    model_error = _validate_model(request.model)
    if model_error:
        return model_error
    checks = [
        (request.n not in (None, 1), "n"),
        (request.frequency_penalty not in (None, 0, 0.0), "frequency_penalty"),
        (request.presence_penalty not in (None, 0, 0.0), "presence_penalty"),
        (request.logit_bias is not None, "logit_bias"),
        (request.seed is not None, "seed"),
        (bool(request.stop), "stop"),
        (request.store is True, "store"),
        (request.audio is not None, "audio"),
        (bool(request.modalities and "audio" in request.modalities), "modalities"),
        (request.moderation is not None, "moderation"),
        (request.prediction is not None, "prediction"),
        (request.prompt_cache_options is not None, "prompt_cache_options"),
        (request.web_search_options is not None, "web_search_options"),
    ]
    for condition, param in checks:
        if condition:
            return _unsupported(param)
    format_error = _validate_output_format(
        request.response_format, "response_format",
        request.tools if request.tool_choice != "none" else None,
        request.ignore_eos,
    )
    if format_error:
        return format_error
    if request.tools:
        for tool in request.tools:
            if tool.type != "function":
                return _unsupported("tools", "Only function tools are supported.")
    choice = request.tool_choice
    if choice not in (None, "none", "auto"):
        return _unsupported(
            "tool_choice",
            "This runtime supports tool_choice='none' and 'auto'; forced and allowed tool choices are not enforceable by the loaded model.",
        )
    return None


def _validate_response_capabilities(request: ResponseRequest):
    model_error = _validate_model(request.model)
    if model_error:
        return model_error
    checks = [
        (request.background is True, "background"),
        (request.conversation is not None, "conversation"),
        # This stateless backend emits no encrypted reasoning items. Asking
        # for their optional encrypted_content therefore adds no output.
        (bool(set(request.include or []) - {"reasoning.encrypted_content"}), "include"),
        (request.max_tool_calls is not None, "max_tool_calls"),
        (request.moderation is not None, "moderation"),
        (request.prompt is not None, "prompt"),
        (request.prompt_cache_options is not None, "prompt_cache_options"),
        (request.top_logprobs is not None, "top_logprobs"),
        (request.truncation == "auto", "truncation"),
    ]
    for condition, param in checks:
        if condition:
            return _unsupported(param)
    text_format = (request.text or {}).get("format")
    format_error = _validate_output_format(
        text_format, "text.format", request.tools if request.tool_choice != "none" else None
    )
    if format_error:
        return format_error
    if request.tool_choice not in (None, "none", "auto"):
        return _unsupported(
            "tool_choice",
            "This runtime supports tool_choice='none' and 'auto' for Responses.",
        )
    return None


@router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@router.get("/version")
async def version():
    return JSONResponse(content={"version": "0.0.7"})


@router.get("/server_info")
async def server_info():
    return JSONResponse(
        content={
            "model": llm.model_path if llm else "",
            "version": "0.0.7",
            "status": "running",
        }
    )


@router.get("/v1/models")
async def show_available_models():
    model_id = _public_model_id()
    models = ModelList(
        data=[
            ModelCard(
                id=model_id,
                root=model_id,
                max_model_len=llm.model_max_length,
                permission=[ModelPermission()],
            )
        ]
    )
    return JSONResponse(content=models.model_dump())


def _chat_template_kwargs(request: ChatCompletionRequest):
    """Forward reasoning controls while preserving explicit template overrides."""
    kwargs = dict(request.chat_template_kwargs or {})
    if request.reasoning_effort == "none":
        # Some templates reject "none" as an effort and use a separate switch.
        kwargs.setdefault("enable_thinking", False)
        kwargs.setdefault("thinking", False)
    elif request.reasoning_effort is not None:
        # Effort names are model-specific; leave validation to the template.
        kwargs.setdefault("reasoning_effort", request.reasoning_effort)
    return kwargs or None


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    capability_error = _validate_chat_capabilities(request)
    if capability_error:
        return capability_error

    effective_tools = request.tools if request.tool_choice != "none" else None
    chat_template_kwargs = _chat_template_kwargs(request)

    mm_contents = await make_async(llm.model_runner.extract_modify_mm)(request.messages)
    # Encoder-disaggregation frontend (design §3.1 / §5.4): tokenize the *text
    # only* into a skeleton (one sentinel per item) and ship the raw items to
    # the encoder via the LM PP0 worker. The LM never opens pixels and never
    # carries ``mm_contents``. Falls back to the monolith processor path for
    # text requests and when disaggregation is off.
    disagg = getattr(llm, "is_disagg_lm", False)
    mm_items = None
    if disagg and mm_contents is not None:
        mm_items = await make_async(llm.model_runner.extract_mm_items_ordered)(
            request.messages
        )
        token_ids = await make_async(llm.model_runner.encode_skeleton)(
            request.messages, chat_template_kwargs=chat_template_kwargs or None
        )
        mm_contents = None  # LM holds no pixels; embeddings arrive over NIXL
    else:
        token_ids = await make_async(llm.model_runner.encode)(
            request.messages,
            chat=True,
            has_mm=mm_contents is not None,
            chat_template_kwargs=chat_template_kwargs or None,
            # Serialize the pydantic tool schemas to plain dicts; the chat
            # templates (and Kimi's ``encode_tools_to_typescript_style``)
            # expect JSON-like dicts, not pydantic models.
            tools=(
                [
                    t.model_dump(exclude_none=True, by_alias=True)
                    for t in effective_tools
                ]
                if effective_tools
                else None
            ),
        )
    # OpenAI deprecated ``max_tokens`` for chat completions in favor of
    # ``max_completion_tokens`` but most clients (including curl examples,
    # the OpenAI Python SDK pre-1.40, and ``benchmark_serving.py``) still
    # send the legacy field. Honour it as a fallback so the decode cap
    # actually takes effect — otherwise a request without
    # ``max_completion_tokens`` decodes until EOS / model_max_length,
    # which on a broken model produces thousands of garbage tokens.
    # Pydantic intentionally warns whenever the deprecated attribute is read,
    # even when the client did not send it.  Read the validated fallback from
    # the model storage so modern requests do not produce a spurious warning.
    max_output_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.__dict__.get("max_tokens")
    )
    # OpenAI chat logprobs: ``logprobs`` (bool) turns them on, ``top_logprobs``
    # (0-20) is how many alternatives to report per token. Clamp to the OpenAI
    # ceiling to bound the per-step top-k work.
    logprobs_enabled = bool(request.logprobs)
    num_top_logprobs = min(request.top_logprobs or 0, 20) if logprobs_enabled else 0
    prompt_logprobs_enabled = request.prompt_logprobs is not None
    num_prompt_logprobs = (
        min(request.prompt_logprobs, 20) if prompt_logprobs_enabled else 0
    )
    if llm.check_seq_length(token_ids, max_output_tokens):
        try:
            structured_output = await _prepare_output_format(
                request.response_format, token_ids, tools=effective_tools,
                parallel_tool_calls=request.parallel_tool_calls is not False,
            )
            if request.ignore_eos and structured_output is not None and structured_output.schema is None:
                structured_output = None
        except (ValueError, RuntimeError, ImportError) as exc:
            return _openai_error(str(exc), param="response_format", code="invalid_output_format")
        stream = await llm.add_requests_async(
            raw_request,
            token_ids,
            max_output_tokens,
            request.ignore_eos,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty,
            mm_contents,
            mm_items,
            dp_index=getattr(raw_request.app.state, "dp_index", None),
            logprobs_enabled=logprobs_enabled,
            num_top_logprobs=num_top_logprobs,
            prompt_logprobs_enabled=prompt_logprobs_enabled,
            num_prompt_logprobs=num_prompt_logprobs,
            structured_output=structured_output,
        )
        # Prometheus: register this request for lifecycle metrics.
        rid = uuid.uuid4().hex
        _metrics_begin(rid, "chat_completion", streaming=bool(request.stream),
                       prompt_tokens=len(token_ids))
        _metrics_track_stream(rid, stream)
    else:
        return _openai_error(
            "This request exceeds the model's maximum context length.",
            HTTPStatus.BAD_REQUEST.value,
            param="messages",
            code="context_length_exceeded",
        )
    reasoning_parser = create_reasoning_parser(
        getattr(llm.model_runner, "tokenizer", None), token_ids
    )
    if request.stream:
        generator = chat_completion_stream_generator(
            stream, request, tool_parser, reasoning_parser
        )
        return RequestStreamingResponse(
            generator, stream, on_complete=lambda: _metrics_finish(rid)
        )
    else:
        try:
            generator = await chat_completion_generator(
                stream, request, tool_parser, reasoning_parser
            )
        except ToolParseError as exc:
            return _openai_error(str(exc), status_code=500, code="invalid_tool_output")
        finally:
            _abort_stream(stream)
        _metrics_finish(rid)
        return JSONResponse(content=generator.model_dump(exclude_none=True))


@router.post("/v1/responses")
async def create_response(request: ResponseRequest, raw_request: Request):
    capability_error = _validate_response_capabilities(request)
    if capability_error:
        return capability_error
    if request.previous_response_id:
        previous = response_store.get(request.previous_response_id)
        if previous is None:
            return _openai_error(
                "The previous response was not found.", 404,
                param="previous_response_id", code="response_not_found",
            )
        # Both names have to identify the checkpoint currently loaded by this
        # server. A path, basename and HF cache alias are not different models.
        if previous["model"] not in _served_model_ids():
            return _openai_error(
                "The previous response uses a different model.",
                param="previous_response_id", code="invalid_request_error",
            )
        current_items = (
            [{"type": "message", "role": "user", "content": request.input}]
            if isinstance(request.input, str) else list(request.input)
        )
        request = request.model_copy(update={
            "input": (
                list(previous["input_items"])
                + _previous_output_to_input_items(previous["response"])
                + current_items
            ),
        })
    try:
        # File URLs involve blocking I/O; keep them off the FastAPI event loop
        # while building the native text/image message.
        if request.store:
            request = await make_async(snapshot_response_files)(request)
        chat_request = await make_async(make_chat_request)(request)
    except ValueError as exc:
        param, message = exc.args if len(exc.args) == 2 else ("input", str(exc))
        return _unsupported(param, message)

    effective_tools = chat_request.tools if chat_request.tool_choice != "none" else None
    chat_template_kwargs = _chat_template_kwargs(chat_request)
    try:
        mm_contents = await make_async(llm.model_runner.extract_modify_mm)(
            chat_request.messages
        )
        if mm_contents is not None and not llm.model_runner.use_mm:
            return _unsupported(
                "input",
                "The loaded model does not support image inputs.",
            )
        disagg = getattr(llm, "is_disagg_lm", False)
        mm_items = None
        if disagg and mm_contents is not None:
            mm_items = await make_async(llm.model_runner.extract_mm_items_ordered)(
                chat_request.messages
            )
            token_ids = await make_async(llm.model_runner.encode_skeleton)(
                chat_request.messages,
                chat_template_kwargs=chat_template_kwargs or None,
            )
            mm_contents = None
        else:
            token_ids = await make_async(llm.model_runner.encode)(
                chat_request.messages,
                chat=True,
                has_mm=mm_contents is not None,
                chat_template_kwargs=chat_template_kwargs or None,
                tools=(
                    [
                        tool.model_dump(exclude_none=True, by_alias=True)
                        for tool in effective_tools
                    ]
                    if effective_tools
                    else None
                ),
            )
    except (TypeError, ValueError, TemplateError) as exc:
        return _openai_error(str(exc), param="input", code="invalid_input")

    if not llm.check_seq_length(token_ids, request.max_output_tokens):
        return _openai_error(
            "This request exceeds the model's maximum context length.",
            param="input",
            code="context_length_exceeded",
        )
    try:
        from gllm.entrypoints.response_tools import custom_tool_formats

        structured_output = await _prepare_output_format(
            (request.text or {}).get("format"), token_ids, tools=effective_tools,
            custom_formats=custom_tool_formats(request.tools) if effective_tools else None,
            parallel_tool_calls=request.parallel_tool_calls is not False,
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        return _openai_error(str(exc), param="text.format", code="invalid_output_format")
    stream = await llm.add_requests_async(
        raw_request,
        token_ids,
        request.max_output_tokens,
        False,
        request.temperature,
        request.top_p,
        None,
        None,
        mm_contents,
        mm_items,
        dp_index=getattr(raw_request.app.state, "dp_index", None),
        structured_output=structured_output,
    )
    # Prometheus: register this request for lifecycle metrics.
    rid = uuid.uuid4().hex
    _metrics_begin(rid, "response", streaming=bool(request.stream),
                   prompt_tokens=len(token_ids))
    _metrics_track_stream(rid, stream)
    reasoning_parser = create_reasoning_parser(
        getattr(llm.model_runner, "tokenizer", None), token_ids
    )
    if request.stream:
        generator = response_stream_generator(
            stream, request, chat_request, tool_parser, reasoning_parser
        )
        if request.store:
            source_generator = generator
            async def storing_generator():
                async for line in source_generator:
                    data_line = next(
                        (part[6:] for part in line.splitlines() if part.startswith("data: ")),
                        None,
                    )
                    if data_line is not None:
                        try:
                            event = json.loads(data_line)
                        except ValueError:
                            continue
                        if event.get("type") in (
                            "response.completed", "response.incomplete", "response.failed"
                        ):
                            if request.store:
                                response = event["response"]
                                normalized_input = (
                                    [{"type": "message", "role": "user", "content": request.input}]
                                    if isinstance(request.input, str) else list(request.input)
                                )
                                response_store.put(response["id"], {
                                    "response": response,
                                    "input_items": normalized_input,
                                    "model": str(llm.model_path),
                                })
                    yield line
            generator = storing_generator()
        return RequestStreamingResponse(
            generator, stream, on_complete=lambda: _metrics_finish(rid)
        )
    try:
        response = await response_completion_generator(
            stream, request, chat_request, tool_parser, reasoning_parser
        )
    except ValueError as exc:
        return _openai_error(str(exc), status_code=500, code="invalid_tool_output")
    finally:
        _abort_stream(stream)
        _metrics_finish(rid)
    if request.store:
        normalized_input = (
            [{"type": "message", "role": "user", "content": request.input}]
            if isinstance(request.input, str) else list(request.input)
        )
        response_store.put(response["id"], {
            "response": response,
            "input_items": normalized_input,
            "model": str(llm.model_path),
        })
    return JSONResponse(content=response)


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    model_error = _validate_model(request.model)
    if model_error:
        return model_error
    if isinstance(request.prompt, str):
        token_ids = await make_async(llm.model_runner.encode)(request.prompt)
    else:
        # Tokenized prompts must reach the engine unchanged: decoding and
        # re-encoding can merge token boundaries or alter special tokens.
        if not request.prompt:
            return _openai_error("Token prompt must not be empty.", 400, param="prompt")
        # Pydantic has already enforced homogeneous lists and strict integers.
        if not isinstance(request.prompt[0], int):
            return _openai_error(
                "Batched prompts are not supported; provide one string or one token ID array.",
                400, param="prompt",
            )
        token_ids = request.prompt
        vocab_size = llm.model_runner.model_loader.vocab_size
        if min(token_ids) < 0 or max(token_ids) >= vocab_size:
            return _openai_error(
                f"Prompt token IDs must be in [0, {vocab_size}).", 400, param="prompt",
            )
    # OpenAI completions ``logprobs`` is an int: the number of top alternatives
    # to report (the sampled token's logprob is always included). ``None`` /
    # unset disables it. Clamp to the OpenAI ceiling.
    logprobs_enabled = request.logprobs is not None
    num_top_logprobs = min(request.logprobs or 0, 20) if logprobs_enabled else 0
    prompt_logprobs_enabled = request.prompt_logprobs is not None
    num_prompt_logprobs = (
        min(request.prompt_logprobs, 20) if prompt_logprobs_enabled else 0
    )
    if llm.check_seq_length(token_ids, request.max_tokens):
        stream = await llm.add_requests_async(
            raw_request,
            token_ids,
            request.max_tokens,
            request.ignore_eos,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repetition_penalty,
            dp_index=getattr(raw_request.app.state, "dp_index", None),
            logprobs_enabled=logprobs_enabled,
            num_top_logprobs=num_top_logprobs,
            prompt_logprobs_enabled=prompt_logprobs_enabled,
            num_prompt_logprobs=num_prompt_logprobs,
        )
        # Prometheus: register this request for lifecycle metrics.
        rid = uuid.uuid4().hex
        _metrics_begin(rid, "completion", streaming=bool(request.stream),
                       prompt_tokens=len(token_ids))
        _metrics_track_stream(rid, stream)
    else:
        return _openai_error(
            "This request exceeds the model's maximum context length.",
            HTTPStatus.BAD_REQUEST.value,
            param="prompt",
            code="context_length_exceeded",
        )
    if request.stream:
        generator = completion_stream_generator(stream, request)
        return RequestStreamingResponse(
            generator, stream, on_complete=lambda: _metrics_finish(rid)
        )
    else:
        try:
            generator = await completion_generator(stream, request)
        finally:
            _abort_stream(stream)
        _metrics_finish(rid)
        return JSONResponse(content=generator.model_dump())


@router.post("/start_profile")
async def start_profile():
    await llm.start_profile_async()
    return JSONResponse(content={"message": "Profiler started", "success": True})


@router.post("/stop_profile")
async def stop_profile():
    await llm.stop_profile_async()
    return JSONResponse(content={"message": "Profiler stopped", "success": True})


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus text-format scrape endpoint."""
    m = get_frontend_metrics()
    body = m.render()
    return fastapi.Response(content=body, media_type=m.content_type())


def _build_app(dp_index=None):
    """One FastAPI app. ``dp_index`` (via ``app.state``) pins every request that
    arrives on this app to a specific DP replica; ``None`` = round-robin."""
    app = fastapi.FastAPI()

    @app.exception_handler(RequestCapacityError)
    async def cache_capacity_error_handler(_, exc):
        return _openai_error(str(exc), status_code=503, code="cache_capacity_exceeded")

    @app.exception_handler(RequestValidationError)
    async def openai_validation_error_handler(_, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = first.get("loc", ())
        param = ".".join(str(part) for part in location if part != "body") or None
        message = first.get("msg", "Invalid request")
        return _openai_error(
            message,
            HTTPStatus.BAD_REQUEST.value,
            param=param,
            code="invalid_request",
        )

    @app.exception_handler(Exception)
    async def openai_internal_error_handler(_, exc: Exception):
        logger.exception("Unhandled API request error", exc_info=exc)
        return _openai_error(
            "Internal server error",
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
            error_type="server_error",
            code="internal_error",
        )

    app.include_router(router)
    app.state.dp_index = dp_index
    return app


def _endpoint_ports(args):
    """Ports for the per-DP-replica endpoints: explicit ``--endpoint-per-dp-ports``
    (comma-separated, one per replica) or auto-allocated free ports."""
    dp_size = getattr(args, "dp", 1)
    if getattr(args, "endpoint_per_dp_ports", None):
        ports = [int(p) for p in args.endpoint_per_dp_ports.split(",") if p != ""]
        assert (
            len(ports) == dp_size
        ), f"--endpoint-per-dp-ports has {len(ports)} ports but dp_size={dp_size}"
        return ports
    return find_free_ports(dp_size, args.host)


async def run_server(args):
    loop = asyncio.get_running_loop()

    # Per-DP-replica endpoints: one HTTP listener per replica, each pinning its
    # requests to that replica (the single engine still runs the shared schedule
    # loop and routes outputs back by seq_id). Off by default => one endpoint,
    # requests round-robined across replicas.
    if getattr(args, "endpoint_per_dp", False) and getattr(args, "dp", 1) > 1:
        ports = _endpoint_ports(args)
        servers = [
            uvicorn.Server(uvicorn.Config(_build_app(d), port=port, host=args.host))
            for d, port in enumerate(ports)
        ]
        logger.info(
            "DP per-replica endpoints enabled: %s",
            ", ".join(f"dp{d}->:{p}" for d, p in enumerate(ports)),
        )
        tasks = [loop.create_task(s.serve()) for s in servers]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for s in servers:
                await s.shutdown()
        return

    port = args.port if args.port is not None else find_free_ports(1, args.host)[0]
    logger.info("HTTP endpoint on %s:%d", args.host, port)
    server = uvicorn.Server(uvicorn.Config(_build_app(), port=port, host=args.host))
    server_task = loop.create_task(server.serve())
    try:
        await server_task
    except asyncio.CancelledError:
        await server.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the OpenAI-compatible server.

    Every engine-facing flag comes from :mod:`gllm.entrypoints.cli_args`, which
    ``lm_server`` shares; only the front-end / topology flags below belong to
    this entrypoint.
    """
    parser = argparse.ArgumentParser(description="Launch gLLM server")
    cli_args.add_engine_args(parser)
    cli_args.add_frontend_args(parser)
    # Network. Ports default to ``None`` -> a free port is auto-allocated at
    # startup and logged. Pass explicit values for multi-node runs where every
    # node must agree on the same ports.
    parser.add_argument("--host", type=str, help="Host addr", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        help="Uvicorn HTTP port (auto-selects a free port when unset).",
        default=None,
    )
    # Model
    # Runtime
    # Parallelism
    parser.add_argument("--pp", type=int, help="Number of pipeline stages", default=1)
    parser.add_argument(
        "--dp",
        type=int,
        help=(
            "Number of data-parallel (DP-attention) replicas. World size is "
            "pp*dp*tp; with EP enabled the MoE experts are sharded across "
            "EP = dp*tp ranks per pipeline stage."
        ),
        default=1,
    )
    parser.add_argument(
        "--endpoint-per-dp",
        dest="endpoint_per_dp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Expose one HTTP endpoint per DP replica (dp_size > 1). Requests that "
            "arrive on endpoint d are pinned to DP replica d (its KV cache lives "
            "there) instead of being round-robined. A single engine still runs "
            "the shared per-iter schedule/barrier and routes outputs back by "
            "seq_id. Off => one endpoint on --port, round-robin across replicas."
        ),
    )
    parser.add_argument(
        "--endpoint-per-dp-ports",
        dest="endpoint_per_dp_ports",
        type=str,
        default=None,
        help=(
            "Comma-separated ports for the per-replica endpoints (one per DP "
            "replica, in DP-rank order), used with --endpoint-per-dp. Defaults "
            "to auto-allocated free ports."
        ),
    )
    parser.add_argument(
        "--enable-ep",
        dest="enable_ep",
        action="store_true",
        default=False,
        help=(
            "Enable expert parallelism. EP is OFF by default because for many "
            "MoE configs (e.g. Qwen3-30B-A3B with num_experts=128, top_k=8 on "
            "TP=4 / a single node) the EP path leaves each rank with only a "
            "small slice of experts, so the per-expert GEMM is too thin to "
            "saturate the SMs. Pass --enable-ep to opt into expert parallelism."
        ),
    )
    parser.add_argument(
        "--assigned-layers",
        type=str,
        help="If the model have 64 layers, we can set it to 16,16,16,16 or 16,16,17,15",
        default=None,
    )
    # Token Throttling
    # Multi-Node deployment
    parser.add_argument(
        "--launch-mode",
        type=str,
        choices=["normal", "master", "slave"],
        default="normal",
    )
    parser.add_argument(
        "--ranks", type=str, help="Specify the ranks of worker like 0,1", default=None
    )
    # MultiModal
    return parser


def resolve_tool_parser(name=None):
    """Resolve the module-level tool-call parser from the loaded model.

    Explicit ``--tool-call-parser`` wins, else auto-detect from the model
    architecture; ``None`` (unknown model) leaves tool-call markup in ``content``
    unparsed. Both entrypoints call this -- ``lm_server`` serves this same app,
    so a model whose tool calls parse here must parse there too.
    """
    global tool_parser

    architecture = getattr(
        getattr(getattr(llm, "model_runner", None), "model_loader", None),
        "architecture",
        None,
    )
    # DeepSeek-V3.2's tool-call parser uses the checkpoint's reference decoder
    # for exact-typed argument parsing. Load it in this (API server) process from
    # the model dir; None => parser falls back to a lenient regex.
    deepseek_encoder = None
    model_path = getattr(
        getattr(getattr(llm, "model_runner", None), "model_loader", None),
        "model_path",
        None,
    ) or getattr(getattr(llm, "model_runner", None), "model_path", None)
    encoder_variant = {
        "DeepseekV32ForCausalLM": "dsv32",
        "DeepseekV4ForCausalLM": "dsv4",
    }.get(architecture)
    if model_path and encoder_variant is not None:
        from gllm.tokenizers.deepseek_official import load_deepseek_encoder

        deepseek_encoder = load_deepseek_encoder(model_path, encoder_variant)
    tool_parser = get_tool_parser(
        architecture=architecture,
        name=name,
        encoder=deepseek_encoder,
    )
    if name or tool_parser is not None:
        logger.info(
            "Tool-call parser: %s (arch=%s, --tool-call-parser=%s)",
            tool_parser.name if tool_parser else "none",
            architecture,
            name,
        )
    return tool_parser


def main():
    from gllm.runtime.model_loader import quiet_hub_logging

    quiet_hub_logging()
    # ``llm`` is the module-level handle every route reads; this used to be a
    # plain module-scope assignment under ``if __name__ == "__main__"``.
    global llm

    args = build_arg_parser().parse_args()

    llm = AsyncLLM(
        host=args.host,
        launch_mode=args.launch_mode,
        worker_ranks=args.ranks,
        pp_size=args.pp,
        dp_size=args.dp,
        use_ep=args.enable_ep,
        assigned_layers=args.assigned_layers,
        **cli_args.engine_kwargs(args),
    )

    # Bind the model identity so /metrics request counters carry a real
    # ``model=`` label (no-op when metrics are disabled).
    # Prefer the stable HF-style alias (e.g. "Qwen/Qwen3-0.6B") over the raw
    # checkpoint path for a readable Prometheus label.
    _mids = {mid for mid in args.model_path.split(os.sep)
             if mid.startswith("models--")}
    _label = (
        sorted(_mids)[0].removeprefix("models--").replace("--", "/")
        if _mids
        else (os.path.basename(args.model_path) or args.model_path)
    )
    set_frontend_model_name(_label)

    resolve_tool_parser(args.tool_call_parser)

    if args.launch_mode != "slave":
        asyncio.run(run_server(args))
    else:
        try:
            for process in llm.process_list:
                process.join()
        except KeyboardInterrupt as e:
            pass
        except Exception as e:
            logger.error(e)
            traceback.print_exc()


if __name__ == "__main__":
    main()
