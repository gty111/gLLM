import argparse
import asyncio
import traceback
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import fastapi
import uvicorn
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from logger import logger

from gllm.engine.async_llm import AsyncLLM
from gllm.entrypoints import cli_args
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
    make_chat_request,
    response_completion_generator,
    response_stream_generator,
)
from gllm.tokenizers.tool_parsers import get_tool_parser
from gllm.utils import find_free_ports, make_async

router = APIRouter()

llm: AsyncLLM = None
# Resolved once at startup (see ``run`` / ``__main__``): turns model-native
# tool-call markup into structured ``tool_calls``. ``None`` => model has no
# known tool-call format, raw text passes through as content.
tool_parser = None


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
    if request.response_format is not None and request.response_format.type != "text":
        return _unsupported(
            "response_format",
            "Structured output response formats are not wired to this runtime yet.",
        )
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
        (bool(request.include), "include"),
        (request.max_tool_calls is not None, "max_tool_calls"),
        (request.moderation is not None, "moderation"),
        (request.previous_response_id is not None, "previous_response_id"),
        (request.prompt is not None, "prompt"),
        (request.prompt_cache_options is not None, "prompt_cache_options"),
        (request.store is True, "store"),
        (request.top_logprobs is not None, "top_logprobs"),
        (request.truncation == "auto", "truncation"),
    ]
    for condition, param in checks:
        if condition:
            return _unsupported(param)
    text_format = (request.text or {}).get("format") or {"type": "text"}
    if text_format.get("type", "text") != "text":
        return _unsupported(
            "text.format", "Structured Responses output is not wired yet."
        )
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


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    capability_error = _validate_chat_capabilities(request)
    if capability_error:
        return capability_error

    effective_tools = request.tools if request.tool_choice != "none" else None
    chat_template_kwargs = dict(request.chat_template_kwargs or {})
    if request.reasoning_effort == "none":
        # Qwen and other reasoning templates commonly expose one or both names.
        chat_template_kwargs.setdefault("enable_thinking", False)
        chat_template_kwargs.setdefault("thinking", False)

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
        )
    else:
        return _openai_error(
            "This request exceeds the model's maximum context length.",
            HTTPStatus.BAD_REQUEST.value,
            param="messages",
            code="context_length_exceeded",
        )
    if request.stream:
        generator = chat_completion_stream_generator(stream, request, tool_parser)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        generator = await chat_completion_generator(stream, request, tool_parser)
        return JSONResponse(content=generator.model_dump(exclude_none=True))


@router.post("/v1/responses")
async def create_response(request: ResponseRequest, raw_request: Request):
    capability_error = _validate_response_capabilities(request)
    if capability_error:
        return capability_error
    try:
        # File URLs involve blocking I/O; keep them off the FastAPI event loop
        # while building the native text/image message.
        chat_request = await make_async(make_chat_request)(request)
    except ValueError as exc:
        param, message = exc.args if len(exc.args) == 2 else ("input", str(exc))
        return _unsupported(param, message)

    effective_tools = chat_request.tools if chat_request.tool_choice != "none" else None
    chat_template_kwargs = {}
    if chat_request.reasoning_effort == "none":
        chat_template_kwargs.update(enable_thinking=False, thinking=False)
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
    except (TypeError, ValueError) as exc:
        return _openai_error(str(exc), param="input", code="invalid_input")

    if not llm.check_seq_length(token_ids, request.max_output_tokens):
        return _openai_error(
            "This request exceeds the model's maximum context length.",
            param="input",
            code="context_length_exceeded",
        )
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
    )
    if request.stream:
        generator = response_stream_generator(
            stream, request, chat_request, tool_parser
        )
        return StreamingResponse(content=generator, media_type="text/event-stream")
    response = await response_completion_generator(
        stream, request, chat_request, tool_parser
    )
    return JSONResponse(content=response)


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    model_error = _validate_model(request.model)
    if model_error:
        return model_error
    token_ids = await make_async(llm.model_runner.encode)(request.prompt)
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
    else:
        return _openai_error(
            "This request exceeds the model's maximum context length.",
            HTTPStatus.BAD_REQUEST.value,
            param="prompt",
            code="context_length_exceeded",
        )
    if request.stream:
        generator = completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        generator = await completion_generator(stream, request)
        return JSONResponse(content=generator.model_dump())


@router.post("/start_profile")
async def start_profile():
    await llm.start_profile_async()
    return JSONResponse(content={"message": "Profiler started", "success": True})


@router.post("/stop_profile")
async def stop_profile():
    await llm.stop_profile_async()
    return JSONResponse(content={"message": "Profiler stopped", "success": True})


def _build_app(dp_index=None):
    """One FastAPI app. ``dp_index`` (via ``app.state``) pins every request that
    arrives on this app to a specific DP replica; ``None`` = round-robin."""
    app = fastapi.FastAPI()

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
