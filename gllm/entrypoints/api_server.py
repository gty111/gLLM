import argparse
import asyncio
import traceback
from http import HTTPStatus

import fastapi
import uvicorn
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from logger import logger

from gllm.async_llm_engine import PipeAsyncLLM
from gllm.entrypoints.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from gllm.entrypoints.serving_chat import (
    chat_completion_generator,
    chat_completion_stream_generator,
)
from gllm.entrypoints.serving_completions import (
    completion_generator,
    completion_stream_generator,
)
from gllm.entrypoints.tool_parsers import get_tool_parser
from gllm.utils import find_free_ports, make_async

router = APIRouter()

llm: PipeAsyncLLM = None
# Resolved once at startup (see ``run`` / ``__main__``): turns model-native
# tool-call markup into structured ``tool_calls``. ``None`` => model has no
# known tool-call format, raw text passes through as content.
tool_parser = None


@router.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@router.get("/version")
async def version():
    return JSONResponse(content={"version": "0.0.6.post1"})


@router.get("/server_info")
async def server_info():
    return JSONResponse(content={
        "model": llm.model_path if llm else "",
        "version": "0.0.6.post1",
        "status": "running",
    })


@router.get("/v1/models")
async def show_available_models():
    models = ModelList(
        data=[ModelCard(id=llm.model_path, root=llm.model_path,
                        max_model_len=llm.model_max_length,
                        permission=[ModelPermission()])]
    )
    return JSONResponse(content=models.model_dump())


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
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
            request.messages, chat_template_kwargs=request.chat_template_kwargs
        )
        mm_contents = None  # LM holds no pixels; embeddings arrive over NIXL
    else:
        token_ids = await make_async(llm.model_runner.encode)(
            request.messages,
            chat=True,
            has_mm=mm_contents is not None,
            chat_template_kwargs=request.chat_template_kwargs,
            # Serialize the pydantic tool schemas to plain dicts; the chat
            # templates (and Kimi's ``encode_tools_to_typescript_style``)
            # expect JSON-like dicts, not pydantic models.
            tools=(
                [t.model_dump(exclude_none=True) for t in request.tools]
                if request.tools
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
    max_output_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
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
        )
    else:
        return ErrorResponse(
            message="seq length exceeds max model length",
            type="BadRequestError",
            code=HTTPStatus.BAD_REQUEST.value,
        )
    if request.stream:
        generator = chat_completion_stream_generator(stream, request, tool_parser)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        generator = await chat_completion_generator(stream, request, tool_parser)
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    token_ids = await make_async(llm.model_runner.encode)(request.prompt)
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
        )
    else:
        return ErrorResponse(
            message="seq length exceeds max model length",
            type="BadRequestError",
            code=HTTPStatus.BAD_REQUEST.value,
        )
    if request.stream:
        generator = completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        generator = await completion_generator(stream, request)
        return JSONResponse(content=generator.model_dump())


@router.post("/start_profile")
async def start_profile():
    await make_async(llm.start_profile)()
    return JSONResponse(content={"message": "Profiler started", "success": True})


@router.post("/stop_profile")
async def stop_profile():
    await make_async(llm.stop_profile)()
    return JSONResponse(content={"message": "Profiler stopped", "success": True})


def _build_app(dp_index=None):
    """One FastAPI app. ``dp_index`` (via ``app.state``) pins every request that
    arrives on this app to a specific DP replica; ``None`` = round-robin."""
    app = fastapi.FastAPI()
    app.include_router(router)
    app.state.dp_index = dp_index
    return app


def _endpoint_ports(args):
    """Ports for the per-DP-replica endpoints: explicit ``--endpoint-per-dp-ports``
    (comma-separated, one per replica) or auto-allocated free ports."""
    dp_size = getattr(args, "dp", 1)
    if getattr(args, "endpoint_per_dp_ports", None):
        ports = [int(p) for p in args.endpoint_per_dp_ports.split(",") if p != ""]
        assert len(ports) == dp_size, (
            f"--endpoint-per-dp-ports has {len(ports)} ports but dp_size={dp_size}"
        )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch gLLM server")
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
    parser.add_argument("--master-addr", type=str, help="NCCL addr", default="0.0.0.0")
    parser.add_argument(
        "--master-port",
        type=str,
        help="NCCL rendezvous port (auto-selects a free port when unset).",
        default=None,
    )
    # Model
    parser.add_argument(
        "--model-path",
        help="Path to the model, either from local disk or from huggingface",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--load-format",
        type=str,
        choices=["auto", "dummy"],
        help="auto: actually load model weights; dummy: initialize the model with random values",
        default="auto",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        help="Maximum sequence length supported by the model (including prompt and generated tokens)",
        default=None,
    )
    # Runtime
    parser.add_argument(
        "--overlap-scheduling",
        dest="overlap_scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CPU/GPU overlap scheduling with FutureMap (default: on; requires pp=1)",
    )
    parser.add_argument(
        "--gpu-memory-util",
        type=float,
        help="GPU memory utilization for KV cache (excluding model weights)",
        default=0.9,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV cache reuse across requests (default: on)",
    )
    parser.add_argument(
        "--page-size", type=int, help="Number of tokens in a page", default=16
    )
    parser.add_argument(
        "--mla-decode-backend",
        type=str,
        choices=["fa3", "flashmla", "triton"],
        default="fa3",
        help=(
            "MLA decode attention backend. 'fa3' (default) uses FA3 absorbed "
            "MLA decode via sgl_kernel (SGLang-compatible); 'flashmla' uses "
            "DeepSeek FlashMLA (auto-bumps page_size to 64); 'triton' uses "
            "the in-tree Triton kernel. Unavailable backends fall back "
            "automatically."
        ),
    )
    parser.add_argument(
        "--disable-cuda-graph",
        help="Enable full cuda graph for decode batch",
        action="store_true",
    )
    parser.add_argument(
        "--max-cuda-graph-bs",
        type=int,
        help=(
            "Maximum batch size for CUDA graph capture. "
            "Larger values allow more decode batches to benefit from CUDA graphs "
            "but increase startup time and GPU memory usage during graph capture. "
            "Default: 512."
        ),
        default=512,
    )
    # Parallelism
    parser.add_argument("--pp", type=int, help="Number of pipeline stages", default=1)
    parser.add_argument(
        "--tp", type=int, help="Number of tensor parallel degrees", default=1
    )
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
    parser.add_argument(
        "--maxd",
        type=int,
        help="Maximum decode token count per batch (Token Throttling)",
        default=512,
    )
    parser.add_argument(
        "--maxp",
        type=int,
        help="Maximum prefill token count per batch (Token Throttling) or token budget in Sarathi-Serve",
        default=8192,
    )
    parser.add_argument(
        "--minp",
        type=int,
        help="Minimum prefill token count per batch (Token Throttling)",
        default=32,
    )
    parser.add_argument(
        "--iterp",
        type=int,
        help="Number of iterations to process waiting prefill tokens (Token Throttling)",
        default=8,
    )
    parser.add_argument(
        "--init-new-token-ratio",
        type=float,
        help="Initial/ceiling fraction of remaining output length reserved for "
        "in-flight decodes (adaptive KV admission control)",
        default=0.7,
    )
    parser.add_argument(
        "--min-new-token-ratio",
        type=float,
        help="Floor the new-token-ratio decays toward when the system is stable "
        "(adaptive KV admission control)",
        default=0.1,
    )
    parser.add_argument(
        "--schedule-method",
        type=str,
        choices=["split_pd", "chunked_prefill", "token_throttling"],
        help="Specify scheduling method",
        default="chunked_prefill",
    )
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
    parser.add_argument(
        "--mm-processor-min-pixels",
        type=int,
        help="Minimum pixels for multimodal processor",
        default=None,
    )
    parser.add_argument(
        "--mm-processor-max-pixels",
        type=int,
        help="Maximum pixels for multimodal processor",
        default=None,
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=["kimi", "qwen"],
        help="Parser for model-native tool-call output -> structured "
        "tool_calls. Default: auto-detect from model architecture; pass a "
        "name to override.",
    )
    args = parser.parse_args()

    llm = PipeAsyncLLM(
        host=args.host,
        master_addr=args.master_addr,
        master_port=args.master_port,
        launch_mode=args.launch_mode,
        worker_ranks=args.ranks,
        load_format=args.load_format,
        model_path=args.model_path,
        gpu_memory_util=args.gpu_memory_util,
        page_size=args.page_size,
        maxd=args.maxd,
        maxp=args.maxp,
        minp=args.minp,
        iterp=args.iterp,
        init_new_token_ratio=args.init_new_token_ratio,
        min_new_token_ratio=args.min_new_token_ratio,
        enable_prefix_caching=args.enable_prefix_caching,
        pp_size=args.pp,
        tp_size=args.tp,
        dp_size=args.dp,
        use_ep=args.enable_ep,
        assigned_layers=args.assigned_layers,
        schedule_method=args.schedule_method,
        overlap_scheduling=args.overlap_scheduling,
        disable_cuda_graph=args.disable_cuda_graph,
        max_cuda_graph_bs=args.max_cuda_graph_bs,
        model_max_length=args.model_max_length,
        mm_processor_min_pixels=args.mm_processor_min_pixels,
        mm_processor_max_pixels=args.mm_processor_max_pixels,
        mla_decode_backend=args.mla_decode_backend,
    )

    # Resolve the tool-call parser once: explicit ``--tool-call-parser`` wins,
    # else auto-detect from the model architecture. ``None`` (unknown model)
    # leaves tool-call markup in ``content`` unparsed.
    architecture = getattr(
        getattr(getattr(llm, "model_runner", None), "model_loader", None),
        "architecture",
        None,
    )
    tool_parser = get_tool_parser(
        architecture=architecture, name=args.tool_call_parser
    )
    if args.tool_call_parser or tool_parser is not None:
        logger.info(
            "Tool-call parser: %s (arch=%s, --tool-call-parser=%s)",
            tool_parser.name if tool_parser else "none",
            architecture,
            args.tool_call_parser,
        )

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
