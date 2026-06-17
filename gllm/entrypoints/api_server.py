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
from gllm.utils import make_async

router = APIRouter()

llm: PipeAsyncLLM = None


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
        )
    else:
        return ErrorResponse(
            message="seq length exceeds max model length",
            type="BadRequestError",
            code=HTTPStatus.BAD_REQUEST.value,
        )
    if request.stream:
        generator = chat_completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        generator = await chat_completion_generator(stream, request)
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


async def run_server(args):
    app = fastapi.FastAPI()
    app.include_router(router)

    server = uvicorn.Server(uvicorn.Config(app, port=args.port, host=args.host))

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    try:
        await server_task
    except asyncio.CancelledError:
        await server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch gLLM server")
    # Network
    parser.add_argument("--host", type=str, help="Host addr", default="0.0.0.0")
    parser.add_argument("--port", type=int, help="Uvicorn port", default=8000)
    parser.add_argument("--master-addr", type=str, help="NCCL addr", default="0.0.0.0")
    parser.add_argument("--master-port", type=str, help="NCCL port", default="8001")
    parser.add_argument("--zmq-port-base", type=int, help="ZeroMQ port", default=8002)
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
        choices=["flashmla", "triton"],
        default="flashmla",
        help=(
            "MLA decode attention backend. 'flashmla' (default) uses the "
            "DeepSeek FlashMLA kernel (auto-bumps page_size to 64; falls back "
            "to Triton if unavailable); 'triton' uses the in-tree Triton "
            "kernel. Only affects MLA models (e.g. DeepSeek)."
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
    args = parser.parse_args()

    llm = PipeAsyncLLM(
        host=args.host,
        master_addr=args.master_addr,
        master_port=args.master_port,
        zmq_port_base=args.zmq_port_base,
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
