import uvicorn
import fastapi
import asyncio
import argparse
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from http import HTTPStatus
from typing import Union

from gllm.utils import make_async
from gllm.entrypoints.protocol import ChatCompletionRequest, CompletionRequest, ModelList, ModelCard, ModelPermission, ErrorResponse
from gllm.async_llm_engine import AsyncLLM, PipeAsyncLLM
from gllm.entrypoints.serving_chat import chat_completion_stream_generator, chat_completion_generator
from gllm.entrypoints.serving_completions import completion_stream_generator, completion_generator

router = APIRouter()

llm: Union[AsyncLLM|PipeAsyncLLM] = None


@router.get("/v1/models")
async def show_available_models():
    models = ModelList(
        data=[ModelCard(id=llm.model_path, permission=[ModelPermission()])])
    return JSONResponse(content=models.model_dump())


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    token_ids = await make_async(llm.model_runner.tokenize)(request.messages, chat=True)
    if llm.check_seq_length(token_ids, request.max_tokens):
        stream = await llm.add_requests_async(raw_request, token_ids, request.max_tokens, request.ignore_eos,
                                              request.temperature, request.top_p, request.top_k, request.repetition_penalty)
    else:
        return ErrorResponse(message="seq length exceeds max model length",
                             type="BadRequestError",
                             code=HTTPStatus.BAD_REQUEST.value)
    if request.stream:
        generator = chat_completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type='text/event-stream')
    else:
        generator = await chat_completion_generator(stream, request)
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    token_ids = await make_async(llm.model_runner.tokenize)(request.prompt)
    if llm.check_seq_length(token_ids, request.max_tokens):
        stream = await llm.add_requests_async(raw_request, token_ids, request.max_tokens, request.ignore_eos,
                                              request.temperature, request.top_p, request.top_k, request.repetition_penalty)
    else:
        return ErrorResponse(message="seq length exceeds max model length",
                             type="BadRequestError",
                             code=HTTPStatus.BAD_REQUEST.value)
    if request.stream:
        generator = completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type='text/event-stream')
    else:
        generator = await completion_generator(stream, request)
        return JSONResponse(content=generator.model_dump())


async def run_server(args):
    app = fastapi.FastAPI()
    app.include_router(router)

    server = uvicorn.Server(uvicorn.Config(app,
                                           port=args.port,
                                           host=args.host))

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    try:
        await server_task
    except asyncio.CancelledError:
        await server.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch gLLM server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, help='Uvicorn port', default=8000)
    parser.add_argument('--nccl-port', type=str, help='NCCL port', default='8001')
    parser.add_argument('--model-path', help='Path to the model, either from local disk or from huggingface', type=str, required=True)
    parser.add_argument('--disable-pipe-schedule', help='Use AsyncLLM backend (used for performance comparsion)', action="store_true")
    parser.add_argument('--gpu-memory-util', type=float, help='GPU memory utilization for KV cache (excluding model weights)', default=0.9)
    parser.add_argument('--page-size', type=int, help='Number of tokens in a page', default=16)
    parser.add_argument('--maxd', type=int, help='Maximum decode token count, used in AsyncLLM and offline infernce', default=512)
    parser.add_argument('--maxp', type=int, help='Maximum token count in prefill', default=2048)
    parser.add_argument('--minp', type=int, help='Minimum token count in prefill, used in PipeAsyncLLM', default=32)
    parser.add_argument('--iterp', type=int, help='Number of iterations to process waiting prefill tokens, used in PipeAsyncLLM', default=8)
    parser.add_argument('--kvthresh', type=float, help='KV cache threshold for prefill operations', default=0.05)
    parser.add_argument('--use-naive-schedule', help='Use scheduling policy in Sarathi-Serve', action='store_true')
    parser.add_argument('--enable-prefix-caching', help='Enable KV cache reuse across requests', action='store_true')
    parser.add_argument('--pp', type=int, help='Number of pipeline stages', default=1)
    parser.add_argument('--load-format', type=str, choices=['auto','dummy'], help='auto: actually load model weights; dummy: initialize the model with random values', default='auto')
    parser.add_argument('--assigned-layers', type=str, help='If the model have 64 layers, we can set it to 16,16,16,16 or 16,16,17,15', default=None)
    parser.add_argument('--use-async-worker', help='Experimental feature for worker implemented by async', action='store_true')
    args = parser.parse_args()

    llm_cls = PipeAsyncLLM if not args.disable_pipe_schedule else AsyncLLM
    llm = llm_cls(host=args.host,
                  nccl_port=args.nccl_port,
                  load_format=args.load_format,
                  model_path=args.model_path,
                  gpu_memory_util=args.gpu_memory_util,
                  page_size=args.page_size,
                  maxd=args.maxd,
                  maxp=args.maxp,
                  minp=args.minp,
                  iterp=args.iterp,
                  kvthresh=args.kvthresh,
                  enable_prefix_caching=args.enable_prefix_caching,
                  pp_size=args.pp,
                  assigned_layers=args.assigned_layers,
                  use_naive_schedule=args.use_naive_schedule,
                  use_async_worker=args.use_async_worker)

    asyncio.run(run_server(args))
