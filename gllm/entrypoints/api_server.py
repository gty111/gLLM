import uvicorn
import fastapi
import asyncio
import argparse
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from http import HTTPStatus

from gllm.utils import make_async
from gllm.entrypoints.protocol import ChatCompletionRequest, CompletionRequest, ModelList, ModelCard, ModelPermission, ErrorResponse
from gllm.async_llm_engine import AsyncLLM, PipeAsyncLLM
from gllm.entrypoints.serving_chat import chat_completion_stream_generator, chat_completion_generator
from gllm.entrypoints.serving_completions import completion_stream_generator, completion_generator

router = APIRouter()

llm: AsyncLLM = None


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
                                              request.temperature, request.top_p, request.top_k)
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
                                              request.temperature, request.top_p, request.top_k)
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
                                           port=args.port))

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    try:
        await server_task
    except asyncio.CancelledError:
        await server.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch GLLM server')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--disable-pipe-schedule', action="store_true")
    parser.add_argument('--gpu-memory-util', type=float, default=0.9)
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--max-decode-seqs', type=int, default=512)
    parser.add_argument('--max-batch-tokens', type=int, default=8192)
    parser.add_argument('--ratio-free-pages', type=float, default=0.2)
    parser.add_argument('--enable-prefix-caching', action='store_true')
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--load-format', type=str, choices=['auto','dummy'],default='auto')
    args = parser.parse_args()

    llm_cls = PipeAsyncLLM if not args.disable_pipe_schedule else AsyncLLM
    llm = llm_cls(load_format=args.load_format,
                  model_path=args.model_path,
                  gpu_memory_util=args.gpu_memory_util,
                  page_size=args.page_size,
                  max_decode_seqs=args.max_decode_seqs,
                  max_batch_tokens=args.max_batch_tokens,
                  ratio_threshold_free_pages=args.ratio_free_pages,
                  enable_prefix_caching=args.enable_prefix_caching,
                  pp_size=args.pp)

    asyncio.run(run_server(args))
