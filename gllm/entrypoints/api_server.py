import uvicorn
import fastapi
import asyncio
import argparse
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gllm.entrypoints.protocol import (ChatCompletionRequest, CompletionRequest, ModelList, ModelCard, ModelPermission, ChatCompletionStreamResponse,
                                       random_uuid, ChatCompletionResponseStreamChoice, DeltaMessage, CompletionStreamResponse, CompletionResponseStreamChoice)
from gllm.async_llm_engine import AsyncLLM, AsyncStream

router = APIRouter()

llm: AsyncLLM = None


async def chat_completion_stream_generator(stream: AsyncStream, request: ChatCompletionRequest):
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())
    chunk_object_type = "chat.completion.chunk"
    async for text in stream:
        choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                         delta=DeltaMessage(
                                                             content=text),
                                                         logprobs=None,
                                                         finish_reason=None)
        chunk = ChatCompletionStreamResponse(id=request_id,
                                             object=chunk_object_type,
                                             created=created_time,
                                             choices=[choice_data],
                                             model=request.model)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f'data: {data}\n\n'
    yield "data: [DONE]\n\n"


async def completion_stream_generator(stream: AsyncStream, request: CompletionRequest):
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())
    async for text in stream:
        choice_data = CompletionResponseStreamChoice(index=0,
                                                     text=text,
                                                     logprobs=None,
                                                     finish_reason=None,
                                                     stop_reason=None)
        chunk = CompletionStreamResponse(id=request_id,
                                         created=created_time,
                                         choices=[choice_data],
                                         model=request.model)
        data = chunk.model_dump_json(exclude_unset=False)
        yield f'data: {data}\n\n'
    yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def show_available_models():
    models = ModelList(
        data=[ModelCard(id=llm.model_path, permission=[ModelPermission()])])
    return JSONResponse(content=models.model_dump())


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    # print(request.messages)
    # print(llm.model_runner.tokenizer.apply_chat_template(request.messages))
    if request.stream:
        token_ids = llm.model_runner.tokenizer.apply_chat_template(
            request.messages, add_generation_prompt=True)
        stream = await llm.add_requests_async(token_ids, request.max_tokens)
        generator = chat_completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type='text/event-stream')
    else:
        return JSONResponse({})


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    if request.stream:
        token_ids = llm.model_runner.tokenizer.encode(request.prompt)
        stream = await llm.add_requests_async(token_ids, request.max_tokens)
        generator = completion_stream_generator(stream, request)
        return StreamingResponse(content=generator, media_type='text/event-stream')
    else:
        return JSONResponse({})


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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    llm = AsyncLLM(args.model_path)

    asyncio.run(run_server(args))
