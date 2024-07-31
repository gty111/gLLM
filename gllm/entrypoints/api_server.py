import uvicorn
import fastapi
import asyncio
import argparse
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/generate")
async def create_chat_completion(request: Request):
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    print(prompt,stream)
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
    args = parser.parse_args()

    asyncio.run(run_server(args))
