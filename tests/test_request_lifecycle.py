"""Cancelled/failed requests must release frontend and worker ownership."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from gllm.distributed.comm import IPCPackage
from gllm.engine.async_llm import AsyncLLM, AsyncStream
from gllm.engine.llm import LLM
from gllm.entrypoints.api_server import RequestStreamingResponse, _build_app
from gllm.runtime.sequence import GenerationSequence, RequestCapacityError


def frontend():
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.running_maps = {}
    llm.async_streams = {}
    llm.wait_lists = []
    llm.abort_ids = []
    llm._pending_lock = threading.Lock()
    llm.dp_size = 1
    # Monolith fixture: no standalone session filtering on the output path.
    llm.standalone_frontend = False
    llm.frontend_epoch = "test-epoch"
    llm.schedule_engine = object()
    llm.allocate_seq = lambda *args, **kwargs: GenerationSequence(7, [1], [], 8)
    return llm


def test_cancel_while_waiting_for_first_token_enqueues_abort_once():
    async def run():
        llm = frontend()
        stream = await llm.add_requests_async(None, [1], 8, False, 1, 1, 1, 1)
        waiter = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        stream.abort()
        assert llm.abort_ids == [7]
        assert llm.wait_lists[0].is_abort
        assert stream.finished
        stream.put('discarded')
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    asyncio.run(run())


def test_abort_only_package_retires_maps_and_id_exactly_once():
    llm = frontend()
    seq = GenerationSequence(7, [1], [], 8)
    stream = AsyncStream(None, seq=seq)
    llm.running_maps[7] = seq
    llm.async_streams[7] = stream
    freed = []
    llm.free_finish_ids = lambda ids: freed.extend(ids)
    package = IPCPackage([])
    package.free_ids = [7]
    assert LLM._apply_ipc_package(llm, package) == 1
    assert LLM._apply_ipc_package(llm, package) == 0
    assert not llm.running_maps and not llm.async_streams
    assert stream.finished
    assert freed == [7]


def test_capacity_error_reaches_waiting_stream_and_cleans_maps():
    async def run():
        llm = frontend()
        seq = GenerationSequence(7, [1], [], 8)
        stream = AsyncStream(None, seq=seq)
        llm.running_maps[7] = seq
        llm.async_streams[7] = stream
        llm.free_finish_ids = lambda ids: None
        package = IPCPackage([])
        package.free_ids = [7]
        package.request_errors = {7: 'No cache capacity'}
        LLM._apply_ipc_package(llm, package)
        with pytest.raises(RequestCapacityError, match='No cache capacity'):
            await anext(stream)
        assert not llm.running_maps and not llm.async_streams
    asyncio.run(run())


def test_stream_send_cancellation_aborts_even_outside_iterator():
    async def run():
        aborted = []
        stream = AsyncStream(None, on_abort=lambda: aborted.append(True))
        entered = asyncio.Event()
        async def content():
            yield b'data: chunk\n\n'
        async def send(message):
            if message['type'] == 'http.response.body':
                entered.set()
                await asyncio.Event().wait()
        response = RequestStreamingResponse(content(), stream)
        task = asyncio.create_task(response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, None, send))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert aborted == [True]
    asyncio.run(run())


@pytest.mark.parametrize('spec_version', ['2.0', '2.4'])
def test_disconnect_aborts_stream(spec_version):
    async def run():
        aborted = []
        stream = AsyncStream(None, on_abort=lambda: aborted.append(True))
        async def content():
            yield b'data: chunk\n\n'
            await asyncio.Event().wait()
        async def receive():
            return {'type': 'http.disconnect'}
        async def send(message):
            if spec_version == '2.4' and message['type'] == 'http.response.body':
                raise OSError('Disconnected')
        response = RequestStreamingResponse(content(), stream)
        try:
            await response({'type': 'http', 'asgi': {'spec_version': spec_version}}, receive, send)
        except Exception:
            if spec_version != '2.4':
                raise
        assert aborted == [True]
    asyncio.run(run())


def test_normal_completion_does_not_abort():
    async def run():
        aborted = []
        stream = AsyncStream(None, on_abort=lambda: aborted.append(True))
        stream.put('done')
        stream.finish()
        async def content():
            async for chunk in stream:
                yield chunk
        sent = []
        async def send(message): sent.append(message)
        await RequestStreamingResponse(content(), stream)(
            {'type': 'http', 'asgi': {'spec_version': '2.4'}}, None, send)
        assert not aborted
        assert sent[-1]['more_body'] is False
    asyncio.run(run())


def test_stream_capacity_error_is_an_sse_error():
    async def run():
        stream = AsyncStream(None)
        stream.put(RequestCapacityError('No cache capacity'))
        stream.finish()
        async def content():
            async for chunk in stream:
                yield chunk
        sent = []
        async def send(message): sent.append(message)
        await RequestStreamingResponse(content(), stream)(
            {'type': 'http', 'asgi': {'spec_version': '2.4'}}, None, send)
        body = b''.join(m.get('body', b'') for m in sent).decode()
        assert 'event: error\n' in body
        event = json.loads(body.split('data: ', 1)[1])
        assert event['error']['code'] == 'cache_capacity_exceeded'
    asyncio.run(run())


def test_nonstream_capacity_error_returns_503():
    from fastapi.testclient import TestClient
    app = _build_app()
    @app.get('/capacity-test')
    async def fail():
        raise RequestCapacityError('No cache capacity')
    response = TestClient(app).get('/capacity-test')
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'cache_capacity_exceeded'
