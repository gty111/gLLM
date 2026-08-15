import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from gllm.engine.async_llm import AsyncLLM
from gllm.engine.llm import LLM


def test_engine_io_calls_stay_on_one_thread():
    llm = AsyncLLM.__new__(AsyncLLM)
    llm._engine_io_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gllm-engine-io-test",
    )

    async def run_calls():
        first = await llm._run_engine_io(threading.get_ident)
        await asyncio.sleep(0)
        second = await llm._run_engine_io(threading.get_ident)
        return first, second

    try:
        first, second = asyncio.run(run_calls())
    finally:
        llm._engine_io_executor.shutdown(wait=True)

    assert first == second
    assert first != threading.get_ident()


def test_frontend_comm_is_created_on_engine_io_thread():
    llm = AsyncLLM.__new__(AsyncLLM)
    llm._engine_io_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gllm-engine-io-test",
    )
    owner_threads = []

    try:
        with patch.object(
            LLM,
            "_init_frontend_comm",
            autospec=True,
            side_effect=lambda _self: owner_threads.append(threading.get_ident()),
        ):
            llm._init_frontend_comm()
        engine_thread = llm._engine_io_executor.submit(threading.get_ident).result()
    finally:
        llm._engine_io_executor.shutdown(wait=True)

    assert owner_threads == [engine_thread]
    assert engine_thread != threading.get_ident()
