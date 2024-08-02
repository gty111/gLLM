import asyncio
import logger
from functools import partial
from typing import Callable, List, Dict

from gllm.utils import make_async
from gllm.llm_engine import LLM


class AsyncStream:

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: str):
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self):
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class AsyncEngineDeadError(RuntimeError):
    pass


def _log_task_completion(task: asyncio.Task,
                         error_callback: Callable[[Exception], None]) -> None:
    """This function is only intended for the `engine.run_engine_loop()` task.

    In particular, that task runs a `while True` loop that can only exit if
    there is an exception.
    """

    exception = None
    try:
        return_value = task.result()
        # raise AssertionError(
        #     f"The engine background task should never finish without an "
        #     f"exception. {return_value}")
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            "Task finished unexpectedly. This should never happen! "
            "Please open an issue on Github. See stack trace above for the"
            "actual cause.") from e


class AsyncLLM(LLM):

    def __init__(self, model_path, gpu_memory_utilization=0.9, page_size=16, max_decode_seqs=256, max_batch_tokens=8192, ratio_threshold_free_pages=0.2):
        super().__init__(model_path, gpu_memory_utilization, page_size,
                         max_decode_seqs, max_batch_tokens, ratio_threshold_free_pages)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.background_engine = None

    async def add_requests_async(self, token_ids: List[int], output_len: int):
        seq = self.allocate_seq(token_ids, output_len)
        stream = AsyncStream()
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.background_engine is None:
            self.start_background_engine()
        return stream

    async def step_async(self, temperature, top_p):
        scheduled_seqs = self.scheduler.schedule()
        await make_async(self.model_runner.step_once)(seqs=scheduled_seqs, temperature=temperature, top_p=top_p)
        for seq in scheduled_seqs:
            delta_text = self.model_runner.tokenizer.decode(seq.token_ids[-1])
            self.async_streams[seq.seq_id].put(delta_text)
        self.scheduler.update_finish_seqs()
        finished_seqs = []
        for seq in self.scheduler.finish_lists.values():
            self.async_streams[seq.seq_id].finish()
            del self.async_streams[seq.seq_id]
            finished_seqs.append(seq)
        self.free_requests(finished_seqs)

    async def run_engine(self):
        while self.scheduler.has_seqs():
            await self.step_async(temperature=0.6, top_p=0.9)
        self.background_engine = None

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    def start_background_engine(self):
        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine())
        self._background_loop_unshielded.add_done_callback(
            partial(_log_task_completion, error_callback=self._error_callback))
        self.background_engine = asyncio.shield(
            self._background_loop_unshielded)
