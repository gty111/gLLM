import asyncio
from logger import logger
from typing import List, Dict

from gllm.utils import make_async
from gllm.sequence import Sequence
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


def _log_task_completion(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        logger.error("Engine background task failed", exc_info=e)


class AsyncLLM(LLM):

    def __init__(self, model_path, gpu_memory_utilization=0.9, page_size=16, max_decode_seqs=256, max_batch_tokens=8192, ratio_threshold_free_pages=0.2):
        super().__init__(model_path, gpu_memory_utilization, page_size,
                         max_decode_seqs, max_batch_tokens, ratio_threshold_free_pages)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None
        self.process_output_engine = None
        self.gpu_engine = None
        self.schedule_outputs: asyncio.Queue = asyncio.Queue(maxsize=2)
        self.run_outputs: asyncio.Queue = asyncio.Queue()
        

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
        next_tokens = await make_async(self.model_runner.step_once)(seqs=scheduled_seqs, temperature=temperature, top_p=top_p)
        self.scheduler.update_seqs(scheduled_seqs, next_tokens)
        for seq in scheduled_seqs:
            self.async_streams[seq.seq_id].put(seq.detokenize_inc(self.model_runner.tokenizer))
        finished_seqs = []
        for seq in self.scheduler.finish_lists:
            self.async_streams[seq.seq_id].finish()
            del self.async_streams[seq.seq_id]
            finished_seqs.append(seq)
        self.free_requests(finished_seqs)
        
        
    async def run_schedule_engine(self):
        while True:
            scheduled_seqs = self.scheduler.schedule()
            self.schedule_outputs.put(scheduled_seqs)
        
    async def run_gpu_engine(self):
        while True:
            scheduled_seqs = await self.schedule_outputs.get()
            next_tokens = await make_async(self.model_runner.step_once)(seqs=scheduled_seqs, temperature=0.6, top_p=0.9)
            self.run_outputs.put_nowait(next_tokens)

    # async def run_engine(self):
    #     while self.scheduler.has_seqs():
    #         await self.step_async(temperature=0.6, top_p=0.9)
    #     self.background_engine = None

    # def start_background_engine(self):
    #     self._background_loop_unshielded = asyncio.get_event_loop(
    #     ).create_task(self.run_engine())
    #     self._background_loop_unshielded.add_done_callback(
    #         _log_task_completion)
    #     self.background_engine = asyncio.shield(
    #         self._background_loop_unshielded)
