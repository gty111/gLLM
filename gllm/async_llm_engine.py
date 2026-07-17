import asyncio
from typing import Dict, List

from fastapi import Request
from logger import logger

from gllm.llm_engine import LLM
from gllm.utils import make_async


class AsyncStream:

    def __init__(self, raw_request: Request, seq=None):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
        self._raw_request = raw_request
        # The owning Sequence, kept so response builders can report accurate
        # token usage (prompt_len / generated count) and finish_reason once the
        # stream drains. The engine appends generated ids to this same object,
        # so it holds the final counts by the time the stream finishes.
        self.seq = seq

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

    async def is_disconnected(self):
        return await self._raw_request.is_disconnected()


def _log_task_completion(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        logger.error("Engine background task failed", exc_info=e)


class PipeAsyncLLM(LLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None

    async def add_requests_async(
        self,
        raw_request: Request,
        token_ids: List[int],
        output_len: int,
        ignore_eos: bool,
        temperature: float,
        top_p: float,
        top_k: float,
        repetition_penalty: float,
        mm_contents=None,
        mm_items=None,
        dp_index=None,
        logprobs_enabled=False,
        num_top_logprobs=0,
        prompt_logprobs_enabled=False,
        num_prompt_logprobs=0,
    ):
        seq = self.allocate_seq(
            token_ids,
            output_len,
            ignore_eos,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            mm_contents,
            mm_items,
            logprobs_enabled=logprobs_enabled,
            num_top_logprobs=num_top_logprobs,
            prompt_logprobs_enabled=prompt_logprobs_enabled,
            num_prompt_logprobs=num_prompt_logprobs,
        )
        # Pin to a specific DP replica when the request came in on a per-replica
        # endpoint (``--endpoint-per-dp``); ``None`` keeps the round-robin default.
        if dp_index is not None and self.dp_size > 1:
            seq.target_dp = dp_index % self.dp_size
        stream = AsyncStream(raw_request, seq=seq)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream

    async def check_abort_seqs(self):
        # Snapshot: the engine step (``send_ipc_package`` / ``_apply_ipc_package``
        # on an executor thread) mutates ``running_maps`` concurrently, so
        # iterating it live could raise "dictionary changed size during
        # iteration". ``async_streams`` may also drop an id between the snapshot
        # and the lookup, so guard the read.
        for id, seq in list(self.running_maps.items()):
            stream = self.async_streams.get(id)
            if stream is None:
                continue
            if await stream.is_disconnected() and not seq.is_abort:
                with self._pending_lock:
                    self.abort_ids.append(id)
                seq.is_abort = True

    async def schedule(self):
        while True:
            await self.check_abort_seqs()
            await make_async(super().schedule)()
            await asyncio.sleep(0)

    def start_schedule_engine(self):
        # launch schedule engine
        self._schedule_task = asyncio.get_event_loop().create_task(self.schedule())
        self._schedule_task.add_done_callback(_log_task_completion)
        self.schedule_engine = asyncio.shield(self._schedule_task)
