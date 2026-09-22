import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List

from fastapi import Request
from logger import logger

from gllm.engine.llm import LLM


class AsyncStream:

    def __init__(self, raw_request: Request, seq=None, on_abort=None):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
        self._raw_request = raw_request
        self._on_abort = on_abort
        # The owning GenerationSequence, kept so response builders can report accurate
        # token usage (prompt_len / generated count) and finish_reason once the
        # stream drains. The engine appends generated ids to this same object,
        # so it holds the final counts by the time the stream finishes.
        self.seq = seq

    def put(self, item: str):
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self):
        if self._finished:
            return
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    def abort(self):
        if self._finished:
            return
        if self._on_abort is not None:
            self._on_abort()
        # The consumer has gone away. Do not accumulate its remaining tokens.
        while not self._queue.empty():
            self._queue.get_nowait()
        self.finish()

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._finished and self._queue.empty():
            raise StopAsyncIteration
        try:
            result = await self._queue.get()
        except asyncio.CancelledError:
            self.abort()
            raise
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


class AsyncLLM(LLM):
    """Asynchronous request and stream facade over :class:`LLM`."""

    def __init__(self, *args, **kwargs):
        # The executor must exist before ``LLM.__init__`` reaches
        # ``_init_frontend_comm``: ZeroMQ requires a socket to be created and
        # subsequently used by the same owner thread.
        self._engine_io_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gllm-engine-io",
        )
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._engine_io_executor.shutdown(wait=True)
            raise

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None

    def _init_frontend_comm(self):
        # LLM's synchronous constructor waits for this short task. The same
        # persistent executor later owns every frontend-side send and receive.
        self._engine_io_executor.submit(super()._init_frontend_comm).result()

    async def _run_engine_io(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        call = partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._engine_io_executor, call)

    async def start_profile_async(self):
        await self._run_engine_io(self.start_profile)

    async def stop_profile_async(self):
        await self._run_engine_io(self.stop_profile)

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
        structured_output=None,
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
            structured_output=structured_output,
        )
        # Pin to a specific DP replica when the request came in on a per-replica
        # endpoint (``--endpoint-per-dp``); ``None`` keeps the round-robin default.
        if dp_index is not None and self.dp_size > 1:
            seq.target_dp = dp_index % self.dp_size
        def abort():
            with self._pending_lock:
                if not seq.is_abort:
                    seq.is_abort = True
                    self.abort_ids.append(seq.seq_id)

        stream = AsyncStream(raw_request, seq=seq, on_abort=abort)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        # Enqueue before exposing a cancellable await. This is just a locked
        # list append; executor submission could race cancellation and intake.
        self.add_requests(requests=[seq])
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream

    async def health_async(self):
        # Liveness probe for the (possibly separately deployed) worker fleet.
        # Raises when the fleet is down; the API layer maps that to /health.
        return await self._run_engine_io(self._probe_worker_fleet)

    def _probe_worker_fleet(self):
        if self.standalone_frontend:
            self.check_standalone_worker()
        else:
            self.check_worker_alive()
        return True

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
                stream.abort()

    async def schedule(self):
        while True:
            await self.check_abort_seqs()
            try:
                await self._run_engine_io(super().schedule)
            except Exception as e:
                # Decoupled deployment: the worker fleet died (endpoint file
                # vanished / unreadable, raised by check_standalone_worker
                # *before* any transport use). Fail every in-flight stream
                # fast instead of hanging them; keep the event loop alive so
                # a frontend *process* restart is NOT required -- once the
                # worker fleet republishes its endpoint file, the watcher
                # reconnects in-process and service resumes. Note the uuid
                # change does NOT raise out of the engine IO (the watcher
                # reconnects inside it); that path terminates streams via
                # :meth:`on_standalone_reconnect` instead. A monolith engine
                # never raises here in practice (check_worker_alive does
                # sys.exit), so this is a no-op on the legacy path.
                # Log UNCONDITIONALLY: a persistent non-fleet exception
                # (pickle failure, KeyError in _apply_ipc_package, ...)
                # would otherwise retry silently at 1 Hz forever -- the
                # service looks alive but never processes another token.
                logger.error(
                    "Engine IO tick failed; failing open in-flight streams "
                    "and retrying: %s", e, exc_info=True)
                self._fail_open_streams(e)
                await asyncio.sleep(1.0)
            await asyncio.sleep(0)

    def on_standalone_reconnect(self, reason: Exception):
        """Explicit cleanup at a transport transition (worker fleet restart).

        The restarted fleet has no memory of the sequences this frontend is
        tracking, and the transition does NOT surface as an exception to
        :meth:`schedule` (the watcher reconnects inside the engine IO call
        and returns normally) -- so without this hook the client streams
        would wait forever and their ids would leak.
        """
        self._fail_open_streams(reason)

    def _fail_open_streams(self, exc: Exception):
        """Terminate every outstanding async stream with ``exc`` and drop the
        frontend-side bookkeeping so the frontend can keep serving new
        requests once the worker fleet comes back.

        Mirrors the normal-completion path (``_apply_ipc_package``) for the
        resources it must release: per-stream bookkeeping (``running_maps`` +
        ``async_streams``), the allocator ids (``free_finish_ids``), and the
        pending intake queues (``wait_lists`` / ``abort_ids``). Skipping any
        of these leaks ids or leaves dangling sequences for the next tick.
        """
        if not self.async_streams and not self.wait_lists:
            return
        retired = []
        logger.error(
            "Worker fleet unavailable; failing %d in-flight request(s): %s",
            len(self.async_streams),
            exc,
        )
        for sid, stream in list(self.async_streams.items()):
            self.running_maps.pop(sid, None)
            self.async_streams.pop(sid, None)
            retired.append(sid)
            try:
                if not stream.finished:
                    stream.put(exc)
                    stream.finish()
            except Exception:
                pass
        self.free_finish_ids(retired)
        # Drop any not-yet-dispatched requests too: the dead fleet never saw
        # them, and their streams are in ``async_streams`` (already failed
        # above) while their seq objects would otherwise sit in wait_lists
        # for the next dispatch to a fleet that has no state for them.
        with self._pending_lock:
            self.wait_lists = []
            self.abort_ids = []

    def start_schedule_engine(self):
        # launch schedule engine
        self._schedule_task = asyncio.get_event_loop().create_task(self.schedule())
        self._schedule_task.add_done_callback(_log_task_completion)
        self.schedule_engine = asyncio.shield(self._schedule_task)
