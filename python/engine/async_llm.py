import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Iterable, List, Optional

from fastapi import Request
from logger import logger

from gllm.engine.llm import LLM
from gllm.runtime.sequence import RequestCapacityError
from gllm.runtime.sleep_mode import ALL_TAGS, normalize_tags


class SleepModeError(RuntimeError):
    """A sleep/wake request that cannot be served.

    ``client_error`` marks a bad request (sleep mode disabled, invalid tags or
    mode) as opposed to a worker-side failure or timeout.
    """

    def __init__(self, message: str, client_error: bool = False):
        super().__init__(message)
        self.client_error = client_error


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

        # Sleep mode state. Every transition runs under ``_memory_lock`` (made
        # lazily: it must belong to the server's event loop).
        self.sleep_mode_enabled = bool(self.config.sleep_mode)
        self.sleep_idle_timeout = float(self.config.sleep_idle_timeout)
        self.sleep_idle_level = self.config.sleep_idle_level
        self._sleeping_tags: set = set()
        self._workers_suspended = False
        self._memory_lock: Optional[asyncio.Lock] = None
        self._wake_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._last_activity = time.monotonic()
        # Upper bound for one sleep/wake round trip (draining requests in
        # ``wait`` mode is not included).
        self.memory_cmd_timeout = 600.0

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
        self._last_activity = time.monotonic()
        if self._hold_new_requests:
            # Asleep (or falling asleep): the request waits in ``wait_lists``
            # until the memory is back.
            self._ensure_wake_task()
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream

    # ------------------------------------------------------------------
    # Sleep mode
    # ------------------------------------------------------------------
    @property
    def sleeping_tags(self) -> List[str]:
        return sorted(self._sleeping_tags)

    def is_sleeping(self) -> bool:
        return bool(self._sleeping_tags)

    @property
    def sleep_level(self) -> int:
        if self._workers_suspended:
            return 2
        return 1 if self._sleeping_tags else 0

    def _lock(self) -> asyncio.Lock:
        if self._memory_lock is None:
            self._memory_lock = asyncio.Lock()
        return self._memory_lock

    def _require_sleep_mode(self):
        if not self.sleep_mode_enabled:
            raise SleepModeError(
                "sleep mode is disabled (server started with --no-sleep-mode)",
                client_error=True,
            )

    async def _wait_until_idle(self, mode: str) -> None:
        """Wait until no request is in flight on any worker.

        New requests are already held back. ``abort`` cancels the in-flight
        ones; ``wait`` lets them finish.
        """
        if mode == "abort":
            for stream in list(self.async_streams.values()):
                if not stream.finished:
                    stream.abort()
        while self.running_maps:
            self.check_worker_alive()
            await asyncio.sleep(0.01)

    async def _memory_command(self, name: str, tags) -> List[dict]:
        """Send sleep/wake to every replica and wait for all their acks."""
        if self.schedule_engine is None:
            # Acks arrive through the schedule loop's output polling.
            self.start_schedule_engine()
        await self._run_engine_io(self.send_memory_command, name, tags)
        acks: List[dict] = []
        deadline = time.monotonic() + self.memory_cmd_timeout
        while len(acks) < self.dp_size:
            self.check_worker_alive()
            if time.monotonic() > deadline:
                raise SleepModeError(f"timed out waiting for workers to {name}")
            await asyncio.sleep(0.005)
            acks.extend(a for a in self.take_control_acks() if a.get("cmd") == name)
        errors = [a["error"] for a in acks if not a.get("ok")]
        if errors:
            raise SleepModeError(f"{name} failed: {'; '.join(sorted(set(errors)))}")
        return acks

    @staticmethod
    def _summarize(acks: List[dict], tags) -> dict:
        nbytes = {t: sum(a.get("bytes", {}).get(t, 0) for a in acks) for t in tags}
        return {
            "tags": list(tags),
            "gib": {t: round(b / (1 << 30), 3) for t, b in nbytes.items()},
            "seconds": round(max((a.get("seconds", 0.0) for a in acks), default=0.0), 3),
        }

    async def sleep(
        self, tags: Optional[Iterable[str]] = None, mode: str = "wait", *, level: int = 1
    ) -> dict:
        """Release the GPU memory of ``tags`` (default: weights and KV cache).

        New requests are held from the moment this is called; with
        ``mode="wait"`` in-flight requests finish first, ``"abort"`` cancels
        them. A request that arrives meanwhile wakes the engine afterwards.
        """
        self._require_sleep_mode()
        if mode not in ("wait", "abort"):
            raise SleepModeError(
                f"invalid mode {mode!r}; expected 'wait' or 'abort'", client_error=True
            )
        try:
            tags = normalize_tags(tags)
        except ValueError as e:
            raise SleepModeError(str(e), client_error=True) from e
        if level not in (1, 2):
            raise SleepModeError(
                f"invalid sleep level {level}; expected 1 or 2", client_error=True
            )
        if level == 2:
            if not self.config.supports_full_sleep:
                raise SleepModeError(
                    "sleep level 2 requires a local, non-disaggregated server",
                    client_error=True,
                )
            if set(tags) != set(ALL_TAGS):
                raise SleepModeError("sleep level 2 requires all tags", client_error=True)
        # Shielded: a caller that goes away must not abandon a transition
        # halfway and leave the workers and this frontend disagreeing.
        return await asyncio.shield(self._sleep(tags, mode, level))

    async def _sleep(self, tags, mode: str, level: int) -> dict:
        async with self._lock():
            if self._workers_suspended:
                if self.process_list or self.comm is not None:
                    return await self._sleep_all_workers(mode)
                return {
                    "tags": [], "gib": {}, "seconds": 0.0,
                    "sleeping_tags": self.sleeping_tags, "level": 2,
                }
            if level == 2:
                return await self._sleep_all_workers(mode)
            todo = [t for t in tags if t not in self._sleeping_tags]
            result = {"tags": [], "gib": {}, "seconds": 0.0}
            if todo:
                self._hold_new_requests = True
                try:
                    await self._wait_until_idle(mode)
                except BaseException:
                    self._hold_new_requests = bool(self._sleeping_tags)
                    raise
                try:
                    acks = await self._memory_command("sleep", todo)
                except Exception:
                    await self._rollback_sleep(todo)
                    raise
                self._sleeping_tags.update(todo)
                result = self._summarize(acks, todo)
                logger.info(
                    "Engine asleep (%s): released %s in %.2fs",
                    ",".join(self.sleeping_tags),
                    ", ".join(f"{t}={g:.2f} GiB" for t, g in result["gib"].items()),
                    result["seconds"],
                )
            if self.wait_lists:
                # Requests that arrived while falling asleep.
                self._ensure_wake_task()
        result["sleeping_tags"] = self.sleeping_tags
        result["level"] = self.sleep_level
        return result

    def _stop_local_workers(self) -> None:
        # Run on the engine-I/O thread, which owns the frontend ZMQ sockets.
        # The worker processes own the CUDA contexts. Exiting the whole group
        # releases graph pools, NCCL, and any pinned level-1 weight backups.
        started = [p for p in self.process_list if p.pid is not None]
        for process in started:
            if process.is_alive():
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        for process in started:
            process.join(timeout=max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in started):
            raise RuntimeError("timed out waiting for local workers to exit")
        for process in self.process_list:
            process.close()
        self.process_list.clear()
        if self.comm is not None:
            self.comm.ctx.destroy(linger=0)
            self.comm = None

    def _restart_local_workers(self) -> None:
        # A failed transition may have left workers alive. Never start a new
        # group while an old one still owns CUDA memory or a rendezvous port.
        self._stop_local_workers()
        self._reset_worker_state()
        super()._init_frontend_comm()
        self._launch_workers(show_progress=False)

    async def _sleep_all_workers(self, mode: str) -> dict:
        if not self._workers_suspended:
            self._hold_new_requests = True
            try:
                # Wait out a schedule tick that may have already removed a batch
                # from wait_lists but hasn't yet registered it in running_maps.
                await self._run_engine_io(lambda: None)
                await self._wait_until_idle(mode)
            except BaseException:
                self._hold_new_requests = bool(self._sleeping_tags)
                raise
        start = time.monotonic()
        self._workers_suspended = True
        self._sleeping_tags.update(ALL_TAGS)
        try:
            await self._run_engine_io(self._stop_local_workers)
        except Exception as exc:
            raise SleepModeError(f"sleep level 2 failed: {exc}") from exc
        finally:
            if self.wait_lists:
                self._ensure_wake_task()
        elapsed = round(time.monotonic() - start, 3)
        logger.info("Engine asleep (level 2): local workers exited in %.2fs", elapsed)
        return {
            "tags": self.sleeping_tags, "gib": {}, "seconds": elapsed,
            "sleeping_tags": self.sleeping_tags, "level": 2,
        }

    async def _wake_all_workers(self) -> dict:
        start = time.monotonic()
        try:
            await self._run_engine_io(self._restart_local_workers)
            while True:
                states = list(self.mp_alive)
                if -1 in states or any(p.exitcode is not None for p in self.process_list):
                    raise RuntimeError("a worker exited during initialization")
                if all(state == 1 for state in states):
                    break
                if time.monotonic() - start > self.memory_cmd_timeout:
                    raise RuntimeError("timed out waiting for local workers to initialize")
                await asyncio.sleep(0.05)
        except Exception as exc:
            try:
                await self._run_engine_io(self._stop_local_workers)
            except Exception:
                logger.exception("Failed to clean up workers after wake-up failure")
            raise SleepModeError(f"wake-up from level 2 failed: {exc}") from exc

        tags = self.sleeping_tags
        self._sleeping_tags.clear()
        self._workers_suspended = False
        elapsed = round(time.monotonic() - start, 3)
        logger.info("Engine woke from level 2: local workers initialized in %.2fs", elapsed)
        return {"tags": tags, "gib": {}, "seconds": elapsed}

    async def wake_up(self, tags: Optional[Iterable[str]] = None) -> dict:
        """Restore ``tags`` (default: everything asleep).

        Requests resume once no tag is asleep any more.
        """
        self._require_sleep_mode()
        try:
            tags = normalize_tags(tags)
        except ValueError as e:
            raise SleepModeError(str(e), client_error=True) from e
        return await asyncio.shield(self._wake_up(tags))

    async def _rollback_sleep(self, todo) -> None:
        """After a failed sleep, wake ``todo`` on every rank (idempotent).

        Some ranks may have released their memory. If even that fails the
        tags are treated as asleep, so requests stay held and the next one
        retries the wake instead of running on released memory.
        """
        try:
            await self._memory_command("wake_up", todo)
        except Exception:
            logger.exception("Rolling back a failed sleep failed; treating %s as asleep", todo)
            self._sleeping_tags.update(todo)
        self._hold_new_requests = bool(self._sleeping_tags)
        if self._hold_new_requests and self.wait_lists:
            self._ensure_wake_task()

    async def _wake_up(self, tags) -> dict:
        async with self._lock():
            todo = [t for t in tags if t in self._sleeping_tags]
            result = {"tags": [], "gib": {}, "seconds": 0.0}
            if self._workers_suspended:
                if set(tags) != set(ALL_TAGS):
                    raise SleepModeError(
                        "waking from sleep level 2 requires all tags", client_error=True
                    )
                result = await self._wake_all_workers()
            elif todo:
                acks = await self._memory_command("wake_up", todo)
                self._sleeping_tags.difference_update(todo)
                result = self._summarize(acks, todo)
                logger.info(
                    "Engine woke %s in %.2fs",
                    ",".join(todo),
                    result["seconds"],
                )
            if not self._sleeping_tags:
                self._hold_new_requests = False
                self._last_activity = time.monotonic()
        result["sleeping_tags"] = self.sleeping_tags
        result["level"] = self.sleep_level
        return result

    def _ensure_wake_task(self) -> None:
        if self._wake_task is not None and not self._wake_task.done():
            return

        async def wake_all():
            try:
                # Lets an in-progress sleep finish first (same lock).
                await self.wake_up(None)
            except Exception as e:
                logger.exception("Automatic wake-up failed")
                self._fail_held_requests(f"server is asleep and failed to wake up: {e}")

        self._wake_task = asyncio.get_running_loop().create_task(wake_all())

    def _fail_held_requests(self, message: str) -> None:
        """Answer every held request with HTTP 503; the next one retries."""
        with self._pending_lock:
            held, self.wait_lists = self.wait_lists, []
            held_ids = {seq.seq_id for seq in held}
            self.abort_ids = [seq_id for seq_id in self.abort_ids if seq_id not in held_ids]
        for seq in held:
            stream = self.async_streams.pop(seq.seq_id, None)
            if stream is not None:
                stream.put(RequestCapacityError(message, code="wake_up_failed"))
                stream.finish()
        self.free_finish_ids([seq.seq_id for seq in held])

    def start_idle_sleep_watcher(self) -> None:
        """Sleep after ``sleep_idle_timeout`` seconds with nothing in flight."""
        if self.sleep_idle_timeout <= 0 or self._idle_task is not None:
            return
        self._idle_task = asyncio.get_running_loop().create_task(self._idle_sleep_loop())
        self._idle_task.add_done_callback(_log_task_completion)
        logger.info("Idle sleep enabled: sleep after %.1fs without requests", self.sleep_idle_timeout)

    async def _idle_sleep_loop(self) -> None:
        timeout = self.sleep_idle_timeout
        poll = min(1.0, max(timeout / 4, 0.05))
        while True:
            await asyncio.sleep(poll)
            now = time.monotonic()
            if self._sleeping_tags or self._lock().locked():
                self._last_activity = now
                continue
            if self.running_maps or self.wait_lists:
                self._last_activity = now
                continue
            if now - self._last_activity < timeout:
                continue
            logger.info("No requests for %.1fs; going to sleep", now - self._last_activity)
            try:
                await self.sleep(None, mode="wait", level=self.sleep_idle_level)
            except Exception:
                logger.exception("Idle sleep failed")
            self._last_activity = time.monotonic()

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
            if self._workers_suspended:
                await asyncio.sleep(0.01)
                continue
            await self.check_abort_seqs()
            if self._workers_suspended:
                continue
            await self._run_engine_io(super().schedule)
            await asyncio.sleep(0)

    def start_schedule_engine(self):
        self._schedule_task = asyncio.get_event_loop().create_task(self.schedule())
        self._schedule_task.add_done_callback(_log_task_completion)
        self.schedule_engine = asyncio.shield(self._schedule_task)
