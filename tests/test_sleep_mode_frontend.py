"""Frontend sleep-mode state machine (AsyncLLM), with the workers faked."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gllm.engine.async_llm import AsyncLLM, AsyncStream, SleepModeError
from gllm.runtime.sequence import RequestCapacityError


def make_llm(script=None):
    """An AsyncLLM without workers; ``_memory_command`` follows ``script``.

    ``script`` maps a command name to a list of outcomes consumed in order:
    ``None`` succeeds, an exception instance is raised.
    """
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.sleep_mode_enabled = True
    llm.sleep_idle_timeout = 0.0
    llm.sleep_idle_level = 1
    llm.config = SimpleNamespace(supports_full_sleep=True)
    llm._sleeping_tags = set()
    llm._workers_suspended = False
    llm.process_list = []
    llm.comm = None
    llm._memory_lock = None
    llm._wake_task = None
    llm._idle_task = None
    llm._last_activity = 0.0
    llm._hold_new_requests = False
    llm._pending_lock = threading.Lock()
    llm.wait_lists = []
    llm.abort_ids = []
    llm.running_maps = {}
    llm.async_streams = {}
    llm.freed_ids = []
    llm.id_allocator = SimpleNamespace(free=llm.freed_ids.append)
    llm.check_worker_alive = lambda: None
    llm.sent = []
    script = {k: list(v) for k, v in (script or {}).items()}

    async def memory_command(name, tags):
        llm.sent.append((name, tuple(tags)))
        await asyncio.sleep(0.01)
        outcome = script.get(name, []).pop(0) if script.get(name) else None
        if outcome is not None:
            raise outcome
        return [
            {
                "cmd": name,
                "ok": True,
                "bytes": {t: 1 << 30 for t in tags},
                "seconds": 0.1,
            }
        ]

    llm._memory_command = memory_command
    return llm


def test_sleep_then_wake_holds_and_releases_requests():
    llm = make_llm()

    async def run():
        r = await llm.sleep()
        assert r["sleeping_tags"] == ["kv_cache", "weights"]
        assert r["gib"] == {"kv_cache": 1.0, "weights": 1.0}
        assert llm._hold_new_requests and llm.is_sleeping()
        r = await llm.wake_up(["weights"])
        assert r["sleeping_tags"] == ["kv_cache"] and llm._hold_new_requests
        await llm.wake_up()
        assert not llm._hold_new_requests and not llm.is_sleeping()

    asyncio.run(run())


def test_failed_sleep_is_rolled_back():
    llm = make_llm({"sleep": [SleepModeError("sleep failed: OOM")]})

    async def run():
        with pytest.raises(SleepModeError):
            await llm.sleep()

    asyncio.run(run())
    assert llm.sent == [
        ("sleep", ("kv_cache", "weights")),
        ("wake_up", ("kv_cache", "weights")),
    ]
    assert not llm.is_sleeping() and not llm._hold_new_requests


def test_failed_rollback_keeps_requests_held():
    llm = make_llm({"sleep": [SleepModeError("x")], "wake_up": [SleepModeError("y")]})

    async def run():
        with pytest.raises(SleepModeError):
            await llm.sleep(["kv_cache"])

    asyncio.run(run())
    assert llm.sleeping_tags == ["kv_cache"] and llm._hold_new_requests


def test_cancelled_caller_does_not_abandon_transition():
    llm = make_llm()

    async def run():
        task = asyncio.ensure_future(llm.sleep())
        await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert llm.sleeping_tags == ["kv_cache", "weights"] and llm._hold_new_requests


def test_failed_auto_wake_answers_held_requests_with_503():
    llm = make_llm({"wake_up": [SleepModeError("wake_up failed: CUDA OOM")]})

    async def run():
        await llm.sleep()
        seq = SimpleNamespace(seq_id=7)
        stream = AsyncStream(None, seq=seq)
        llm.async_streams[7] = stream
        llm.wait_lists.append(seq)
        llm._ensure_wake_task()
        await llm._wake_task
        with pytest.raises(RequestCapacityError) as err:
            async for _ in stream:
                pass
        return err.value

    err = asyncio.run(run())
    assert err.code == "wake_up_failed" and "CUDA OOM" in str(err)
    assert llm.wait_lists == [] and llm.async_streams == {} and llm.freed_ids == [7]
    # Still asleep: the next request retries the wake.
    assert llm.is_sleeping() and llm._hold_new_requests


@pytest.mark.parametrize("level", [1, 2])
def test_idle_watcher_sleeps_only_when_enabled(level):
    llm = make_llm()
    llm.sleep_idle_level = level

    async def engine_io(func):
        return func()

    llm._run_engine_io = engine_io
    llm._stop_local_workers = lambda: None

    async def run(timeout):
        llm.sleep_idle_timeout = timeout
        llm.start_idle_sleep_watcher()
        await asyncio.sleep(0.4)
        if llm._idle_task is not None:
            llm._idle_task.cancel()

    asyncio.run(run(0.0))
    assert llm._idle_task is None and not llm.is_sleeping()
    asyncio.run(run(0.1))
    assert llm.sleep_level == level


def test_invalid_input_is_a_client_error():
    llm = make_llm()

    async def run():
        for kwargs in ({"tags": ["bogus"]}, {"mode": "later"}, {"level": 3}):
            with pytest.raises(SleepModeError) as err:
                await llm.sleep(**kwargs)
            assert err.value.client_error
        llm.config.supports_full_sleep = False
        with pytest.raises(SleepModeError, match="local, non-disaggregated") as err:
            await llm.sleep(level=2)
        assert err.value.client_error
        llm.sleep_mode_enabled = False
        with pytest.raises(SleepModeError) as err:
            await llm.wake_up()
        assert err.value.client_error

    asyncio.run(run())


def test_level_two_stops_workers_after_drain_and_can_resume():
    llm = make_llm()
    events = []

    async def engine_io(func):
        return func()

    async def drain(mode):
        assert llm._hold_new_requests
        assert not llm._workers_suspended
        events.append(mode)

    def restart():
        assert llm._hold_new_requests and llm._workers_suspended
        events.append("restart")
        llm.mp_alive = [1]
        llm.process_list = [SimpleNamespace(exitcode=None)]

    llm._run_engine_io = engine_io
    llm._wait_until_idle = drain
    llm._stop_local_workers = lambda: events.append("stop")
    llm._restart_local_workers = restart

    async def run():
        await llm.sleep(["weights"])
        assert (await llm.sleep(mode="abort", level=2))["level"] == 2
        assert llm.sleeping_tags == ["kv_cache", "weights"]
        assert (await llm.sleep())["level"] == 2
        with pytest.raises(SleepModeError, match="all tags"):
            await llm.wake_up(["weights"])
        assert llm.is_sleeping()

        llm._ensure_wake_task()
        await llm._wake_task
        assert llm.sleep_level == 0 and not llm._hold_new_requests

    asyncio.run(run())
    assert events == ["wait", "abort", "stop", "restart"]
    assert llm.sent == [("sleep", ("weights",))]


def test_level_two_failed_start_cleans_up_and_allows_retry():
    llm = make_llm()
    killed = []

    async def engine_io(func):
        return func()

    def failed_start():
        llm.mp_alive = [-1]
        llm.process_list = [SimpleNamespace(exitcode=1)]

    llm._run_engine_io = engine_io
    llm._stop_local_workers = lambda: killed.append(True)
    llm._restart_local_workers = failed_start

    async def run():
        await llm.sleep(level=2)
        with pytest.raises(SleepModeError, match="exited during initialization"):
            await llm.wake_up()
        assert llm.sleep_level == 2 and llm._hold_new_requests
        assert killed == [True, True]
        llm._restart_local_workers = lambda: setattr(llm, "mp_alive", [1])
        llm.process_list = [SimpleNamespace(exitcode=None)]
        assert (await llm.wake_up())["level"] == 0
        assert not llm._hold_new_requests

    asyncio.run(run())
