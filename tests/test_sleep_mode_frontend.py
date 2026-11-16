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
    llm._sleeping_tags = set()
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


def test_idle_watcher_sleeps_only_when_enabled():
    llm = make_llm()

    async def run(timeout):
        llm.sleep_idle_timeout = timeout
        llm.start_idle_sleep_watcher()
        await asyncio.sleep(0.4)
        if llm._idle_task is not None:
            llm._idle_task.cancel()

    asyncio.run(run(0.0))
    assert llm._idle_task is None and not llm.is_sleeping()
    asyncio.run(run(0.1))
    assert llm.is_sleeping()


def test_invalid_input_is_a_client_error():
    llm = make_llm()

    async def run():
        for kwargs in ({"tags": ["bogus"]}, {"mode": "later"}):
            with pytest.raises(SleepModeError) as err:
                await llm.sleep(**kwargs)
            assert err.value.client_error
        llm.sleep_mode_enabled = False
        with pytest.raises(SleepModeError) as err:
            await llm.wake_up()
        assert err.value.client_error

    asyncio.run(run())
