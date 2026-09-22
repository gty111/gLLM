"""Fleet supervisor for a standalone (decoupled) frontend.

A standalone frontend is a stateless, freely restartable process that
serves a SEPARATELY deployed worker fleet (see
docs/frontend_worker_decoupling.md). Its whole relationship to that
fleet is the concern this component owns, so that ``LLM`` keeps only
the thin one-liner touchpoints:

* CONNECT -- wait for the worker endpoint file and build the frontend
  ZMQ sockets for the published transport (:meth:`FleetSupervisor.connect`);
* LIVENESS -- endpoint-file probes for the dispatch path and the
  heartbeat check (:meth:`lively`, :meth:`heartbeat`);
* RECONNECT -- detect a worker-fleet restart (new transport uuid) and
  re-build the transport IN-PROCESS, no frontend process restart. The
  heartbeat path does this via :meth:`_rebuild` (reactive, fleet already
  republished); when the fleet is fully down the schedule loop polls
  :meth:`wait_ready` on a bounded time slice and the NEXT heartbeat
  commits the rebuild once the fleet republishes. (The blocking
  :meth:`reconnect` is deprecated and no longer on this path.)
* DISPATCH GATE -- refuse to ship work into a dead fleet's 512MB send
  buffer, requeueing it for the next tick
  (:meth:`ship`, :meth:`stamp_abort_sessions`).

Threading: every entry point runs on the engine IO executor thread
(the same thread that owns the sockets), so the transport swap inside
:meth:`_rebuild` is race-free with send/recv. The supervisor holds a
reference to its host :class:`~gllm.engine.llm.LLM` and calls back
into ``host.on_standalone_reconnect`` after every successful
transport transition so the host (AsyncLLM) can fail in-flight
streams and release their ids.
"""

import time

from logger import logger

from gllm.distributed.comm import zmqComm
from gllm.entrypoints.worker_endpoint import (
    endpoint_file_age_seconds,
    read_worker_endpoint_file,
)
from gllm.utils import random_uuid

_GRACE_SECONDS = 3  # tolerate atomic rename / one slow heartbeat
_CONNECT_TIMEOUT = 300  # initial wait for the endpoint file to appear
_RECONNECT_TIMEOUT = 600  # standby wait after a fleet restart


class FleetDownError(RuntimeError):
    """Raised when the worker fleet is GONE (endpoint file vanished or went
    stale) -- as opposed to a uuid-CHANGE transition (handled in-process by
    :meth:`FleetSupervisor.heartbeat` -> :meth:`_rebuild`, no raise) or an
    unrelated engine-IO bug.

    A dedicated type (rather than a string-matched RuntimeError) so the
    schedule loop can route fleet-down to the time-sliced standby
    (:meth:`FleetSupervisor.wait_ready`) via ``isinstance`` without coupling
    to the exact message text -- changing a log line can no longer silently
    demote fleet-down to a plain 1 Hz retry.
    """


class FleetSupervisor:
    def __init__(self, llm, dp_size: int):
        # The host LLM: the supervisor reads its comm / paths /
        # bookkeeping and calls back into its on_standalone_reconnect.
        # (The llm.fleet back-pointer is never used here, so there is no
        # recursive access hazard.) The endpoint file is read FROM the
        # host at call time, so tests/harnesses may re-point it freely.
        self.llm = llm
        self.dp_size = dp_size
        # Transport-incarnation + liveness state. NOTE: these must be
        # initialized here (NOT after the property below -- that is
        # unreachable code): heartbeat() may run its first check before
        # connect() ever succeeds, and a missing _gone_since would raise
        # AttributeError and bypass the grace window.
        self._worker_transport_uuid = None
        self._worker_endpoints = None
        self._gone_since = None

    @property
    def endpoint_file(self) -> str:
        return self.llm.worker_endpoint_file

    def _reg(self):
        """The host's endpoint registry, or None for the default file mode.

        A plain ``LLM`` (or a test's ``_bare_llm`` shell) has no
        ``endpoint_registry`` attribute, so this returns None and the
        read helpers below fall back to the file -- preserving the
        historical zero-dependency default and all file-based tests.
        """
        return getattr(self.llm, "endpoint_registry", None)

    def _reg_latest(self):
        """Current ``(uuid, endpoints)`` from the registry, falling back to
        the endpoint file when no registry is configured."""
        reg = self._reg()
        if reg is not None:
            return reg.latest()
        return read_worker_endpoint_file(self.endpoint_file)

    def _reg_age(self) -> float:
        """Seconds since the last registry update, falling back to the file
        mtime. ``None`` means "no endpoint visible"."""
        reg = self._reg()
        if reg is not None:
            return reg.age()
        return endpoint_file_age_seconds(self.endpoint_file)

    # ------------------------------------------------------------------
    # CONNECT
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Wait for the worker endpoint file, then build the transport.

        Called from the host constructor while the fleet is coming up
        (polls until the file appears or ``_CONNECT_TIMEOUT`` elapses).
        """
        from gllm.entrypoints.worker_endpoint import DEFAULT_POLL_INTERVAL

        path = self.endpoint_file
        deadline = time.time() + _CONNECT_TIMEOUT
        while True:
            transport_uuid, endpoints = self._reg_latest()
            if transport_uuid is not None and endpoints:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"No worker endpoint file appeared at {path} within "
                    f"{deadline - time.time() + _CONNECT_TIMEOUT:.0f}s"
                )
            logger.info(
                "Waiting for worker endpoint file %s (standalone frontend)...", path
            )
            time.sleep(DEFAULT_POLL_INTERVAL)
        logger.info(
            "Connected to standalone worker fleet (transport uuid %s, ranks %s) via %s",
            transport_uuid,
            sorted(endpoints),
            path,
        )
        self._worker_transport_uuid = transport_uuid
        self._worker_endpoints = endpoints
        self._build_comm(endpoints)

    def _build_comm(self, endpoints: dict):
        """Create/replace the frontend ZMQ sockets for one transport
        incarnation (called on the engine IO thread)."""
        llm = self.llm
        # Defensive read: on the FIRST standalone connect the host has
        # not run _init_frontend_comm (the ctor skips it), so the attr
        # may not exist yet -- getattr mirrors the original behavior.
        prev = getattr(llm, "comm", None)
        if prev is not None:
            try:
                prev.drain_request_buffer()
            except Exception:
                pass
            try:
                prev.close()
            except Exception:
                pass
            llm.comm = None
        # One transport path per rank; rank 0 carries schedule/output/token.
        # (The rank-0 presence check already ran in _rebuild/connect
        # BEFORE any teardown; this is defense-in-depth for direct
        # callers.)
        if 0 not in endpoints:
            raise ValueError(
                f"Malformed endpoint file: no rank-0 row in {sorted(endpoints)!r}; "
                "the standalone transport connects rank 0 only."
            )
        ep = endpoints[0]
        llm.comm = zmqComm(
            llm.host,
            llm.launch_mode,
            llm.master_addr,
            ep["schedule"],
            ep["output"],
            ep["token"],
            frontend=True,
            dp_size=self.dp_size,
            standalone_remote=True,
        )
        llm.comm.init()
        # Keep the (stale) host paths updated for logging/debugging.
        llm.schedule_path = ep["schedule"]
        llm.output_path = ep["output"]
        llm.token_path = ep["token"]

    # ------------------------------------------------------------------
    # LIVENESS
    # ------------------------------------------------------------------

    def lively(self) -> bool:
        """Cheap fleet liveness probe for the DISPATCH path.

        zmq cannot tell "sent to a dead peer's buffer" from "sent to a
        live one" (SNDBUF=512MB absorbs either), so the endpoint file is
        the source of truth: present and freshly heartbeat-ed. Unlike
        :meth:`heartbeat` this NEVER raises -- a probe error means
        "unknown", and unknown is treated as live so dispatch falls
        through to the bounded non-blocking send.
        """
        try:
            from gllm.entrypoints.worker_endpoint import STALE_AFTER_SECONDS

            age = self._reg_age()
            return age is not None and age <= STALE_AFTER_SECONDS
        except Exception:
            return True

    def heartbeat(self) -> None:
        """Heartbeat/liveness check; raises when the fleet is down.

        Cheap: one file stat + a JSON read when the mtime changed enough.
        Raises RuntimeError when the endpoint file has vanished (worker
        fleet gone) -- the schedule loop converts that into a terminal
        error for every in-flight async stream instead of hanging them.

        Also detects a BACKGROUND fleet restart (same file, new uuid) and
        reconnects in-process. This runs on the engine IO executor
        thread -- the same thread that owns the sockets -- so the swap
        is race-free with send/recv.
        """
        from gllm.entrypoints.worker_endpoint import STALE_AFTER_SECONDS

        path = self.endpoint_file
        age = self._reg_age()
        # The fleet is "gone" when the endpoint file is absent *or* stale
        # (mtime older than STALE_AFTER_SECONDS, i.e. the heartbeat thread
        # is no longer refreshing it -- the SIGKILL / power-loss backstop).
        gone = age is None or age > STALE_AFTER_SECONDS
        if gone:
            # Only declare the fleet dead after a grace period so a brief
            # atomic-rewrite window (rename) or a one-off slow heartbeat
            # cannot false-trip. The grace is much shorter than the old 30s
            # so a crashed worker is recovered quickly.
            self._gone_since = self._gone_since or time.monotonic()
            if time.monotonic() - self._gone_since > _GRACE_SECONDS:
                raise FleetDownError(
                    f"Worker endpoint file {path} is "
                    f"{'missing' if age is None else f'stale (age {age:.1f}s)'}; "
                    f"the worker fleet appears to be down."
                )
            return
        self._gone_since = None
        transport_uuid, endpoints = self._reg_latest()
        if transport_uuid is None or not endpoints:
            # File present but unparseable / empty: treat as gone, with the
            # same short grace as above.
            if self._gone_since is None:
                self._gone_since = time.monotonic()
            if time.monotonic() - self._gone_since > _GRACE_SECONDS:
                raise FleetDownError(
                    f"Worker endpoint file {path} is unreadable; the worker fleet "
                    f"appears to be down."
                )
            return
        if transport_uuid != self._worker_transport_uuid:
            self._rebuild(transport_uuid, endpoints)

    # ------------------------------------------------------------------
    # RECONNECT (standby: fleet down, waiting for a new incarnation)
    # ------------------------------------------------------------------

    def wait_ready(self, max_wait: float) -> bool:
        """Time-sliced fleet liveness probe (the standby primitive).

        Polls the endpoint file for up to ``max_wait`` seconds:

        * returns True as soon as the fleet is READY (endpoint file present
          + parseable). The caller's NEXT heartbeat tick commits the
          :meth:`_rebuild` via the normal uuid-change path (or no-ops when
          the uuid is unchanged); ``wait_ready`` deliberately does NOT
          rebuild, keeping the heavier rebuild in one place.
        * returns False after ``max_wait`` without readiness (fleet still
          down). No exception: the caller just retries on its next tick.

        Bounding the wait is the whole point: ``schedule()`` runs on the
        SINGLE engine-IO thread (max_workers=1), so an unbounded standby
        (the old blocking :meth:`reconnect`, up to ``_RECONNECT_TIMEOUT``)
        would freeze /health (queued behind the standby) and stall process
        shutdown (executor.shutdown(wait=True)). Sliced polls leave the
        thread free between ticks.
        """
        from gllm.entrypoints.worker_endpoint import STALE_AFTER_SECONDS

        path = self.endpoint_file
        deadline = time.time() + max_wait
        while True:
            age = self._reg_age()
            present = age is not None and age <= STALE_AFTER_SECONDS
            if present:
                transport_uuid, endpoints = self._reg_latest()
                if (
                    transport_uuid is not None
                    and endpoints
                    and transport_uuid != self._worker_transport_uuid
                ):
                    # A FRESH incarnation: the next heartbeat will commit the
                    # _rebuild. (Same-uuid means a transient glitch or the
                    # recorded uuid was never set -- keep polling so the loop
                    # does not spin on a stale-but-present file.)
                    return True
            if time.time() >= deadline:
                return False
            time.sleep(0.5)

    def reconnect(self, terminate_reason: Exception = None) -> None:
        """DEPRECATED -- superseded by :meth:`wait_ready` + heartbeat.

        Kept for API compatibility; the standalone frontend no longer
        calls it. The old fully-down path blocked this method for up to
        ``_RECONNECT_TIMEOUT`` (600s) on the SINGLE engine-IO thread, which
        froze /health (queued behind the standby) and stalled process
        shutdown. Recovery is now time-sliced: ``schedule()`` polls
        :meth:`wait_ready` for a bounded ``_STANDBY_SLICE`` each tick, and
        the NEXT heartbeat commits the ``_rebuild`` once the fleet
        republishes. This blocking variant is only safe off the engine-IO
        thread.

        Re-resolves the worker fleet after it restarted (new uuid).

        In-flight zmq messages are lost with the old transport; that is
        the intended blast radius (a restarted worker loses its KV cache
        anyway).

        ``terminate_reason``: when given (the caller caught a
        worker-down error), it is handed to ``on_transition`` AFTER the
        new transport is up, so the host terminates every in-flight
        client stream and releases its ids as an explicit step of the
        transition.
        """
        path = self.endpoint_file
        deadline = time.time() + _RECONNECT_TIMEOUT
        last_err = None
        while True:
            transport_uuid, endpoints = self._reg_latest()
            if transport_uuid is not None and endpoints:
                if transport_uuid == self._worker_transport_uuid:
                    # Same incarnation; transient read glitch.
                    return
                logger.warning(
                    "Worker fleet restarted: transport uuid %s -> %s; reconnecting "
                    "frontend ZMQ transport.",
                    self._worker_transport_uuid,
                    transport_uuid,
                )
                self._rebuild(
                    transport_uuid, endpoints, terminate_reason=terminate_reason
                )
                return
            last_err = "endpoint file absent (worker down?)"
            if time.time() > deadline:
                raise RuntimeError(
                    f"Standby timeout: worker endpoint file {path} not republished "
                    f"within {_RECONNECT_TIMEOUT}s ({last_err}); restarting the worker "
                    f"fleet will recover the frontend without a frontend restart."
                )
            logger.warning(
                "Worker down; polling %s for republish (%s)", path, last_err
            )
            time.sleep(1.0)

    def _rebuild(
        self,
        transport_uuid: str,
        endpoints: dict,
        terminate_reason: Exception = None,
    ) -> None:
        """Swap to a new transport incarnation and run the transition.

        ORDER MATTERS (atomicity): the NEW comm is built and fully
        initialised BEFORE the uuid is committed. If zmqComm.init()
        fails, the old transport is still in place AND the recorded
        uuid still differs from the file, so heartbeat() retries the
        rebuild on the next tick -- a build failure must never wedge
        the frontend (that would defeat the in-process recovery this
        component exists for). Endpoint rows are validated up front:
        a partial/corrupt file is rejected without touching state.
        """
        # All validation BEFORE the old comm is torn down (inside
        # _build_comm): a malformed file must not destroy a working
        # transport, even temporarily.
        for rank, ep in (endpoints or {}).items():
            if not all(k in ep for k in ('schedule', 'output', 'token')):
                raise ValueError(
                    f"Malformed endpoint file: rank {rank} row lacks "
                    f"schedule/output/token keys: {ep!r}"
                )
        if 0 not in (endpoints or {}):
            raise ValueError(
                f"Malformed endpoint file: no rank-0 row in {sorted(endpoints or {})!r}; "
                "the standalone transport connects rank 0 only."
            )
        # Build FIRST, commit AFTER: the host keeps its old comm until
        # the new one is ready (zmqComm.init raised -> old one intact).
        self._build_comm(endpoints)
        self._worker_transport_uuid = transport_uuid
        self._worker_endpoints = endpoints
        # The restarted fleet has no memory of these sequences: drop the
        # frontend-side bookkeeping.
        llm = self.llm
        with llm._pending_lock:
            llm.wait_lists = []
            llm.abort_ids = []
        llm.running_maps.clear()
        # Fresh epoch for the new incarnation: a false-trip fail-open
        # (e.g. a stopped heartbeat) can recycle ids under the SAME
        # epoch, and the fleet's late rows for those recycled ids must
        # never pass the stamp gate of the successor requests.
        llm.frontend_epoch = random_uuid()
        # Explicit stream termination + id release for this transition
        # (LLM base is a no-op; AsyncLLM fails open every stream).
        llm.on_standalone_reconnect(
            terminate_reason
            or RuntimeError("worker fleet restarted; stale request")
        )

    # ------------------------------------------------------------------
    # DISPATCH GATE
    # ------------------------------------------------------------------

    def stamp_abort_sessions(self, ipc_package, abort_ids, wait_lists) -> None:
        """Fill ``ipc_package.abort_sessions`` (IPCPackage protocol) for
        the given CLIENT ids: each abort names the session that owns it,
        so a surviving fleet resolves it to the right (possibly
        other-session) request instead of whatever shares the bare id.

        UNRESOLVABLE aborts (no live seq holds the id anymore) are DROPPED
        rather than stamped with the current epoch: the request already
        finished, and a stale abort carrying our LIVE epoch could be
        matched by the fleet against an INNOCENT new request that happens
        to reuse the id under this epoch. The worker side already ignores
        stamped aborts that match no live seq, so dropping is lossless.
        """
        llm = self.llm
        stamped_ids, stamped_sessions = [], []
        for a in abort_ids:
            live = llm.running_maps.get(a)
            if live is None:
                live = next((s for s in wait_lists if s.seq_id == a), None)
            session = getattr(live, "frontend_session", None)
            if session is None:
                continue  # unresolvable: drop, see above
            stamped_ids.append(a)
            stamped_sessions.append(session)
        ipc_package.abort_ids = stamped_ids
        ipc_package.abort_sessions = stamped_sessions

    def ship(self, ipc_package, wait_lists) -> bool:
        """Send *ipc_package* to the fleet with dead-fleet protection.

        A DEAD fleet (endpoint file gone/stale) would absorb every
        dispatch into the 512MB send buffer and ACK it, so requeue the
        pending work instead of shipping it into the void: the liveness
        check surfaces the outage and the transition hook cleans up.
        The standalone transport has a peer that can legitimately be
        gone; a blocking send would then park the single engine-IO
        thread (taking the liveness check and /health down with it), so
        the send is non-blocking and False is returned when the
        transport refuses.
        """
        if not self.lively():
            self._requeue(ipc_package, wait_lists)
            return False
        if self.llm.comm.send_ipc_package_nonblocking(ipc_package):
            return True
        self._requeue(ipc_package, wait_lists)
        return False

    def _requeue(self, ipc_package, wait_lists) -> None:
        llm = self.llm
        with llm._pending_lock:
            # Undo the bookkeeping and let the next tick retry.
            for seq in wait_lists:
                llm.running_maps.pop(seq.seq_id, None)
            llm.wait_lists = wait_lists + llm.wait_lists
            llm.abort_ids = ipc_package.abort_ids + llm.abort_ids
        logger.warning(
            "Worker transport not ready; requeued %d pending request(s) "
            "for the next tick.",
            len(wait_lists),
        )
