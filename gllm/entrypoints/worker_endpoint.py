"""Frontend/worker decoupling: standalone worker endpoint discovery.

When a gLLM worker fleet is deployed as independent processes (so that a
frontend crash/restart never takes down the GPU workers, and a worker crash
never drags down the frontend), the two sides rendezvous through a small
JSON *endpoint file*::

    {
        "uuid": "<transport id, random per worker launch>",
        "updated_at": 1789974106.123,
        "endpoints": {
            "<local_rank>": {
                "schedule": "ipc:///tmp/<uuid>_gllm_schedule" or "tcp://host:port",
                "output":   "ipc:///tmp/<uuid>_gllm_output"  or "tcp://host:port",
                "token":    "ipc:///tmp/<uuid>_gllm_token"   or "tcp://host:port"
            },
            ...
        }
    }

The worker process writes it (after binding its frontend-facing sockets);
the frontend polls it. On a worker restart the transport uuid changes, which
forces the frontend to tear down and re-connect its ZMQ sockets -- the
frontend process itself never has to restart. Conversely the frontend may
restart freely: it simply re-reads the endpoint file of the still-running
worker fleet.
"""

import atexit
import threading
import errno
import json
import os
import tempfile
import time
import uuid as uuid_mod
from pathlib import Path
from typing import Dict, Optional, Tuple

from logger import logger

ENDPOINT_FILE_VERSION = 1

# Poll cadence for the frontend's endpoint watcher. Tight enough that a
# worker restart is picked up within ~1-2 s, cheap otherwise (one stat per
# poll).
DEFAULT_POLL_INTERVAL = 0.5

# The worker refreshes the endpoint file every ``DEFAULT_HEARTBEAT_INTERVAL``
# seconds. A file older than ``STALE_AFTER_SECONDS`` is considered dead even
# though it still exists -- this is the backstop for the SIGKILL / power-loss
# case where atexit never runs and the file is left behind. 5x the heartbeat
# (10s) is loose enough to survive a slow heartbeat under GPU load but tight
# enough to recover far faster than the prior 30s grace.
DEFAULT_HEARTBEAT_INTERVAL = 2.0
STALE_AFTER_SECONDS = 10.0


def new_transport_uuid() -> str:
    return uuid_mod.uuid4().hex


def _atomic_write_json(path: str, obj: dict) -> None:
    path = str(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".worker_endpoint_", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class WorkerEndpointWriter:
    """Worker side: publish/refresh the endpoint file, keep it fresh.

    Liveness: the file's ``updated_at`` is refreshed every
    ``heartbeat_interval`` seconds by the driver loop (or manually via
    :meth:`heartbeat`). A frontend may optionally treat a stale file as
    "worker gone"; the load-bearing signal for restart detection is the
    transport uuid change, which happens atomically with the rewrite.
    """

    def __init__(self, path: str, heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL):
        self.path = str(path)
        self.heartbeat_interval = max(0.2, float(heartbeat_interval))
        self.transport_uuid = new_transport_uuid()
        self._endpoints: Dict[int, dict] = {}
        self._last_heartbeat = 0.0
        self._closed = False
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        atexit.register(self.cleanup)

    def set_endpoints(self, endpoints: Dict[int, dict]) -> str:
        """Publish a (potentially partial) rank -> endpoint map.

        Returns the transport uuid of this worker launch.
        """
        with self._lock:
            self._endpoints.update(endpoints)
            self._write_locked()
        self._start_heartbeat_thread()
        logger.info(
            "Published worker endpoint file %s (transport uuid %s, ranks %s)",
            self.path,
            self.transport_uuid,
            sorted(self._endpoints),
        )
        return self.transport_uuid

    def heartbeat(self) -> bool:
        """Refresh ``updated_at`` if it is due. Returns True when rewritten."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_heartbeat >= self.heartbeat_interval:
                self._write_locked()
                return True
        return False

    def _heartbeat_loop(self) -> None:
        """Background heartbeats: keep the file fresh even when the calling
        loop is blocked (e.g. the standalone worker parent sleeps between
        child liveness checks)."""
        while not self._closed:
            time.sleep(min(0.1, self.heartbeat_interval))
            self.heartbeat()

    def _start_heartbeat_thread(self) -> None:
        if self._heartbeat_thread is not None:
            return
        t = threading.Thread(target=self._heartbeat_loop, name="gllm-worker-endpoint-heartbeat", daemon=True)
        t.start()
        self._heartbeat_thread = t

    def _write_locked(self) -> None:
        """Caller must hold ``self._lock``."""
        if self._closed:
            return
        self._last_heartbeat = time.monotonic()
        _atomic_write_json(
            self.path,
            {
                "version": ENDPOINT_FILE_VERSION,
                "uuid": self.transport_uuid,
                "updated_at": time.time(),
                "endpoints": self._endpoints,
            },
        )

    def cleanup(self) -> None:
        """Remove the endpoint file so no frontend can dial a dead worker.

        Best-effort: the OS kills the process before the writer could run
        (SIGKILL / power loss), in which case the frontend's liveness timeout
        is the backstop.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        try:
            # ABA guard: only unlink a file that still carries OUR transport
            # uuid. Between teardown and this call a SUCCESSOR fleet may have
            # already published the same path, and removing that fresh file
            # would blind every frontend to a live worker.
            try:
                with open(self.path, "r") as f:
                    on_disk = json.load(f)
            except (OSError, json.JSONDecodeError):
                on_disk = None
            if on_disk is None:
                return  # gone (or unreadable) -- nothing to do
            if on_disk.get("uuid") != self.transport_uuid:
                logger.warning(
                    "Not removing worker endpoint file %s: it now belongs to "
                    "transport %s (this writer owned %s).",
                    self.path, on_disk.get("uuid"), self.transport_uuid,
                )
                return
            os.unlink(self.path)
            logger.info("Removed worker endpoint file %s", self.path)
        except OSError as e:
            logger.warning("Could not remove worker endpoint file %s: %s", self.path, e)


def read_worker_endpoint_file(
    path: str,
) -> Tuple[Optional[str], Optional[Dict[int, dict]]]:
    """Read the endpoint file; ``None`` payload when absent / unreadable.

    Returns ``(transport_uuid, endpoints)``.
    """
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        endpoints = obj.get("endpoints")
        if endpoints is not None:
            # JSON object keys are strings; the engine indexes by int rank.
            endpoints = {int(k): v for k, v in endpoints.items()}
        return obj.get("uuid"), endpoints
    except FileNotFoundError:
        return None, None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Worker endpoint file %s unreadable: %s", path, e)
        return None, None


def endpoint_file_age_seconds(path: str) -> Optional[float]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return max(0.0, time.time() - st.st_mtime)


# ============================================================================
# EndpointRegistry abstraction
# ============================================================================
#
# The endpoint "file" above is ONE rendezvous implementation: a neutral,
# on-disk address book the worker writes and the frontend polls. To support
# a separately deployed, network-backed registry (an independent proxy
# middleware, modeled on the PD-disaggregation discovery proxy) without
# touching the engine, the worker/frontend talk to a small interface instead
# of the file directly. The FILE stays the zero-dependency default (backward
# compatible); a PROXY implementation routes the same calls to a standalone
# in-memory registry process (gllm.entrypoints.discovery_server, reusing the
# dependency-free ZMQ DiscoveryServer/NetworkDiscovery).
#
# The middleware is CONTROL-PLANE ONLY: it stores (uuid -> transport rows +
# lease). The data plane stays frontend<->worker POINT-TO-POINT zmq, using the
# addresses the registry hands out -- the registry is never in the request
# path.
#
#   register/renew/revoke  -> worker side (publish + lease heartbeat)
#   latest/age             -> frontend side (discover + staleness backstop)


class EndpointRegistry:
    """Interface between the standalone worker/frontend and their registry.

    Two sides:

    * WORKER: :meth:`register` / :meth:`renew` / :meth:`revoke` (+
      :meth:`uuid`). A background renewal keeps the lease fresh (file
      impl: rewrite; proxy impl: lease renew that auto re-registers on a
      registry restart).
    * FRONTEND: :meth:`latest` (the current ``(uuid, endpoints)`` or
      ``(None, None)``) and :meth:`age` (seconds since the last registry
      update, or ``None`` when there is none) for the staleness backstop.
    """

    # --- worker side -------------------------------------------------
    def register(self, endpoints: Dict[int, dict]) -> str:
        """Publish this worker launch's endpoint rows; returns its uuid."""
        raise NotImplementedError

    def renew(self) -> None:
        """Refresh the lease / freshness timestamp."""
        raise NotImplementedError

    def revoke(self) -> None:
        """Withdraw this worker (idempotent; the file impl ABA-guards)."""
        raise NotImplementedError

    @property
    def uuid(self) -> str:
        raise NotImplementedError

    # --- frontend side ----------------------------------------------
    def latest(self):
        """Return ``(transport_uuid, endpoints)`` or ``(None, None)``."""
        raise NotImplementedError

    def age(self) -> Optional[float]:
        """Seconds since the latest registry update, or ``None``."""
        raise NotImplementedError


class FileEndpointRegistry(EndpointRegistry):
    """File-backed registry. One instance plays ONE side:

    * ``side="worker"``: owns a ``WorkerEndpointWriter`` (writes the file,
      auto-heartbeats, ABA-safe ``revoke`` == ``cleanup``).
    * ``side="frontend"``: read-only ``read_worker_endpoint_file`` /
      ``endpoint_file_age_seconds``.

    This is the existing default behavior, now routed through the interface.
    """

    def __init__(self, path: str, side: str = "frontend"):
        self.path = str(path)
        self.side = side
        self._writer = None  # worker side only

    # --- worker side -------------------------------------------------
    def register(self, endpoints: Dict[int, dict]) -> str:
        if self._writer is None:
            self._writer = WorkerEndpointWriter(self.path)
        return self._writer.set_endpoints(endpoints)

    def renew(self) -> None:
        if self._writer is not None:
            self._writer.heartbeat()

    def revoke(self) -> None:
        if self._writer is not None:
            self._writer.cleanup()

    @property
    def uuid(self) -> str:
        return self._writer.transport_uuid if self._writer is not None else None

    # --- frontend side ----------------------------------------------
    def latest(self):
        return read_worker_endpoint_file(self.path)

    def age(self) -> Optional[float]:
        return endpoint_file_age_seconds(self.path)


class ProxyEndpointRegistry(EndpointRegistry):
    """Network registry backed by a standalone in-memory proxy process.

    The proxy is ``gllm.entrypoints.discovery_server`` (a ZMQ ROUTER
    ``DiscoveryServer``); this adapter is a ``NetworkDiscovery`` client
    scoped to the ``gllm-worker`` role.

    * WORKER: ``register`` publishes the transport rows as the payload
      under identity == the launch uuid and starts the lease-heartbeat
      thread (ttl/3). The proxy reaps the entry when the lease lapses
      (SIGKILL / power-loss backstop), and re-registration is automatic if
      the proxy restarts while the worker lives.
    * FRONTEND: ``latest``/``age`` read the single ``gllm-worker`` member.
    """

    ROLE = "gllm-worker"
    # Lease: 3x STALE_AFTER so a proxy-side expiry maps onto the same
    # "dead fleet" window the file impl uses (STALE_AFTER_SECONDS).
    DEFAULT_TTL_MS = int(STALE_AFTER_SECONDS * 1000 * 3)

    def __init__(self, endpoint: str, side: str = "frontend",
                 ttl_ms: int = DEFAULT_TTL_MS, rpc_timeout_ms: int = 3000):
        from gllm.disagg.discovery import NetworkDiscovery

        self.endpoint = str(endpoint)
        self.side = side
        self._ttl = ttl_ms
        self._nd = NetworkDiscovery(endpoint, ttl_ms=ttl_ms,
                                    rpc_timeout_ms=rpc_timeout_ms)
        self._last_seen_at: Optional[float] = None
        # Workers publish under identity == a fresh launch uuid; frontends
        # only read. (No eager publish in the ctor: the worker calls
        # register() once it has bound its sockets, exactly like the file
        # writer's set_endpoints.)

    # --- worker side -------------------------------------------------
    def register(self, endpoints: Dict[int, dict]) -> str:
        # identity == launch uuid: one launch == one member; a restarted
        # worker mints a fresh uuid (a fresh member) and the old lease
        # reaps out. The payload carries the same uuid so the frontend can
        # read a stable transport id (matching the file format).
        self._launch_uuid = new_transport_uuid()
        self._nd.publish(
            self.ROLE,
            self._launch_uuid,
            {"uuid": self._launch_uuid,
             "endpoints": {str(k): v for k, v in endpoints.items()},
             "updated_at": time.time()},
            ttl_ms=self._ttl,
        )
        logger.info(
            "Registered worker fleet in proxy %s (uuid %s, %d rank(s))",
            self.endpoint, self._launch_uuid, len(endpoints),
        )
        return self._launch_uuid

    def renew(self) -> None:
        # The NetworkDiscovery heartbeat thread renews the lease on its own
        # (ttl/3) and auto re-registers on a registry restart; nothing to do
        # here (kept for interface parity with the file registry's manual
        # heartbeat).
        return None

    def revoke(self) -> None:
        self._nd.revoke()

    @property
    def uuid(self) -> str:
        return getattr(self, "_launch_uuid", None)

    # --- frontend side ----------------------------------------------
    def latest(self):
        members = self._nd.list(self.ROLE)
        if not members:
            return None, None
        payload = members[0]["payload"]
        endpoints = payload.get("endpoints")
        if endpoints is not None:
            endpoints = {int(k): v for k, v in endpoints.items()}
        self._last_seen_at = time.monotonic()
        return (payload.get("uuid") or members[0]["identity"]), endpoints

    def age(self) -> Optional[float]:
        members = self._nd.list(self.ROLE)
        if not members:
            return None
        upd = members[0]["payload"].get("updated_at")
        if upd is None:
            # Fall back to "when we last saw a live member" (monotonic).
            return (
                None if self._last_seen_at is None
                else time.monotonic() - self._last_seen_at
            )
        return max(0.0, time.time() - upd)


def build_endpoint_registry(args_or_dict) -> EndpointRegistry:
    """Choose the registry backend for a standalone endpoint.

    Reads (from an argparse Namespace or dict):
      * ``endpoint_registry``: ``file`` (default) or ``proxy``.
      * ``endpoint_registry_addr``: HOST:PORT (proxy only).
      * ``worker_endpoint_file``: path (file mode; required in file mode).

    Returns an :class:`EndpointRegistry`. In file mode with no side hint the
    caller picks the side; here we infer it from the presence of the
    ``standalone_worker`` flag (worker side writes, frontend side reads).
    """
    if hasattr(args_or_dict, "endpoint_registry"):
        get = lambda k, d=None: getattr(args_or_dict, k, d)
    else:
        get = lambda k, d=None: args_or_dict.get(k, d)

    kind = (get("endpoint_registry") or "file").strip().lower()
    if kind == "proxy":
        addr = get("endpoint_registry_addr")
        if not addr:
            raise ValueError(
                "endpoint_registry='proxy' requires endpoint_registry_addr "
                "(HOST:PORT of the discovery proxy, e.g. 127.0.0.1:9500)."
            )
        side = "worker" if get("standalone_worker") else "frontend"
        return ProxyEndpointRegistry(addr, side=side)
    if kind != "file":
        raise ValueError(f"Unknown endpoint_registry {kind!r} (want 'file'|'proxy').")
    path = get("worker_endpoint_file")
    side = "worker" if get("standalone_worker") else "frontend"
    return FileEndpointRegistry(path, side=side)
