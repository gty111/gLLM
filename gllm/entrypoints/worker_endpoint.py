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
            if os.path.exists(self.path):
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
