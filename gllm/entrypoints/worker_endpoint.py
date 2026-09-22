"""Standalone worker/frontend rendezvous via a network endpoint registry.

When the gLLM frontend and GPU worker fleet are deployed as independent
processes (a frontend crash never takes down the workers, and a worker crash
never drags down the frontend), the two sides rendezvous through a small
*endpoint registry* -- an independent, in-memory proxy middleware (a ZMQ
``DiscoveryServer``, see :mod:`gllm.disagg.discovery`). The worker registers
its frontend-facing transport rows under a per-launch ``uuid`` and leases
them; the frontend discovers them. A worker restart mints a new uuid, which
forces the frontend to re-connect its ZMQ sockets in-process (no frontend
process restart); a frontend restart simply re-discovers the still-running
fleet.

The middleware is CONTROL-PLANE ONLY: it stores ``(uuid -> transport rows +
lease)``. The data plane stays frontend<->worker POINT-TO-POINT zmq using the
addresses the registry hands out -- the registry is never in the request path
(it does not forward tokens), so it is not a request-path single point.
"""

import time
import uuid as uuid_mod
from typing import Dict, Optional

from logger import logger

# Staleness window shared with the frontend's fleet-down grace. The proxy
# lease (below) is a multiple of this so a proxy-side lease expiry maps onto
# the same "dead fleet" detection the liveness grace uses.
STALE_AFTER_SECONDS = 10.0


def new_transport_uuid() -> str:
    return uuid_mod.uuid4().hex


# ============================================================================
# Endpoint registry (network)
# ============================================================================
#
# The standalone worker registers and the frontend discovers through a small
# interface (below). There is a single implementation:
# NetworkEndpointRegistry, a client of the standalone in-memory registry
# process (gllm.entrypoints.discovery_server / gllm.disagg.discovery).
#
#   register/renew/revoke  -> worker side (publish + lease heartbeat)
#   latest/age             -> frontend side (discover + staleness backstop)


class EndpointRegistry:
    """Interface between the standalone worker/frontend and their registry.

    Two sides:

    * WORKER: :meth:`register` / :meth:`renew` / :meth:`revoke` (+
      :meth:`uuid`). A background renewal keeps the lease fresh.
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
        """Withdraw this worker (idempotent)."""
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


class NetworkEndpointRegistry(EndpointRegistry):
    """The endpoint registry: a client of the standalone in-memory proxy.

    The proxy is ``gllm.entrypoints.discovery_server`` (a ZMQ ROUTER
    ``DiscoveryServer``); this adapter is a ``NetworkDiscovery`` client
    scoped to the ``gllm-worker`` role.

    * WORKER: :meth:`register` publishes the transport rows under identity
      == the launch uuid and starts the lease-heartbeat thread (ttl/3). The
      proxy reaps the entry when the lease lapses (SIGKILL / power-loss
      backstop); re-registration is automatic if the proxy restarts while
      the worker lives.
    * FRONTEND: :meth:`latest` / :meth:`age` read the single ``gllm-worker``
      member.
    """

    ROLE = "gllm-worker"
    # Lease: 3x STALE_AFTER so a proxy-side lease expiry maps onto the same
    # "dead fleet" window the liveness grace (STALE_AFTER_SECONDS) uses.
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
        # register() once it has bound its sockets.)

    # --- worker side -------------------------------------------------
    def register(self, endpoints: Dict[int, dict]) -> str:
        # identity == launch uuid: one launch == one member; a restarted
        # worker mints a fresh uuid (a fresh member) and the old lease
        # reaps out. The payload carries the same uuid so the frontend can
        # read a stable transport id.
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
        # here (kept for interface parity).
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
    """Build the network endpoint registry for a standalone endpoint.

    Reads ``endpoint_registry_addr`` (HOST:PORT of the discovery proxy;
    required) and infers the side from ``standalone_worker`` (worker side
    registers, frontend side discovers).
    """
    if hasattr(args_or_dict, "endpoint_registry"):
        get = lambda k, d=None: getattr(args_or_dict, k, d)
    else:
        get = lambda k, d=None: args_or_dict.get(k, d)

    addr = get("endpoint_registry_addr")
    if not addr:
        raise ValueError(
            "standalone deployment requires endpoint_registry_addr "
            "(HOST:PORT of the discovery proxy, e.g. 127.0.0.1:9500). "
            "Start one: python -m gllm.entrypoints.discovery_server "
            "--listen 0.0.0.0:9500."
        )
    side = "worker" if get("standalone_worker") else "frontend"
    return NetworkEndpointRegistry(addr, side=side)
