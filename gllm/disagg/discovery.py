"""Dynamic discovery for encoder disaggregation (design §7.3).

The Encoder and LM servers are fully decoupled processes: neither knows the
other's address at launch. They find each other through a shared *registry*
(the ``--discovery-endpoint``) onto which each ``publish``-es its own NIXL agent
metadata + ZMQ control address + a ``processor_config_hash``, and from which
each ``watch``-es the peer role. Watch events (ADD / UPDATE / REMOVE) drive the
runtime NIXL handshake and teardown, so:

  * either side may start first (publish + watch are symmetric);
  * killing an encoder lets its lease expire -> peers get REMOVE and drop it;
  * a restarted encoder re-publishes -> peers get ADD and reconnect;
  * a processor-config mismatch is rejected at connect time (§5.4.4).

Discovery is network-only: :class:`NetworkDiscovery` talks to a standalone
:class:`DiscoveryServer` over ZMQ DEALER/ROUTER (TCP). Pure network, no shared
filesystem. This is the etcd stand-in for single-/cross-node bring up without an
etcd dependency.

Lease semantics: a publisher renews every ``ttl_ms/3``; a member whose lease is
not renewed within ``ttl_ms`` is reaped and surfaces as a REMOVE on the next
``poll_events``.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import zmq
from logger import logger


# ----------------------------------------------------------------------------
# Events + peer payload
# ----------------------------------------------------------------------------
@dataclass
class Event:
    """A change observed by ``poll_events`` for a watched role."""

    kind: str  # "ADD" | "UPDATE" | "REMOVE"
    identity: str
    payload: dict


def make_payload(
    *,
    role: str,
    agent_name: Optional[str] = None,
    nixl_meta: Optional[bytes] = None,
    zmq_addr: str,
    feat_dim: int = 0,
    processor_config_hash: str = "",
    extra: Optional[dict] = None,
    agent_names: Optional[List[str]] = None,
    nixl_metas: Optional[List[bytes]] = None,
) -> dict:
    """Build the discovery payload dict published for a role member.

    A single NIXL agent (the common case: encoder, or a ``tp_size==1`` LM) is
    published via ``agent_name`` + ``nixl_meta``. An LM with ``tp_size>1``
    publishes *one agent per TP rank* via ``agent_names`` + ``nixl_metas`` (rank
    order; index 0 == TP0). Both forms are always emitted (the single-agent keys
    mirror element 0) so a peer can read either ``payload_nixl_meta`` (first
    agent) or ``payload_nixl_metas`` (all agents) regardless of how it was
    built. ``zmq_addr`` is always the single control endpoint (TP0 for an LM).
    """
    if nixl_metas is None:
        assert nixl_meta is not None, "make_payload needs nixl_meta or nixl_metas"
        nixl_metas = [nixl_meta]
    if agent_names is None:
        assert agent_name is not None, "make_payload needs agent_name or agent_names"
        agent_names = [agent_name]
    assert len(agent_names) == len(nixl_metas), (
        f"agent_names ({len(agent_names)}) != nixl_metas ({len(nixl_metas)})"
    )
    metas_b64 = [base64.b64encode(m).decode() for m in nixl_metas]
    return {
        "role": role,
        # Single-agent (back-compat) view == element 0.
        "agent_name": agent_names[0],
        "nixl_meta_b64": metas_b64[0],
        # Multi-agent (TP) view: one NIXL agent meta per LM TP rank.
        "agent_names": list(agent_names),
        "nixl_metas_b64": metas_b64,
        "zmq_addr": zmq_addr,
        "feat_dim": int(feat_dim),
        "processor_config_hash": processor_config_hash,
        "extra": extra or {},
    }


def payload_nixl_meta(payload: dict) -> bytes:
    return base64.b64decode(payload["nixl_meta_b64"])


def payload_nixl_metas(payload: dict) -> List[bytes]:
    """All per-rank NIXL agent metas (falls back to the single-agent key)."""
    metas_b64 = payload.get("nixl_metas_b64")
    if metas_b64 is None:
        return [base64.b64decode(payload["nixl_meta_b64"])]
    return [base64.b64decode(m) for m in metas_b64]


def payload_agent_names(payload: dict) -> List[str]:
    """All per-rank NIXL agent names (falls back to the single-agent key)."""
    names = payload.get("agent_names")
    if names is None:
        return [payload["agent_name"]]
    return list(names)


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _normalize_tcp(endpoint: str) -> str:
    """Accept ``host:port`` or ``tcp://host:port`` and return a ZMQ tcp URL."""
    if "://" in endpoint:
        return endpoint
    return f"tcp://{endpoint}"


def _endpoint_host(endpoint: str) -> str:
    """Extract the host part from ``host:port`` / ``tcp://host:port``.

    Returns ``""`` for a malformed/empty endpoint so callers fall back to
    public-IP egress detection instead of treating a bogus string as a host
    that fails to connect and collapses to loopback.
    """
    hostport = _normalize_tcp(endpoint).split("://", 1)[-1]
    host = hostport.rsplit(":", 1)[0] if ":" in hostport else hostport
    if not host or "/" in host:
        return ""
    return host


def resolve_advertise_host(advertise: Optional[str], reference_endpoint: str = "") -> str:
    """Resolve the address peers should connect back to (multi-node safe).

    A peer publishes a ZMQ control address built from this host; cross-node
    peers must therefore receive a *routable* IP, never ``127.0.0.1``. When
    ``advertise`` is ``"auto"`` / empty we detect the local egress IP toward the
    discovery server's host (the interface on the same network the registry is
    reached on), so the advertised address is reachable by every other member.
    A literal value is returned as-is (operator override). Falls back to
    loopback only if detection fails (single-node dev).
    """
    if advertise and advertise != "auto":
        return advertise
    target = _endpoint_host(reference_endpoint) or "8.8.8.8"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # No packets are sent for a UDP "connect"; it just selects the egress
        # interface whose source IP we then read back.
        s.connect((target, 9))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ----------------------------------------------------------------------------
# Registry server (standalone process; etcd stand-in)
# ----------------------------------------------------------------------------
class DiscoveryServer:
    """In-memory, lease-based registry served over a ZMQ ROUTER socket.

    Single-threaded request loop; every op is idempotent so DEALER clients can
    safely resend on timeout. State is ``identity -> {role, payload,
    expire_at}``; expired members are reaped lazily on ``list`` + periodically.
    """

    def __init__(self, bind: str, *, ctx: Optional[zmq.Context] = None):
        self.bind_url = _normalize_tcp(bind)
        self.ctx = ctx or zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind(self.bind_url)
        self._members: Dict[str, dict] = {}
        self._stop = False

    def _reap(self) -> None:
        now = _now_ms()
        dead = [i for i, m in self._members.items() if m["expire_at"] < now]
        for i in dead:
            m = self._members.pop(i)
            logger.info(f"[discovery-server] lease expired: {m['role']}/{i}")

    def _handle(self, msg: dict) -> dict:
        op = msg.get("op")
        if op == "register":
            self._members[msg["identity"]] = {
                "role": msg["role"],
                "payload": msg["payload"],
                "expire_at": _now_ms() + float(msg.get("ttl_ms", 10000)),
            }
            logger.info(
                f"[discovery-server] register {msg['role']}/{msg['identity']}"
            )
            return {"ok": True}
        if op == "renew":
            m = self._members.get(msg["identity"])
            if m is None:
                return {"ok": True, "known": False}
            m["expire_at"] = _now_ms() + float(msg.get("ttl_ms", 10000))
            return {"ok": True, "known": True}
        if op == "revoke":
            m = self._members.pop(msg["identity"], None)
            if m is not None:
                logger.info(
                    f"[discovery-server] revoke {m['role']}/{msg['identity']}"
                )
            return {"ok": True}
        if op == "list":
            self._reap()
            role = msg.get("role")
            members = [
                {"identity": i, "payload": m["payload"]}
                for i, m in self._members.items()
                if role is None or m["role"] == role
            ]
            return {"ok": True, "members": members}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def serve_forever(self) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        logger.info(f"[discovery-server] serving at {self.bind_url}")
        last_reap = _now_ms()
        while not self._stop:
            socks = dict(poller.poll(timeout=1000))
            if self.sock in socks:
                while True:
                    try:
                        frames = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    sender, raw = frames[0], frames[-1]
                    try:
                        reply = self._handle(json.loads(raw.decode()))
                    except Exception as e:  # pragma: no cover - guard
                        reply = {"ok": False, "error": str(e)}
                    self.sock.send_multipart([sender, json.dumps(reply).encode()])
            if _now_ms() - last_reap > 2000:
                self._reap()
                last_reap = _now_ms()

    def stop(self) -> None:
        self._stop = True


# ----------------------------------------------------------------------------
# Network client
# ----------------------------------------------------------------------------
class NetworkDiscovery:
    """Client to a :class:`DiscoveryServer`. Publish + heartbeat + watch."""

    def __init__(
        self,
        endpoint: str,
        *,
        ttl_ms: int = 10000,
        rpc_timeout_ms: int = 3000,
        ctx: Optional[zmq.Context] = None,
    ):
        self.endpoint = _normalize_tcp(endpoint)
        self.ttl_ms = ttl_ms
        self.rpc_timeout_ms = rpc_timeout_ms
        self.ctx = ctx or zmq.Context.instance()
        self._ctrl_lock = threading.Lock()
        self._ctrl = self._new_sock()
        self._identity: Optional[str] = None
        self._role: Optional[str] = None
        self._payload: Optional[dict] = None
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        # role -> {identity: payload} snapshot for event diffing.
        self._seen: Dict[str, Dict[str, dict]] = {}

    # -- low level ----------------------------------------------------------
    def _new_sock(self) -> zmq.Socket:
        s = self.ctx.socket(zmq.DEALER)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.endpoint)
        return s

    def _do_rpc(self, sock: zmq.Socket, msg: dict, retries: int = 5):
        """Send ``msg`` and await a reply, reopening the socket on timeout.

        Returns ``(sock, reply)`` because the socket may be replaced. All ops
        are idempotent, so a resend after timeout is safe.
        """
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        for _ in range(retries):
            sock.send(json.dumps(msg).encode())
            if dict(poller.poll(timeout=self.rpc_timeout_ms)).get(sock):
                return sock, json.loads(sock.recv().decode())
            # Timed out: drop the stuck socket and retry on a fresh one.
            poller.unregister(sock)
            sock.close(0)
            sock = self._new_sock()
            poller.register(sock, zmq.POLLIN)
        raise TimeoutError(
            f"discovery RPC '{msg.get('op')}' timed out against {self.endpoint}"
        )

    def _ctrl_rpc(self, msg: dict) -> dict:
        with self._ctrl_lock:
            self._ctrl, reply = self._do_rpc(self._ctrl, msg)
            return reply

    # -- publish / heartbeat ------------------------------------------------
    def publish(
        self, role: str, identity: str, payload: dict, ttl_ms: Optional[int] = None
    ) -> None:
        ttl = int(ttl_ms or self.ttl_ms)
        self._role, self._identity, self._payload = role, identity, payload
        self._ctrl_rpc(
            {
                "op": "register",
                "role": role,
                "identity": identity,
                "payload": payload,
                "ttl_ms": ttl,
            }
        )
        self._hb_thread = threading.Thread(
            target=self._heartbeat, args=(ttl,), name="disc-heartbeat", daemon=True
        )
        self._hb_thread.start()
        logger.info(
            f"[discovery] published {role}/{identity} -> {self.endpoint} (ttl={ttl}ms)"
        )

    def _heartbeat(self, ttl_ms: int) -> None:
        sock = self._new_sock()
        interval = max(0.5, ttl_ms / 3000.0)
        while not self._hb_stop.wait(interval):
            try:
                sock, reply = self._do_rpc(
                    sock, {"op": "renew", "identity": self._identity, "ttl_ms": ttl_ms}
                )
                if reply and not reply.get("known", True):
                    # Registry restarted / lease lost -> re-register ourselves.
                    sock, _ = self._do_rpc(
                        sock,
                        {
                            "op": "register",
                            "role": self._role,
                            "identity": self._identity,
                            "payload": self._payload,
                            "ttl_ms": ttl_ms,
                        },
                    )
            except Exception as e:  # pragma: no cover - operational
                logger.warning(f"[discovery] heartbeat renew failed: {e}")
        sock.close(0)

    # -- watch --------------------------------------------------------------
    def list(self, role: str) -> List[dict]:
        reply = self._ctrl_rpc({"op": "list", "role": role})
        return reply.get("members", [])

    def poll_events(self, role: str) -> List[Event]:
        members = {m["identity"]: m["payload"] for m in self.list(role)}
        prev = self._seen.get(role, {})
        events: List[Event] = []
        for ident, payload in members.items():
            if ident not in prev:
                events.append(Event("ADD", ident, payload))
            elif payload != prev[ident]:
                events.append(Event("UPDATE", ident, payload))
        for ident, payload in prev.items():
            if ident not in members:
                events.append(Event("REMOVE", ident, payload))
        self._seen[role] = members
        return events

    # -- teardown -----------------------------------------------------------
    def revoke(self) -> None:
        self._hb_stop.set()
        if self._identity is not None:
            try:
                self._ctrl_rpc({"op": "revoke", "identity": self._identity})
            except Exception:
                pass

    def close(self) -> None:
        self.revoke()
        with self._ctrl_lock:
            try:
                self._ctrl.close(0)
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def make_discovery(endpoint: str, *, ttl_ms: int = 10000) -> NetworkDiscovery:
    """Construct the (network) discovery client for ``endpoint`` (HOST:PORT)."""
    return NetworkDiscovery(endpoint, ttl_ms=ttl_ms)
