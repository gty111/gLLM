"""Static / file-based discovery + NIXL handshake (design §7.3, Phase 3b).

Phase 3b only needs *one* encoder and *one* LM to find each other and exchange:

    * NIXL agent metadata  (so each can ``add_remote_agent`` the other and the
      encoder can WRITE into the LM's slot pool + the LM can receive notifs)
    * ZMQ control addresses (LM job-out / encoder job-in, encoder meta-out /
      LM meta-in)

We do this through a shared directory: each side atomically writes its own
``<role>.json`` and polls for the peer's. This is deliberately the dumbest
possible registry; Phase 4 swaps it for the ETCD/file publish+watch service
with leases + heartbeats without changing the payload schema below.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class PeerInfo:
    role: str  # "lm" | "encoder"
    agent_name: str
    nixl_meta_b64: str
    zmq_addr: str  # the socket this peer *binds* and the other *connects* to
    feat_dim: int = 0
    extra: Optional[dict] = None

    @property
    def nixl_meta(self) -> bytes:
        return base64.b64decode(self.nixl_meta_b64)


def _path(discovery_dir: str, role: str) -> str:
    return os.path.join(discovery_dir, f"{role}.json")


def _instance_path(discovery_dir: str, role: str, instance_id: str) -> str:
    return os.path.join(discovery_dir, f"{role}.{instance_id}.json")


def publish(
    discovery_dir: str,
    *,
    role: str,
    agent_name: str,
    nixl_meta: bytes,
    zmq_addr: str,
    feat_dim: int = 0,
    extra: Optional[dict] = None,
    instance_id: Optional[str] = None,
) -> None:
    """Atomically write this process's :class:`PeerInfo` into the registry."""
    os.makedirs(discovery_dir, exist_ok=True)
    info = PeerInfo(
        role=role,
        agent_name=agent_name,
        nixl_meta_b64=base64.b64encode(nixl_meta).decode(),
        zmq_addr=zmq_addr,
        feat_dim=feat_dim,
        extra=extra,
    )
    final = (
        _instance_path(discovery_dir, role, instance_id)
        if instance_id is not None
        else _path(discovery_dir, role)
    )
    tmp = f"{final}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(info), f)
    os.replace(tmp, final)
    # Backward-compatible alias for legacy single-peer readers.
    if instance_id is not None:
        alias = _path(discovery_dir, role)
        alias_tmp = f"{alias}.{os.getpid()}.tmp"
        with open(alias_tmp, "w") as f:
            json.dump(asdict(info), f)
        os.replace(alias_tmp, alias)


def read_peer(discovery_dir: str, role: str) -> Optional[PeerInfo]:
    """Non-blocking read of a peer's published info (``None`` if absent)."""
    p = _path(discovery_dir, role)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # mid-write; caller retries
    return PeerInfo(**data)


def read_peers(discovery_dir: str, role: str) -> list[PeerInfo]:
    """Return all published instances for ``role`` (possibly empty)."""
    peers: list[PeerInfo] = []
    for p in sorted(glob.glob(_instance_path(discovery_dir, role, "*"))):
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        try:
            peers.append(PeerInfo(**data))
        except TypeError:
            continue
    if peers:
        return peers
    # Fallback to legacy single-file registry.
    one = read_peer(discovery_dir, role)
    return [one] if one is not None else []


def wait_for_peer(
    discovery_dir: str, role: str, timeout_s: float = 120.0, poll_s: float = 0.2
) -> PeerInfo:
    """Block until the peer of ``role`` publishes its info, or time out."""
    start = time.monotonic()
    while True:
        info = read_peer(discovery_dir, role)
        if info is not None:
            return info
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"timed out after {timeout_s}s waiting for '{role}' to publish "
                f"in {discovery_dir}"
            )
        time.sleep(poll_s)


def wait_for_peers(
    discovery_dir: str,
    role: str,
    *,
    min_peers: int = 1,
    timeout_s: float = 120.0,
    poll_s: float = 0.2,
) -> list[PeerInfo]:
    """Block until at least ``min_peers`` instances of ``role`` are published."""
    start = time.monotonic()
    while True:
        peers = read_peers(discovery_dir, role)
        if len(peers) >= min_peers:
            return peers
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"timed out after {timeout_s}s waiting for >= {min_peers} "
                f"'{role}' peers in {discovery_dir}; found {len(peers)}"
            )
        time.sleep(poll_s)
