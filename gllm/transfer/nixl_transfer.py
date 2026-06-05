"""NIXL transfer layer for gLLM encoder disaggregation.

This module wraps NVIDIA NIXL (``nixl._api.nixl_agent``) and exposes a small,
gLLM-flavoured interface used by the encoder-disaggregation data plane:

    Encoder (initiator) ---- NIXL WRITE (GPU->GPU) ----> LM PP0 Worker (target)

The only payload this layer is ever used for is the per-item *visual
embedding* tensor (see ``docs/encoder_disaggregation_design.md`` §1.2.1 and
§5.2). It is deliberately agnostic to that fact -- it just moves contiguous
GPU tensors between two registered memory regions -- but no other gLLM
subsystem should reuse it to ship KV cache / hidden state / sampling output.

Design notes
------------
* The initiator side ("encoder") issues ``WRITE`` because a write costs one
  fewer round trip than a read (design §4.1).
* The target side ("LM") pre-registers a persistent slot-pool tensor once and
  hands out :class:`RemoteRegion` descriptors (one per slot) to the initiator
  via the control plane; the initiator then writes directly into the slot's
  byte range. No register/unregister happens on the hot path.
* Completion is observable two ways: polling ``is_done(handle)`` on the
  initiator, or a NIXL notification (``notify`` / ``poll_notifs``) delivered to
  the target once the write lands. The design uses the notification as the
  per-item "embedding ready" data-gate signal (§5.5.1 / §6.2 gate B).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

# NIXL import is deferred-friendly: importing this module must not hard-fail in
# environments without NIXL (e.g. the monolith baseline / CI without RDMA). The
# actual agent is only constructed in :meth:`NixlEndpoint.__init__`.
try:  # pragma: no cover - exercised only where NIXL is installed
    from nixl._api import nixl_agent, nixl_agent_config

    _NIXL_AVAILABLE = True
    _NIXL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover
    nixl_agent = None  # type: ignore
    nixl_agent_config = None  # type: ignore
    _NIXL_AVAILABLE = False
    _NIXL_IMPORT_ERROR = _e


def nixl_available() -> bool:
    return _NIXL_AVAILABLE


@dataclass
class RemoteRegion:
    """Serializable descriptor of a (sub)region of a *remote* registered tensor.

    Sent over the control plane (ZMQ / discovery payload) from the target
    (LM PP0) to the initiator (encoder) so the initiator knows exactly which
    bytes to write into. Carries enough shape/dtype metadata to validate the
    write before issuing it.
    """

    agent_name: str  # NIXL agent name of the *owner* (target) of this region
    base_addr: int  # absolute device pointer of the region start (bytes)
    length: int  # region length in bytes
    dev_id: int  # CUDA device ordinal on the owner
    # Optional shape metadata for validation / bookkeeping. Not used by the
    # raw byte transfer itself.
    feat_dim: int = 0
    dtype: str = ""

    def with_offset(self, offset_bytes: int, length_bytes: int) -> "RemoteRegion":
        """Return a sub-region descriptor (used to target a single slot)."""
        assert 0 <= offset_bytes
        assert offset_bytes + length_bytes <= self.length, (
            f"sub-region [{offset_bytes}, {offset_bytes + length_bytes}) "
            f"exceeds region length {self.length}"
        )
        return RemoteRegion(
            agent_name=self.agent_name,
            base_addr=self.base_addr + offset_bytes,
            length=length_bytes,
            dev_id=self.dev_id,
            feat_dim=self.feat_dim,
            dtype=self.dtype,
        )


@dataclass
class RegHandle:
    """Handle to a locally registered memory region.

    Keeps a reference to the backing tensor so it cannot be GC'd while NIXL
    still holds the registration, plus the opaque NIXL reg-descriptor list
    needed to deregister later.
    """

    tensor: torch.Tensor
    reg_descs: object
    base_addr: int
    length: int
    dev_id: int

    def region(self) -> RemoteRegion:
        raise RuntimeError("Use NixlEndpoint.region(); agent_name lives on the endpoint")


class XferHandle:
    """Thin wrapper around a NIXL transfer handle with a stable identity."""

    __slots__ = ("nixl_handle", "src", "remote", "created_at")

    def __init__(self, nixl_handle, src: torch.Tensor, remote: RemoteRegion):
        self.nixl_handle = nixl_handle
        self.src = src
        self.remote = remote
        self.created_at = time.monotonic()


class NixlEndpoint:
    """A NIXL agent with gLLM-semantic register/connect/write/notify helpers.

    One endpoint per process (per role). The encoder constructs one named
    ``encoder-<id>`` and the LM PP0 worker constructs one named ``lm-<id>``.
    """

    def __init__(
        self,
        name: str,
        backends: Sequence[str] = ("UCX",),
        enable_listen_thread: bool = False,
        listen_port: int = 0,
    ):
        if not _NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL is not available in this environment; cannot construct "
                f"NixlEndpoint. Original import error: {_NIXL_IMPORT_ERROR!r}"
            )
        self.name = name
        self.backends = list(backends)
        cfg = nixl_agent_config(
            backends=self.backends,
            enable_listen_thread=enable_listen_thread,
            listen_port=listen_port,
        )
        self.agent = nixl_agent(name, cfg)
        # name -> reg handle (so re-register of the same tensor is idempotent)
        self._regs: List[RegHandle] = []
        # remote agent name -> True (already added)
        self._remote_agents: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Metadata / connection
    # ------------------------------------------------------------------
    def local_meta(self) -> bytes:
        """Serialized agent metadata to publish so peers can ``connect``."""
        return self.agent.get_agent_metadata()

    def connect(self, peer_meta: bytes) -> str:
        """Add a remote agent from its serialized metadata. Returns its name.

        Idempotent: re-adding an already known peer is a cheap no-op.
        """
        peer_name = self.agent.add_remote_agent(peer_meta)
        if isinstance(peer_name, bytes):
            peer_name = peer_name.decode()
        self._remote_agents[peer_name] = True
        return peer_name

    def is_connected(self, peer_name: str) -> bool:
        return peer_name in self._remote_agents

    def disconnect(self, peer_name: str) -> None:
        """Drop a remote agent (peer left / lease expired). Best-effort.

        Used by the discovery REMOVE/UPDATE path so a restarted peer can be
        re-added with fresh metadata. A no-op if the backend lacks
        ``remove_remote_agent`` or the peer is already gone.
        """
        try:
            self.agent.remove_remote_agent(peer_name)
        except Exception:
            pass
        self._remote_agents.pop(peer_name, None)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, tensor: torch.Tensor) -> RegHandle:
        """Register a contiguous GPU tensor as a NIXL memory region (VRAM)."""
        assert tensor.is_contiguous(), "NIXL registration needs a contiguous tensor"
        mem_type = "VRAM" if tensor.is_cuda else "DRAM"
        reg_descs = self.agent.register_memory(tensor, mem_type)
        dev_id = tensor.get_device()
        if dev_id == -1:
            dev_id = 0
        handle = RegHandle(
            tensor=tensor,
            reg_descs=reg_descs,
            base_addr=tensor.data_ptr(),
            length=tensor.numel() * tensor.element_size(),
            dev_id=dev_id,
        )
        self._regs.append(handle)
        return handle

    def region(self, reg: RegHandle, feat_dim: int = 0, dtype: str = "") -> RemoteRegion:
        """Build a publishable :class:`RemoteRegion` for a local registration."""
        return RemoteRegion(
            agent_name=self.name,
            base_addr=reg.base_addr,
            length=reg.length,
            dev_id=reg.dev_id,
            feat_dim=feat_dim,
            dtype=dtype,
        )

    def deregister(self, reg: RegHandle) -> None:
        try:
            self.agent.deregister_memory(reg.reg_descs)
        except Exception:
            pass
        if reg in self._regs:
            self._regs.remove(reg)

    # ------------------------------------------------------------------
    # Data plane: WRITE local src -> remote region
    # ------------------------------------------------------------------
    def write(
        self,
        src: torch.Tensor,
        remote: RemoteRegion,
        notif_msg: bytes = b"",
    ) -> XferHandle:
        """Issue a one-shot WRITE of ``src`` into ``remote``.

        ``src`` must be contiguous and its byte size must match ``remote.length``
        (the caller is expected to size the remote sub-region to the exact item
        tensor; see :meth:`RemoteRegion.with_offset`). The optional
        ``notif_msg`` is delivered to the target on write completion and is how
        the LM learns "(seq_id, item_idx) embedding ready".
        """
        assert src.is_contiguous(), "NIXL write source must be contiguous"
        nbytes = src.numel() * src.element_size()
        assert nbytes == remote.length, (
            f"src bytes {nbytes} != remote region bytes {remote.length}"
        )
        assert remote.agent_name in self._remote_agents, (
            f"remote agent {remote.agent_name!r} not connected; call connect() "
            "with its metadata first"
        )
        local_descs = self.agent.get_xfer_descs(src)
        remote_descs = self.agent.get_xfer_descs(
            [(remote.base_addr, nbytes, remote.dev_id)], "VRAM"
        )
        handle = self.agent.initialize_xfer(
            "WRITE", local_descs, remote_descs, remote.agent_name, notif_msg
        )
        state = self.agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(
                f"NIXL transfer to {remote.agent_name} failed to launch"
            )
        return XferHandle(handle, src, remote)

    def is_done(self, h: XferHandle) -> bool:
        state = self.agent.check_xfer_state(h.nixl_handle)
        if state == "ERR":
            raise RuntimeError("NIXL transfer entered ERR state")
        return state == "DONE"

    def wait(self, h: XferHandle, timeout_s: Optional[float] = 30.0) -> None:
        start = time.monotonic()
        while not self.is_done(h):
            if timeout_s is not None and time.monotonic() - start > timeout_s:
                raise TimeoutError(
                    f"NIXL transfer to {h.remote.agent_name} timed out after "
                    f"{timeout_s}s"
                )
            time.sleep(0.0001)

    def release(self, h: XferHandle) -> None:
        try:
            self.agent.release_xfer_handle(h.nixl_handle)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lightweight control-plane signalling (optional alt to ZMQ)
    # ------------------------------------------------------------------
    def notify(self, peer_name: str, msg: bytes) -> None:
        self.agent.send_notif(peer_name, msg)

    def poll_notifs(self) -> Dict[str, List[bytes]]:
        """Return newly arrived notifications keyed by sender agent name."""
        return self.agent.get_new_notifs()
