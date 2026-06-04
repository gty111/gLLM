"""Wire formats for the encoder-disaggregation control plane (design §5.2).

All messages are plain dataclasses shipped as pickled python objects over ZMQ
PUSH/PULL sockets. They are intentionally tiny -- the bulk payload (the visual
embedding tensor) never travels the control plane; it goes GPU->GPU over NIXL
(:mod:`gllm.transfer.nixl_transfer`). Keep these picklable and dependency-free
(no torch tensors) so the encoder and LM can exchange them without importing
each other's heavy modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from gllm.transfer.nixl_transfer import RemoteRegion


@dataclass
class EncoderJob:
    """LM PP0 -> Encoder: "encode this one mm item into that one slot".

    ``content`` is the *raw* mm reference (image URL / path / base64 / video
    ref) exactly as the OpenAI request carried it -- the encoder owns all pixel
    IO + processing (design §3.1).

    Under LM tensor parallelism the *same* visual embedding is needed (full,
    un-sharded) on every LM TP rank, so the embedding is multi-written: one
    NIXL region per LM TP rank in ``remote_slots`` (rank order; index 0 == TP0),
    all sharing the same ``slot_id``. The encoder writes all of them and then
    sends a *single* notification to TP0 (``lm_agent_names[0]``), so TP0's
    embedding-ready gate implicitly means "every rank's write landed". For
    ``tp_size == 1`` both lists have length one and this reduces to the original
    single-write path.
    """

    seq_id: int
    # Item index in *prompt order* (matching the skeleton sentinel order):
    # ``DisaggCoordinator._try_dispatch`` assigns it by ``enumerate(mm_items)``,
    # so it pairs the i-th sentinel with the i-th encoder job regardless of
    # modality interleaving.
    item_idx: int
    modality: str  # "image" | "video"
    content: object
    # One pre-registered NIXL slot region per LM TP rank (rank order). The
    # encoder WRITEs the embedding into each; ``slot_id`` (identical across
    # ranks) is echoed back in the notification so the LM can match the write to
    # the reservation.
    remote_slots: List[RemoteRegion] = field(default_factory=list)
    slot_id: int = -1
    # LM meta-channel (TP0) + per-rank NIXL agent names so a freshly discovered
    # encoder can reply without a separate registry round-trip (design §5.2 /
    # §7.3). ``lm_agent_names[0]`` is TP0 and is the single notification target.
    lm_meta_addr: str = ""
    lm_agent_names: List[str] = field(default_factory=list)


@dataclass
class MmItemMeta:
    """Encoder -> LM PP0: per-item position/shape/hash, sent *before* the ViT.

    This is the control-plane half of the per-item channel (design §5.4): it
    lets PP0 expand the skeleton sentinel into ``num_tokens`` placeholder ids
    and build the prefix-cache key (``content_hash``) without waiting for the
    embedding bytes. The embedding-ready signal is delivered separately as a
    NIXL notification once the WRITE lands.
    """

    seq_id: int
    item_idx: int
    modality: str
    num_tokens: int  # N_vis_i = prod(grid_thw)/merge**2
    feat_dim: int
    grid_thw: Tuple[int, ...]
    content_hash: bytes
    slot_id: int = -1
    # Optional carry-through for video m-rope timing (unused for images).
    second_per_grid_ts: Optional[float] = None


def emb_notif(seq_id: int, item_idx: int) -> bytes:
    """Canonical NIXL notification payload for "(seq, item) embedding ready"."""
    return f"emb:{seq_id}:{item_idx}".encode()


def parse_emb_notif(msg: bytes) -> Optional[Tuple[int, int]]:
    """Inverse of :func:`emb_notif`; returns ``None`` for unrelated notifs.

    Must never raise: a malformed/stray notification (wrong field count,
    non-integer ids, bad encoding) is treated as non-fatal and returns
    ``None`` so the LM disagg poll loop keeps running instead of crashing.
    """
    try:
        s = msg.decode()
        if not s.startswith("emb:"):
            return None
        _, sid, iid = s.split(":")
        return int(sid), int(iid)
    except (UnicodeDecodeError, ValueError):
        return None
