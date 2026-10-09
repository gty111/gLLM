"""Wire formats for the encoder-disaggregation control plane.

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
    """LM PP0 -> Encoder: "encode this one mm item".

    ``content`` is the *raw* mm reference (image URL / path / base64 / video
    ref) exactly as the OpenAI request carried it -- the encoder owns all pixel
    IO + processing.

    The job carries no destination: the LM learns the item's row count from
    :class:`MmItemMeta`, allocates that many embedding pages in its cache arena
    and replies with an :class:`EmbTarget`. Under LM tensor parallelism the
    *same* embedding is needed on every LM TP rank, so it is multi-written into
    every rank's arena (``dst_regions``, rank order; index 0 == TP0) at the
    same page ids, and a *single* notification goes to TP0
    (``lm_agent_names[0]``).
    """

    seq_id: int
    # Item index in *prompt order* (matching the skeleton sentinel order):
    # ``DisaggCoordinator._try_dispatch`` assigns it by ``enumerate(mm_items)``,
    # so it pairs the i-th sentinel with the i-th encoder job regardless of
    # modality interleaving.
    item_idx: int
    modality: str  # "image" | "video"
    content: object
    # Each LM TP rank's whole registered arena, plus its embedding page layout
    # (identical across ranks): page ``p`` row ``r`` is at byte
    # ``p * dst_page_stride + r * row_bytes`` of the region.
    dst_regions: List[RemoteRegion] = field(default_factory=list)
    dst_page_stride: int = 0
    dst_rows_per_page: int = 0
    # LM meta-channel (TP0) + per-rank NIXL agent names so a freshly discovered
    # encoder can reply without a separate registry round-trip. ``lm_agent_names[0]`` is TP0 and is the single notification target.
    lm_meta_addr: str = ""
    lm_agent_names: List[str] = field(default_factory=list)


@dataclass
class EmbTarget:
    """LM PP0 -> Encoder (job channel): the arena pages, in row order, that
    rows of item ``(seq_id, item_idx)`` must be written to on every LM rank."""

    seq_id: int
    item_idx: int
    pages: List[int]


@dataclass
class EmbCancel:
    """LM PP0 -> Encoder (job channel): stop working on (and writing) item
    ``(seq_id, item_idx)``; it was re-dispatched to another encoder."""

    seq_id: int
    item_idx: int


@dataclass
class MmItemMeta:
    """Encoder -> LM PP0: per-item position/shape/hash, sent *before* the ViT.

    This is the control-plane half of the per-item channel: it
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
    # Optional carry-through for video m-rope timing (unused for images).
    second_per_grid_ts: Optional[float] = None


def emb_notif(seq_id: int, item_idx: int) -> bytes:
    """Canonical NIXL notification payload for "(seq, item) embedding ready"."""
    return f"emb:{seq_id}:{item_idx}".encode()


def emb_partial_notif(seq_id: int, item_idx: int, rows: int) -> bytes:
    """Encoder -> LM TP0: rows ``[0, rows)`` of the item's embedding have
    landed in every rank's pages (segment-streamed video). The final segment is
    signalled with the regular :func:`emb_notif`."""
    return f"embp:{seq_id}:{item_idx}:{rows}".encode()


def emb_fail_notif(seq_id: int, item_idx: int) -> bytes:
    """Encoder -> LM TP0: this replica gave up on the item (bad input,
    transfer failure, ...); the LM re-dispatches it right away."""
    return f"embf:{seq_id}:{item_idx}".encode()


def parse_emb_fail_notif(msg: bytes) -> Optional[Tuple[int, int]]:
    """Inverse of :func:`emb_fail_notif`; ``None`` for anything else."""
    try:
        s = msg.decode()
        if not s.startswith("embf:"):
            return None
        _, sid, iid = s.split(":")
        return int(sid), int(iid)
    except (UnicodeDecodeError, ValueError):
        return None


def parse_emb_partial_notif(msg: bytes) -> Optional[Tuple[int, int, int]]:
    """Inverse of :func:`emb_partial_notif`; ``None`` for anything else."""
    try:
        s = msg.decode()
        if not s.startswith("embp:"):
            return None
        _, sid, iid, rows = s.split(":")
        return int(sid), int(iid), int(rows)
    except (UnicodeDecodeError, ValueError):
        return None


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
