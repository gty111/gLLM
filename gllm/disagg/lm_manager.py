"""LM-side encoder disaggregation: per-rank receiver + TP0 coordinator.

Native LM tensor parallelism (``tp_size >= 1``, ``pp_size == 1``) is supported by
splitting the old monolithic manager into two roles (design: "control
centralized, data multi-write"):

* :class:`DisaggReceiver` -- one per **every PP0 TP rank**. Owns that rank's
  NIXL receive slot pool (a persistent registered GPU tensor
  ``[num_slots, max_vis_tokens, feat_dim]``). The encoder multi-writes the
  *full* (un-sharded) visual embedding into every rank's pool; the receiver
  just clones a ready slot into ``model_runner.disagg_embeds`` on command.
  No ZMQ / discovery / dispatch logic.

* :class:`DisaggCoordinator` -- **TP0 only**. Owns the control plane: the
  :class:`MmItemMeta` ZMQ intake, encoder discovery + NIXL handshake, per-item
  :class:`EncoderJob` dispatch (carrying one slot region per LM TP rank),
  meta/notification aggregation, the re-dispatch watchdog, and slot
  reservation. It is the single authority that decides *when* a seq is admitted
  and *when* each embedding is ready, emitting those decisions as a
  :class:`DisaggEvents` stream once per iteration.

Determinism (column-driver model): the coordinator runs only on TP0 (== rank
0), and its per-iteration :class:`DisaggEvents` are fanned out to every PP0 TP
rank alongside the normal ``broadcast_input_to_tp`` input. Each rank applies the
*same* events in the *same* iteration -- ``ADMIT`` rebuilds the expanded
``Sequence`` into the scheduler, ``EMB_READY`` clones the embedding from that
rank's *own* slot pool -- so every column's scheduler / model runner stays in
lockstep. The encoder writes all N rank pools then sends a *single*
notification to TP0, so TP0's embedding-ready gate implies "every rank's write
landed".

Output stays byte-identical to the monolith (and to the original ``tp_size==1``
disagg): the only change vs. the monolith is *when* prefill is allowed to
advance, not *what* is computed.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from logger import logger

from gllm.disagg.discovery import make_discovery, make_payload, payload_nixl_meta
from gllm.disagg.protocol import EncoderJob, MmItemMeta, parse_emb_notif
from gllm.layers.rotary_embedding import MRotaryEmbedding
from gllm.model_runner import DisaggSeqState, ModelRunner
from gllm.transfer.nixl_transfer import NixlEndpoint, RemoteRegion


@dataclass
class _PendingItem:
    item_idx: int
    modality: str
    slot_id: int
    meta: Optional[MmItemMeta] = None
    embedding_ready: bool = False
    # Set once the EMB_READY event for this item has been emitted (its slot
    # returned to the free list). The actual clone out of the slot pool happens
    # per-rank when the event is applied; the coordinator only tracks emission.
    slot_freed: bool = False
    # Phase 8 failure handling (design §5.5.2): the raw item content is retained
    # so the watchdog can RE-DISPATCH the EncoderJob to another replica if the
    # original encoder crashes / goes silent. Dropped once the item is done.
    content: object = None
    encoder_identity: Optional[str] = None  # replica it was last sent to
    dispatched_at: float = 0.0              # monotonic time of last (re)dispatch
    attempts: int = 0                       # (re)dispatch count
    gave_up: bool = False                   # exceeded max attempts (logged once)

    @property
    def done(self) -> bool:
        """Both control plane (meta) and data plane (embedding) have landed."""
        return self.meta is not None and self.embedding_ready


@dataclass
class _EncoderConn:
    """A live encoder replica discovered + NIXL-connected at runtime."""

    identity: str
    agent_name: str
    zmq_addr: str
    job_sock: object


@dataclass
class _PendingSeq:
    seq: object
    items: List[_PendingItem]
    dispatched: bool = False
    # A seq is *admitted* once all per-item meta arrive (positions/hashes
    # determined; gate A); its embeddings then stream in progressively (gate B).
    # ``ordered`` is the image-then-video ordering of ``items`` (== the
    # ``embed_multimodal`` tuple order and the order of :class:`DisaggSeqState`),
    # fixed at admission.
    admitted: bool = False
    ordered: Optional[List[_PendingItem]] = None
    # Number of image/video sentinels in the skeleton ``seq.token_ids``. The
    # i-th sentinel pairs with ``items[i]``; admission expands each sentinel to
    # its item's ``num_tokens``. Recorded at dispatch so admission can verify
    # the item list matches the skeleton (guards against a sentinel/item count
    # skew, which would otherwise IndexError in ``_admit_meta_complete``).
    n_sentinels: int = 0

    @property
    def num_items(self) -> int:
        return len(self.items)

    @property
    def meta_complete(self) -> bool:
        # Gate A is met only when every skeleton sentinel has a paired item AND
        # every item carries its meta. The count guard prevents admitting (and
        # then crashing in ``_admit_meta_complete``) on a sentinel/item skew.
        return (
            len(self.items) == self.n_sentinels
            and all(it.meta is not None for it in self.items)
        )

    @property
    def all_embeddings_ready(self) -> bool:
        return all(it.embedding_ready for it in self.items)


@dataclass
class DisaggEvents:
    """Per-iteration disagg control decisions, generated authoritatively on TP0.

    Fanned out to every PP0 TP rank (riding the normal ``broadcast_input_to_tp``
    input) so each column applies the identical decisions in the identical
    iteration -- the basis of column-driver determinism under TP>1.
    """

    # (expanded ``Sequence``, freshly-built :class:`DisaggSeqState`):
    # register the state + add the seq to the scheduler.
    admits: List[Tuple[object, DisaggSeqState]] = field(default_factory=list)
    # (seq_id, ordered_idx, slot_id, num_tokens): clone the embedding from this
    # rank's *own* slot pool into ``model_runner.disagg_embeds``.
    emb_ready: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # seq_ids to abort: an *already-admitted* seq whose encode failed
    # unrecoverably (watchdog gave up). Fanned out so every column drops it from
    # its scheduler in the same iteration (model_runner.free reclaims the page /
    # SSM slot + disagg state); the coordinator separately reclaims its NIXL
    # slots. Pre-admission give-ups never reach a scheduler, so they are not
    # listed here -- only the coordinator-side slot reclamation runs for them.
    aborts: List[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.admits or self.emb_ready or self.aborts)


class DisaggReceiver:
    """Per-PP0-TP-rank NIXL receive endpoint + slot pool (data plane).

    One per LM TP rank. The encoder multi-writes the full visual embedding into
    every rank's pool; only TP0 receives the completion notification (handled by
    :class:`DisaggCoordinator`). A receiver carries no control logic -- it
    registers memory, exposes per-slot :class:`RemoteRegion` descriptors, and
    clones ready slots into the model runner when the coordinator's fanned-out
    ``EMB_READY`` events say so.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        lm_id: str,
        tp_rank: int,
        num_slots: int,
        max_vis_tokens: int,
        nixl_backend: str = "UCX",
    ):
        self.mr = model_runner
        self.lm_id = lm_id
        self.tp_rank = tp_rank
        self.num_slots = num_slots
        self.max_vis_tokens = max_vis_tokens
        self.nixl_backend = nixl_backend

        cfg = self.mr.model.config
        self.feat_dim = int(
            cfg.vision_config.out_hidden_size
            * (1 + len(getattr(cfg.vision_config, "deepstack_visual_indexes", [])))
        )
        self.dtype = self.mr.model_loader.dtype
        self.device = next(self.mr.model.parameters()).device

        self.nixl: Optional[NixlEndpoint] = None
        self.slot_pool: Optional[torch.Tensor] = None
        self.slot_reg = None
        self.pool_region: Optional[RemoteRegion] = None
        self.slot_stride_bytes = 0

    # ------------------------------------------------------------------
    def setup(self) -> None:
        # One NIXL agent per LM TP rank (distinct names so the encoder can add
        # all of them as remote agents and write each independently).
        self.nixl = NixlEndpoint(
            name=f"lm-{self.lm_id}-tp{self.tp_rank}", backends=(self.nixl_backend,)
        )
        # Zero-init (not ``empty``): a slot is read only after its NIXL write,
        # but zero-init makes any accidental early read fail loudly (all-zero
        # embedding) instead of leaking uninitialized garbage.
        self.slot_pool = torch.zeros(
            (self.num_slots, self.max_vis_tokens, self.feat_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.slot_reg = self.nixl.register(self.slot_pool)
        self.pool_region = self.nixl.region(
            self.slot_reg, feat_dim=self.feat_dim, dtype=str(self.dtype)
        )
        self.slot_stride_bytes = (
            self.max_vis_tokens * self.feat_dim * self.slot_pool.element_size()
        )
        logger.info(
            f"[lm-disagg {self.lm_id} tp{self.tp_rank}] receiver up: slot pool "
            f"{self.num_slots}x{self.max_vis_tokens}x{self.feat_dim} "
            f"({self.dtype}) agent={self.nixl.name}"
        )

    def slot_region(self, slot_id: int) -> RemoteRegion:
        return self.pool_region.with_offset(
            slot_id * self.slot_stride_bytes, self.slot_stride_bytes
        )

    def handshake(self) -> dict:
        """Per-rank NIXL handshake info gathered to TP0 during setup."""
        return {
            "agent_name": self.nixl.name,
            "nixl_meta": self.nixl.local_meta(),
            "region": self.pool_region,
        }

    def sync(self) -> None:
        """Make landed NIXL writes visible to subsequent clones.

        The NIXL notification only signals the WRITE completed; this device must
        be synchronized before reading so the written bytes are visible to the
        clone (mirrors ``tests/test_nixl_transfer.py``'s post-notif sync). One
        sync per batch of clones is sufficient.
        """
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def clone_slot(self, slot_id: int, num_tokens: int) -> torch.Tensor:
        """Clone one item's embedding out of the local slot pool. Call
        :meth:`sync` once before a batch of clones."""
        return self.slot_pool[slot_id, :num_tokens, :].clone()


class DisaggCoordinator:
    """TP0-only control plane for encoder disaggregation.

    Owns ZMQ meta intake, encoder discovery + NIXL handshake, per-item
    :class:`EncoderJob` dispatch (one slot region per LM TP rank), aggregation,
    the re-dispatch watchdog, and slot reservation. Produces a per-iteration
    :class:`DisaggEvents` stream; it never mutates the model runner directly --
    the events are applied uniformly by every column (including TP0) so all
    ranks stay deterministic.

    The NIXL endpoint used to connect to encoders and to receive completion
    notifications is **TP0's own** :class:`DisaggReceiver` endpoint (the encoder
    only notifies TP0). The slot regions for the *other* ranks are gathered into
    ``rank_handshakes`` at setup and packed into each :class:`EncoderJob`.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        recv: DisaggReceiver,
        *,
        rank_handshakes: List[dict],
        lm_id: str,
        discovery_endpoint: str,
        processor_config_hash: str = "",
        advertise_host: str = "127.0.0.1",
        meta_bind: str = "tcp://0.0.0.0:0",
        num_slots: int = 32,
        nixl_backend: str = "UCX",
        encoder_dp: int = 1,
    ):
        self.mr = model_runner
        self.recv = recv  # TP0 receiver: owns the NIXL endpoint + notifs
        self.lm_id = lm_id
        self.discovery_endpoint = discovery_endpoint
        self.processor_config_hash = processor_config_hash
        self.advertise_host = advertise_host
        self.meta_bind = meta_bind
        self.num_slots = num_slots
        self.nixl_backend = nixl_backend
        self.encoder_dp = max(1, int(encoder_dp))

        # Per-rank NIXL slot-pool regions (rank order; index 0 == TP0) gathered
        # at setup; one EncoderJob carries one sub-region per rank.
        self.rank_regions: List[RemoteRegion] = [h["region"] for h in rank_handshakes]
        self.rank_agent_names: List[str] = [h["agent_name"] for h in rank_handshakes]
        self.rank_metas: List[bytes] = [h["nixl_meta"] for h in rank_handshakes]
        self.num_ranks = len(rank_handshakes)
        self.slot_stride_bytes = recv.slot_stride_bytes

        # Phase 6 intra-request encode/prefill overlap (design §6.2). When on,
        # a seq is admitted as soon as all meta arrive and prefill advances
        # per-item under the two-layer gate. When off (default), admission waits
        # for *all* embeddings (Phase 3b timing) -> single-chunk prefill, which
        # is byte-identical to the unchunked monolith (used as a determinism
        # baseline). Set GLLM_DISAGG_OVERLAP=1 to enable overlap.
        self.overlap = os.environ.get("GLLM_DISAGG_OVERLAP", "0") != "0"
        # Phase 8 watchdog (design §5.5.2): an in-flight item whose encoder has
        # gone silent or left the pool is re-dispatched to a live replica,
        # reusing the same slot (idempotent overwrite). After
        # ``max_redispatch_attempts`` we stop retrying that item.
        self.redispatch_timeout_s = float(
            os.environ.get("GLLM_DISAGG_REDISPATCH_TIMEOUT_S", "20.0")
        )
        self.max_redispatch_attempts = int(
            os.environ.get("GLLM_DISAGG_MAX_REDISPATCH", "5")
        )

        cfg = self.mr.model.config
        self.image_token_id = int(cfg.image_token_id)
        self.video_token_id = int(cfg.video_token_id)
        self.feat_dim = recv.feat_dim

        # ZMQ state (built in setup()).
        import zmq

        self._zmq = zmq
        self.zmq_ctx = None
        self.meta_sock = None  # PULL: MmItemMeta in
        self.meta_addr: str = ""

        # Discovery + dynamic encoder registry (identity -> _EncoderConn).
        self.disc = None
        self._encoders: Dict[str, _EncoderConn] = {}
        self._rr_encoder_idx = 0
        # The PP0 overlap loop spins at full rate when idle; throttle the
        # (networked) discovery poll so it doesn't flood the registry.
        self._disc_poll_interval_s = 0.2
        self._last_disc_poll = 0.0

        # Slot pool reservation (shared layout across all ranks -- a slot_id
        # names the same byte range in every rank's pool).
        self._free_slots: List[int] = list(range(self.num_slots))

        # Bookkeeping.
        self._pending: Dict[int, _PendingSeq] = {}  # seq_id -> _PendingSeq
        self._admission_q: List[object] = []  # seqs waiting for free slots
        # meta/notif that arrived before the seq was registered (rare race).
        self._orphan_meta: List[MmItemMeta] = []
        # Hard cap on the orphan buffer so stale metas (e.g. for aborted seqs
        # that never dispatch) can't accumulate without bound. Generously sized
        # vs. the slot pool; oldest is evicted FIFO past this.
        self._max_orphan_meta = max(256, self.num_slots * 8)

    @property
    def nixl(self) -> NixlEndpoint:
        return self.recv.nixl

    # ------------------------------------------------------------------
    def setup(self) -> None:
        self.zmq_ctx = self._zmq.Context.instance()
        self.meta_sock = self.zmq_ctx.socket(self._zmq.PULL)
        self.meta_sock.bind(self.meta_bind)
        bound = self.meta_sock.getsockopt(self._zmq.LAST_ENDPOINT).decode()
        port = bound.rsplit(":", 1)[-1]
        self.meta_addr = f"tcp://{self.advertise_host}:{port}"

        # Publish self (all rank agent metas + the single TP0 meta intake) +
        # watch encoders. We do NOT block on encoders: text-only requests serve
        # immediately, and mm requests queue in the admission queue until a
        # replica connects (any start order, design §7.3.4).
        self.disc = make_discovery(self.discovery_endpoint)
        self.disc.publish(
            "lm",
            self.lm_id,
            make_payload(
                role="lm",
                agent_names=self.rank_agent_names,
                nixl_metas=self.rank_metas,
                zmq_addr=self.meta_addr,
                feat_dim=self.feat_dim,
                processor_config_hash=self.processor_config_hash,
            ),
        )
        logger.info(
            f"[lm-disagg {self.lm_id}] coordinator up: meta intake at "
            f"{self.meta_addr}; {self.num_ranks} TP rank slot pool(s) "
            f"(agents={self.rank_agent_names}); watching encoders via "
            f"{self.discovery_endpoint}"
        )
        self._drain_discovery(force=True)

    # ------------------------------------------------------------------
    # Discovery: dynamic encoder connect/disconnect
    # ------------------------------------------------------------------
    def _drain_discovery(self, force: bool = False) -> None:
        if self.disc is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_disc_poll) < self._disc_poll_interval_s:
            return
        self._last_disc_poll = now
        try:
            evs = self.disc.poll_events("encoder")
        except Exception as e:
            # A transient registry stall (RPC timeout) must not tear down the
            # shared PP0 disagg loop; skip this poll and retry next iteration.
            logger.warning(
                f"[lm-disagg {self.lm_id}] discovery poll failed "
                f"({type(e).__name__}: {e}); retrying next iteration"
            )
            return
        for ev in evs:
            if ev.kind in ("ADD", "UPDATE"):
                self._connect_encoder(
                    ev.identity, ev.payload, reconnect=ev.kind == "UPDATE"
                )
            elif ev.kind == "REMOVE":
                self._disconnect_encoder(ev.identity)

    def _connect_encoder(self, identity: str, payload: dict, *, reconnect: bool) -> None:
        peer_hash = payload.get("processor_config_hash", "")
        if (
            self.processor_config_hash
            and peer_hash
            and peer_hash != self.processor_config_hash
        ):
            logger.error(
                f"[lm-disagg {self.lm_id}] REJECT encoder {identity}: "
                f"processor_config_hash mismatch ({peer_hash[:12]} != "
                f"{self.processor_config_hash[:12]})"
            )
            return
        if reconnect and identity in self._encoders:
            self._disconnect_encoder(identity, keep_quiet=True)
        # TP0 connects to the encoder so it can receive the single completion
        # notification (the other ranks are write-only targets and never add the
        # encoder as a remote agent).
        agent = self.nixl.connect(payload_nixl_meta(payload))
        sock = self.zmq_ctx.socket(self._zmq.PUSH)
        sock.connect(payload["zmq_addr"])
        self._encoders[identity] = _EncoderConn(
            identity=identity,
            agent_name=agent,
            zmq_addr=payload["zmq_addr"],
            job_sock=sock,
        )
        logger.info(
            f"[lm-disagg {self.lm_id}] encoder {identity} connected "
            f"(agent={agent} job={payload['zmq_addr']}); "
            f"live encoders={len(self._encoders)}"
        )

    def _disconnect_encoder(self, identity: str, *, keep_quiet: bool = False) -> None:
        conn = self._encoders.pop(identity, None)
        if conn is None:
            return
        try:
            conn.job_sock.close(0)
        except Exception:
            pass
        self.nixl.disconnect(conn.agent_name)
        # Phase 8 (design §5.5.2): orphan any in-flight item routed to the
        # departed replica so the watchdog re-dispatches it immediately.
        orphaned = 0
        for ps in self._pending.values():
            for item in ps.items:
                if not item.done and item.encoder_identity == identity:
                    item.encoder_identity = None
                    orphaned += 1
        if not keep_quiet:
            logger.info(
                f"[lm-disagg {self.lm_id}] encoder {identity} left; "
                f"live encoders={len(self._encoders)}"
                + (f"; orphaned {orphaned} in-flight item(s)" if orphaned else "")
            )

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------
    def submit(self, seq) -> None:
        """Queue a disaggregated mm seq (``seq.mm_items`` set) for dispatch."""
        self._admission_q.append(seq)
        logger.debug(
            f"[lm-disagg {self.lm_id}] submit seq={seq.seq_id} "
            f"items={len(seq.mm_items)} (live encoders={len(self._encoders)}, "
            f"free slots={len(self._free_slots)})"
        )

    def _slot_regions(self, slot_id: int) -> List[RemoteRegion]:
        """One full-slot region per LM TP rank (the encoder sub-sizes each to
        the actual item with ``with_offset(0, nbytes)``)."""
        off = slot_id * self.slot_stride_bytes
        return [r.with_offset(off, self.slot_stride_bytes) for r in self.rank_regions]

    def _pick_encoder(self, avoid: Optional[str] = None) -> Optional[_EncoderConn]:
        """Round-robin a live encoder, preferring one other than ``avoid``."""
        if not self._encoders:
            return None
        encs = list(self._encoders.values())
        if avoid is not None and len(encs) > 1:
            preferred = [e for e in encs if e.identity != avoid]
            if preferred:
                encs = preferred
        conn = encs[self._rr_encoder_idx % len(encs)]
        self._rr_encoder_idx += 1
        return conn

    def _send_job(
        self, seq_id: int, item: _PendingItem, *, avoid: Optional[str] = None
    ) -> bool:
        """Send (or re-send) the EncoderJob for ``item`` to a live encoder.

        Re-dispatch reuses the same ``slot_id`` (same registered NIXL regions):
        the encoder overwrites them idempotently (design §5.5.3). Returns False
        if no encoder is currently live (caller retries on a later poll)."""
        conn = self._pick_encoder(avoid=avoid)
        if conn is None:
            return False
        job = EncoderJob(
            seq_id=seq_id,
            item_idx=item.item_idx,
            modality=item.modality,
            content=item.content,
            remote_slots=self._slot_regions(item.slot_id),
            slot_id=item.slot_id,
            lm_meta_addr=self.meta_addr,
            lm_agent_names=list(self.rank_agent_names),
        )
        conn.job_sock.send(pickle.dumps(job))
        item.encoder_identity = conn.identity
        item.dispatched_at = time.monotonic()
        item.attempts += 1
        return True

    def _try_dispatch(self) -> None:
        """Reserve slots + send EncoderJobs for queued seqs (FIFO, best-effort)."""
        while self._admission_q:
            # Backpressure: no encoder connected yet -> hold the request in the
            # queue (it will dispatch once discovery surfaces a replica).
            if not self._encoders:
                break
            seq = self._admission_q[0]
            items = seq.mm_items  # list[(modality, content)] in token order
            k = len(items)
            if len(self._free_slots) < k:
                break  # backpressure: wait for slots to free up
            self._admission_q.pop(0)
            pend_items: List[_PendingItem] = []
            for item_idx, (modality, content) in enumerate(items):
                slot_id = self._free_slots.pop()
                # Retain ``content`` for possible Phase 8 re-dispatch.
                item = _PendingItem(
                    item_idx=item_idx,
                    modality=modality,
                    slot_id=slot_id,
                    content=content,
                )
                self._send_job(seq.seq_id, item)  # per-item round-robin
                pend_items.append(item)
            n_sentinels = sum(
                1
                for tid in seq.token_ids
                if tid == self.image_token_id or tid == self.video_token_id
            )
            if n_sentinels != k:
                # Skeleton sentinels must equal the mm-item count (they come
                # from the same messages). A skew means encode_skeleton and
                # extract_mm_items_ordered disagreed -- admission would later
                # IndexError; ``meta_complete`` stays False so the seq stalls
                # out via the watchdog into a client retry instead.
                logger.error(
                    f"[lm-disagg {self.lm_id}] seq={seq.seq_id} sentinel/item "
                    f"skew: skeleton has {n_sentinels} image/video sentinels "
                    f"but {k} mm items; seq will not be admitted (client retry)"
                )
            ps = _PendingSeq(
                seq=seq, items=pend_items, dispatched=True, n_sentinels=n_sentinels
            )
            self._pending[seq.seq_id] = ps
            routing = ", ".join(
                f"item{it.item_idx}->{it.encoder_identity}" for it in pend_items
            )
            logger.debug(
                f"[lm-disagg {self.lm_id}] dispatched seq={seq.seq_id} K={k} "
                f"slots={[it.slot_id for it in pend_items]} "
                f"routing=[{routing}] across {len(self._encoders)} encoder(s)"
            )
            # Drop the raw mm payload off the *sequence* now that it's on its way:
            # the LM/KV path never holds pixels (design §3.1 / §4.4). The
            # coordinator keeps a transient per-item copy only for re-dispatch.
            seq.mm_items = None
            seq.mm_contents = None
            self._apply_orphans(ps)

    def _apply_orphans(self, ps: _PendingSeq) -> None:
        if not self._orphan_meta:
            return
        still: List[MmItemMeta] = []
        for m in self._orphan_meta:
            if m.seq_id == ps.seq.seq_id:
                self._apply_meta(m)
            else:
                still.append(m)
        self._orphan_meta = still

    # ------------------------------------------------------------------
    # Control-plane drains
    # ------------------------------------------------------------------
    def _drain_meta(self) -> None:
        while True:
            try:
                raw = self.meta_sock.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            self._apply_meta(pickle.loads(raw))

    def _apply_meta(self, meta: MmItemMeta) -> None:
        ps = self._pending.get(meta.seq_id)
        if ps is None:
            # Meta that arrived before its seq was registered (rare race) -- or
            # for a seq that was aborted/never dispatched. Keep a bounded FIFO so
            # a steady trickle of stale metas can't grow unbounded over a long
            # uptime; the matching seq (if any) is dispatched within a few polls.
            self._orphan_meta.append(meta)
            if len(self._orphan_meta) > self._max_orphan_meta:
                dropped = self._orphan_meta.pop(0)
                logger.warning(
                    f"[lm-disagg {self.lm_id}] orphan-meta buffer full "
                    f"({self._max_orphan_meta}); dropping oldest "
                    f"(seq={dropped.seq_id} item={dropped.item_idx})"
                )
            return
        # Defensive bounds check: a buggy/corrupted encoder peer could send an
        # item_idx outside this seq's skeleton-ordered items. Drop it rather
        # than let an IndexError tear down the shared PP0 disagg loop.
        if not 0 <= meta.item_idx < len(ps.items):
            logger.error(
                f"[lm-disagg {self.lm_id}] dropping meta with out-of-range "
                f"item_idx={meta.item_idx} for seq={meta.seq_id} "
                f"(have {len(ps.items)} items)"
            )
            return
        item = ps.items[meta.item_idx]
        if item.meta is not None:
            # Idempotent (design §5.5.3): a re-dispatched (or duplicated) job
            # re-sends meta. The processor_config_hash gate (§5.4.4) guarantees
            # any replica yields the same content_hash/num_tokens; verify and
            # drop the duplicate. A mismatch means inconsistent processor config
            # across replicas -- fatal, since token positions are already fixed.
            if (
                item.meta.content_hash != meta.content_hash
                or item.meta.num_tokens != meta.num_tokens
            ):
                logger.error(
                    f"[lm-disagg {self.lm_id}] FATAL inconsistent duplicate meta "
                    f"seq={meta.seq_id} item={meta.item_idx}: "
                    f"hash {item.meta.content_hash}!={meta.content_hash} or "
                    f"num_tokens {item.meta.num_tokens}!={meta.num_tokens} "
                    f"(processor config mismatch across encoder replicas)"
                )
            return
        item.meta = meta

    def _drain_notifs(self) -> None:
        """Mark items whose (single, TP0) completion notification has arrived.

        The encoder multi-writes every rank's slot pool and then sends ONE notif
        to TP0, so receipt here implies *all* ranks' writes have landed. The
        actual clone into each rank's model runner happens when the resulting
        ``EMB_READY`` event is applied (per rank, from its own pool).
        """
        notifs = self.nixl.poll_notifs()
        for _agent, msgs in notifs.items():
            for msg in msgs:
                parsed = parse_emb_notif(msg)
                if parsed is None:
                    continue
                seq_id, item_idx = parsed
                ps = self._pending.get(seq_id)
                if ps is None:
                    continue
                # Ignore stray/malformed notifs whose item_idx is out of range.
                if not 0 <= item_idx < len(ps.items):
                    continue
                item = ps.items[item_idx]
                if item.embedding_ready:
                    # Idempotent: duplicate notification from a re-dispatched (or
                    # zombie) write. The emit guard (``slot_freed``) prevents a
                    # second EMB_READY event.
                    continue
                item.embedding_ready = True

    # ------------------------------------------------------------------
    # Poll: admit meta-complete seqs; emit ready embeddings; reap done ones
    # ------------------------------------------------------------------
    def _check_watchdog(self, events: DisaggEvents) -> None:
        """Re-dispatch in-flight items whose encoder crashed/went silent.

        Triggers (design §5.5.2): the item's encoder left the pool (orphaned,
        ``encoder_identity is None``) or no meta/embedding landed within
        ``redispatch_timeout_s``. Re-dispatch reuses the same slot and avoids the
        suspect replica when another is available.

        An item that exhausts ``max_redispatch_attempts`` is *unrecoverable*: its
        whole seq is aborted so the reserved NIXL slots (and, if admitted, the
        scheduler page / SSM slot) are reclaimed instead of being leaked forever
        -- otherwise one bad item would permanently shrink the slot pool and
        eventually wedge all mm traffic. The client request fails (-> retry)."""
        if not self._encoders:
            return  # nowhere to re-dispatch to; hold until a replica appears
        now = time.monotonic()
        give_up_seqs: List[int] = []
        for seq_id, ps in self._pending.items():
            for item in ps.items:
                if item.done:
                    continue
                orphaned = item.encoder_identity is None
                timed_out = (now - item.dispatched_at) > self.redispatch_timeout_s
                if not (orphaned or timed_out):
                    continue
                if item.attempts >= self.max_redispatch_attempts:
                    if not item.gave_up:
                        item.gave_up = True
                        logger.error(
                            f"[lm-disagg {self.lm_id}] seq={seq_id} "
                            f"item={item.item_idx} unrecoverable after "
                            f"{item.attempts} dispatch attempts; aborting seq "
                            f"(reclaiming slots; request -> client retry)"
                        )
                        give_up_seqs.append(seq_id)
                    continue
                reason = "orphaned" if orphaned else "timeout"
                if self._send_job(seq_id, item, avoid=item.encoder_identity):
                    logger.warning(
                        f"[lm-disagg {self.lm_id}] re-dispatched seq={seq_id} "
                        f"item={item.item_idx} ({reason}, attempt "
                        f"{item.attempts}) -> {item.encoder_identity} "
                        f"(slot {item.slot_id} reused)"
                    )
        # Reclaim outside the iteration (``_abort_pending`` mutates ``_pending``).
        # ``give_up_seqs`` is de-duped: multiple failed items in one seq abort it
        # once (the second pop is a no-op).
        for seq_id in dict.fromkeys(give_up_seqs):
            self._abort_pending(seq_id, events)

    def _abort_pending(
        self, seq_id: int, events: Optional[DisaggEvents] = None
    ) -> None:
        """Drop a pending seq, reclaim its NIXL slots, and (if it was already
        admitted) fan out a scheduler abort.

        Used by three paths: admission failure (never admitted -> slot
        reclamation only), the watchdog give-up (may be admitted -> also abort
        in every column's scheduler), and an external client abort
        (:meth:`abort`). An *admitted* seq lives in every column's scheduler, so
        its page / SSM slot / disagg state are reclaimed by the fanned-out
        ``events.aborts`` (each rank's ``add_abort_ids`` -> ``model_runner.free``);
        the NIXL receive slots are reclaimed here on TP0. Freed slots may unblock
        queued admissions. Idempotent.
        """
        ps = self._pending.pop(seq_id, None)
        if ps is None:
            return
        for it in ps.items:
            if not it.slot_freed:
                self._free_slots.append(it.slot_id)
                it.slot_freed = True
        if ps.admitted and events is not None:
            events.aborts.append(seq_id)
        self._try_dispatch()

    def abort(self, seq_ids) -> None:
        """Reclaim coordinator-held NIXL slots for externally aborted seqs.

        Called on TP0 from the worker when client aborts arrive (e.g. request
        cancelled / disconnected). The scheduler-side teardown for *admitted*
        seqs is already handled by the worker's own ``add_abort_ids`` fan-out, so
        this only reclaims the reserved receive slots + drops coordinator
        tracking for seqs still pending here. No-op for unknown ids.
        """
        for seq_id in seq_ids:
            if seq_id in self._pending:
                # No events: admitted seqs are torn down in the scheduler by the
                # worker's existing abort_ids path; we only free NIXL slots.
                self._abort_pending(seq_id)

    def poll(self) -> DisaggEvents:
        """Drive the control plane one step; return this iteration's events."""
        events = DisaggEvents()
        self._drain_discovery()
        self._try_dispatch()
        self._drain_meta()
        self._drain_notifs()
        self._check_watchdog(events)

        freed_any = False
        for seq_id, ps in list(self._pending.items()):
            admit_ready = ps.meta_complete and (
                self.overlap or ps.all_embeddings_ready
            )
            if not ps.admitted and admit_ready:
                _arrived = sum(1 for it in ps.items if it.embedding_ready)
                logger.debug(
                    f"[lm-disagg {self.lm_id}] admit seq={seq_id} "
                    f"(meta complete for {ps.num_items} items; "
                    f"embeddings arrived {_arrived}/{ps.num_items} at admit; "
                    f"overlap={'on' if self.overlap else 'off'})"
                )
                try:
                    self._admit_meta_complete(ps, events)  # sets ps.admitted
                except Exception:
                    # A single malformed seq must never take down the PP0 poll
                    # loop (and with it every other in-flight request). Log,
                    # reclaim its resources, and let the client time out/retry.
                    logger.exception(
                        f"[lm-disagg {self.lm_id}] admit failed for seq={seq_id} "
                        f"(n_sentinels={ps.n_sentinels}, items={len(ps.items)}, "
                        f"skeleton_len={len(ps.seq.token_ids)}); aborting seq"
                    )
                    self._abort_pending(seq_id, events)
                    continue
            if ps.admitted:
                # Emit EMB_READY for items whose write has landed but whose event
                # hasn't been generated yet (covers both notif-arrival and the
                # overlap-off case where all embeddings landed before admit).
                self._emit_ready(ps, events)
            if ps.admitted and all(it.slot_freed for it in ps.items):
                # All embeddings emitted + slots freed; the seq now lives
                # entirely in the scheduler / model runner of every column.
                logger.debug(
                    f"[lm-disagg {self.lm_id}] seq={seq_id} all {ps.num_items} "
                    f"embeddings emitted; releasing tracking"
                )
                del self._pending[seq_id]
                freed_any = True
        if events.admits or freed_any:
            # Freed slots may unblock queued admissions.
            self._try_dispatch()
        return events

    def _emit_ready(self, ps: _PendingSeq, events: DisaggEvents) -> None:
        """Emit EMB_READY events for newly-landed item embeddings (gate B)."""
        if ps.ordered is None:
            return
        for o, it in enumerate(ps.ordered):
            if not it.embedding_ready or it.slot_freed:
                continue
            events.emb_ready.append((ps.seq.seq_id, o, it.slot_id, it.meta.num_tokens))
            self._free_slots.append(it.slot_id)
            it.slot_freed = True
            it.content = None  # done: drop retained payload (no re-dispatch)

    # ------------------------------------------------------------------
    # Admit: build expanded token-ids + grids + DisaggSeqState (== monolith)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _admit_meta_complete(self, ps: _PendingSeq, events: DisaggEvents) -> None:
        seq = ps.seq
        items = ps.items  # token order

        # 1) Expand skeleton sentinels -> num_tokens placeholder ids, recording
        #    each item's [start, end) span in the expanded sequence.
        skeleton = seq.token_ids
        expanded: List[int] = []
        token_span: Dict[int, tuple] = {}
        item_cursor = 0
        for tid in skeleton:
            if tid == self.image_token_id or tid == self.video_token_id:
                n = items[item_cursor].meta.num_tokens
                start = len(expanded)
                expanded.extend([tid] * n)
                token_span[item_cursor] = (start, start + n)
                item_cursor += 1
            else:
                expanded.append(tid)
        assert item_cursor == len(items), (
            f"skeleton had {item_cursor} sentinels != {len(items)} items"
        )

        # 2) Group items into (image-then-video) order = embed_multimodal order.
        img = [it for it in items if it.modality == "image"]
        vid = [it for it in items if it.modality == "video"]
        ordered = img + vid
        ps.ordered = ordered

        # 3) Grids (CPU long tensors) for m-rope, in image/video groups.
        def _grid_tensor(group: List[_PendingItem]) -> Optional[torch.Tensor]:
            if not group:
                return None
            return torch.tensor(
                [list(it.meta.grid_thw) for it in group],
                dtype=torch.long,
                device="cpu",
            )

        image_grid_thw = _grid_tensor(img)
        video_grid_thw = _grid_tensor(vid)

        # 4) is_multimodal mask + content-hash splice (== monolith hash_token_ids).
        input_ids_cpu = torch.tensor(expanded, device="cpu")
        placeholder = torch.tensor(
            self.mr.model.get_mm_placeholder_token_ids(), device="cpu"
        )
        is_multimodal_cpu = torch.isin(input_ids_cpu, placeholder)
        item_hashes = [it.meta.content_hash for it in ordered]
        seq.hash_token_ids = ModelRunner._splice_mm_pad_ids(
            expanded, is_multimodal_cpu, item_hashes
        )

        # 5) m-rope positions (same call the monolith makes); valid now that all
        #    grids are known, independent of embedding readiness.
        prompt_positions, mrope_position_delta = MRotaryEmbedding.get_input_positions(
            input_tokens=expanded,
            hf_config=self.mr.model.config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=None,
        )

        # 6) Rewrite the seq into a fully-formed prefill request (positions
        #    fixed); embeddings are NOT on the seq -- they live in the model
        #    runner's disagg_embeds (immune to chunked-prefill deepcopy).
        seq.token_ids = expanded
        # VL placeholder expansion redefines the *original* prompt, so reset
        # both the raw length and the dynamic prefill boundary together.
        seq.raw_prompt_len = len(expanded)
        seq.prompt_len = len(expanded)
        seq.cur_length = seq.prompt_len
        seq._mm_precomputed = None

        # 7) Build the gate state (ordered = image-then-video). Spans for ordered
        #    items map back through their token-order ``item_idx``. Emit an ADMIT
        #    event; every column registers its own copy + adds the seq to its
        #    scheduler (the broadcast pickles a fresh state per rank).
        prompt_len = len(expanded)
        state = DisaggSeqState(
            num_items=len(ordered),
            item_span=[token_span[it.item_idx] for it in ordered],
            item_modality=[it.modality for it in ordered],
            item_ready=[False] * len(ordered),
            item_embed=[None] * len(ordered),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_ids_cpu=input_ids_cpu,
            is_multimodal_cpu=is_multimodal_cpu,
            prompt_positions=prompt_positions,
            mrope_position_delta=mrope_position_delta,
            prompt_len=prompt_len,
        )

        # 8) Mark admitted *before* emitting ready so the emit path runs for
        #    embeddings that landed before admission -- critical when admission
        #    waits for all embeddings (overlap off), where no further notif would
        #    re-trigger the emit.
        ps.admitted = True
        events.admits.append((seq, state))
