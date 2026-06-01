"""LM PP0 side of encoder disaggregation: slot pool + per-item aggregator.

Lives in the LM PP0 worker process (one per LM node; TP>1 is Phase 5). It:

1. Owns the **NIXL receive slot pool** -- a single persistent registered GPU
   tensor ``[num_slots, max_vis_tokens, feat_dim]`` carved into per-item slots
   the encoder WRITEs into (design §5.3).
2. **Dispatches** one :class:`EncoderJob` per mm item to the encoder over ZMQ,
   handing it the reserved slot's :class:`RemoteRegion`.
3. **Aggregates** the per-item control plane: :class:`MmItemMeta` (ZMQ) gives
   ``num_tokens``/``grid``/``content_hash`` so the skeleton token-ids can be
   expanded and the prefix-cache key built; the NIXL notification gives the
   "embedding bytes landed" gate (design §5.4 / §6.2).
4. **Admits** a request to the scheduler as soon as *all* per-item meta have
   arrived (design §6.2 gate A: token positions + prefix-cache hashes are then
   determined). The expanded ``token_ids``/``hash_token_ids``/grids/mrope are
   built and a :class:`DisaggSeqState` is registered in the model runner with
   per-item readiness + (initially empty) embeddings.
5. As each item's NIXL write lands, the embedding is cloned out of the slot
   pool into ``model_runner.disagg_embeds`` and ``embedding_ready[i]`` is set
   (design §6.2 gate B). The scheduler prefills the text prefix + already-ready
   image spans while later items' ViT is still in flight -- intra-request
   encode/prefill overlap. The slot is returned to the free pool per item.

Output stays byte-identical to the monolith: the only change vs. Phase 3b is
*when* prefill is allowed to advance, not *what* is computed.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    # Set once the embedding has been cloned out of the slot pool into the
    # model runner's ``disagg_embeds`` and the slot returned to the free list.
    slot_freed: bool = False
    # Phase 8 failure handling (design §5.5.2): the raw item content is retained
    # so the PP0 watchdog can RE-DISPATCH the EncoderJob to another replica if
    # the original encoder crashes / goes silent. Dropped once the item is done.
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
    # Phase 6 overlap: a seq is *admitted* to the scheduler once all per-item
    # meta arrive (positions/hashes determined; gate A); its embeddings then
    # stream in progressively (gate B). ``ordered`` is the image-then-video
    # ordering of ``items`` (== ``embed_multimodal`` tuple order and the order
    # of :class:`DisaggSeqState`), fixed at admission.
    admitted: bool = False
    ordered: Optional[List[_PendingItem]] = None

    @property
    def num_items(self) -> int:
        return len(self.items)

    @property
    def meta_complete(self) -> bool:
        return all(it.meta is not None for it in self.items)

    @property
    def all_embeddings_ready(self) -> bool:
        return all(it.embedding_ready for it in self.items)


class LMDisaggManager:
    def __init__(
        self,
        model_runner: ModelRunner,
        lm_id: str,
        discovery_endpoint: str,
        *,
        discovery_mode: str = "network",
        processor_config_hash: str = "",
        advertise_host: str = "127.0.0.1",
        meta_bind: str = "tcp://0.0.0.0:0",
        num_slots: int = 32,
        max_vis_tokens: int = 16384,
        nixl_backend: str = "UCX",
        encoder_dp: int = 1,
    ):
        self.mr = model_runner
        self.lm_id = lm_id
        self.discovery_endpoint = discovery_endpoint
        self.discovery_mode = discovery_mode
        self.processor_config_hash = processor_config_hash
        self.advertise_host = advertise_host
        self.meta_bind = meta_bind
        self.num_slots = num_slots
        self.max_vis_tokens = max_vis_tokens
        self.nixl_backend = nixl_backend
        self.encoder_dp = max(1, int(encoder_dp))
        # Phase 6 intra-request encode/prefill overlap (design §6.2). When on
        # (default), a seq is admitted as soon as all meta arrive and prefill
        # advances per-item under the two-layer gate. When off, admission waits
        # for *all* embeddings (Phase 3b timing) -> single-chunk prefill, which
        # is byte-identical to the unchunked monolith (used as a determinism
        # baseline, since chunked prefill is not itself bit-exact w.r.t. chunk
        # boundaries in this engine).
        self.overlap = os.environ.get("GLLM_DISAGG_OVERLAP", "1") != "0"
        # Phase 8 watchdog (design §5.5.2, §8 row 8): an in-flight item whose
        # encoder has gone silent (no meta/embedding within the timeout) or whose
        # encoder left the pool (orphaned) is re-dispatched to a live replica,
        # reusing the same slot (idempotent overwrite, §5.5.3). After
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
        self.feat_dim = int(
            cfg.vision_config.out_hidden_size
            * (1 + len(getattr(cfg.vision_config, "deepstack_visual_indexes", [])))
        )
        self.dtype = self.mr.model_loader.dtype
        self.device = next(self.mr.model.parameters()).device

        # ZMQ / NIXL state (built in setup()).
        import zmq

        self._zmq = zmq
        self.zmq_ctx = None
        self.meta_sock = None  # PULL: MmItemMeta in
        self.nixl: Optional[NixlEndpoint] = None
        self.meta_addr: str = ""

        # Discovery + dynamic encoder registry (identity -> _EncoderConn).
        self.disc = None
        self._encoders: Dict[str, _EncoderConn] = {}
        self._rr_encoder_idx = 0
        # The PP0 overlap loop spins at full rate when idle; throttle the
        # (networked) discovery poll so it doesn't flood the registry with
        # thousands of list RPCs/sec. 200ms is well within connect latency SLO.
        self._disc_poll_interval_s = 0.2
        self._last_disc_poll = 0.0

        # Slot pool.
        self.slot_pool: Optional[torch.Tensor] = None
        self.slot_reg = None
        self.pool_region: Optional[RemoteRegion] = None
        self.slot_stride_bytes = 0
        self._free_slots: List[int] = []

        # Bookkeeping.
        self._pending: Dict[int, _PendingSeq] = {}  # seq_id -> _PendingSeq
        self._admission_q: List[object] = []  # seqs waiting for free slots
        # meta/notif that arrived before the seq was registered (rare race).
        self._orphan_meta: List[MmItemMeta] = []

    # ------------------------------------------------------------------
    def setup(self) -> None:
        self.zmq_ctx = self._zmq.Context.instance()
        self.meta_sock = self.zmq_ctx.socket(self._zmq.PULL)
        self.meta_sock.bind(self.meta_bind)
        bound = self.meta_sock.getsockopt(self._zmq.LAST_ENDPOINT).decode()
        port = bound.rsplit(":", 1)[-1]
        self.meta_addr = f"tcp://{self.advertise_host}:{port}"

        self.nixl = NixlEndpoint(
            name=f"lm-{self.lm_id}", backends=(self.nixl_backend,)
        )
        # Zero-init (not ``empty``): a slot is read only after its NIXL write
        # notif, but zero-init makes any accidental early read fail loudly
        # (all-zero embedding) instead of leaking uninitialized garbage.
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
        self._free_slots = list(range(self.num_slots))

        # Publish self + watch encoders. We do NOT block on encoders: text-only
        # requests serve immediately, and mm requests queue in the admission
        # queue until a replica connects (any start order, design §7.3.4).
        self.disc = make_discovery(self.discovery_mode, self.discovery_endpoint)
        self.disc.publish(
            "lm",
            self.lm_id,
            make_payload(
                role="lm",
                agent_name=self.nixl.name,
                nixl_meta=self.nixl.local_meta(),
                zmq_addr=self.meta_addr,
                feat_dim=self.feat_dim,
                processor_config_hash=self.processor_config_hash,
            ),
        )
        logger.info(
            f"[lm-disagg {self.lm_id}] meta intake at {self.meta_addr}; slot pool "
            f"{self.num_slots}x{self.max_vis_tokens}x{self.feat_dim} ({self.dtype}); "
            f"watching encoders via {self.discovery_mode}:{self.discovery_endpoint}"
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
        for ev in self.disc.poll_events("encoder"):
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
        # departed replica so the watchdog re-dispatches it immediately (rather
        # than waiting out the full timeout) once another replica is live.
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
        logger.info(
            f"[lm-disagg {self.lm_id}] submit seq={seq.seq_id} "
            f"items={len(seq.mm_items)} (live encoders={len(self._encoders)}, "
            f"free slots={len(self._free_slots)})"
        )

    def _slot_region(self, slot_id: int) -> RemoteRegion:
        return self.pool_region.with_offset(
            slot_id * self.slot_stride_bytes, self.slot_stride_bytes
        )

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

        Re-dispatch reuses the same ``slot_id`` (same registered NIXL region):
        the encoder overwrites it idempotently (design §5.5.3). Returns False if
        no encoder is currently live (caller retries on a later poll)."""
        conn = self._pick_encoder(avoid=avoid)
        if conn is None:
            return False
        job = EncoderJob(
            seq_id=seq_id,
            item_idx=item.item_idx,
            modality=item.modality,
            content=item.content,
            remote_slot=self._slot_region(item.slot_id),
            slot_id=item.slot_id,
            lm_meta_addr=self.meta_addr,
            lm_agent_name=self.nixl.name,
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
            ps = _PendingSeq(seq=seq, items=pend_items, dispatched=True)
            self._pending[seq.seq_id] = ps
            routing = ", ".join(
                f"item{it.item_idx}->{it.encoder_identity}" for it in pend_items
            )
            logger.info(
                f"[lm-disagg {self.lm_id}] dispatched seq={seq.seq_id} K={k} "
                f"slots={[it.slot_id for it in pend_items]} "
                f"routing=[{routing}] across {len(self._encoders)} encoder(s)"
            )
            # Drop the raw mm payload off the *sequence* now that it's on its way:
            # the LM/KV path never holds pixels (design §3.1 / §4.4). The manager
            # keeps a transient per-item copy only for re-dispatch resilience.
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
            self._orphan_meta.append(meta)
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
        notifs = self.nixl.poll_notifs()
        touched: set = set()
        for _agent, msgs in notifs.items():
            for msg in msgs:
                parsed = parse_emb_notif(msg)
                if parsed is None:
                    continue
                seq_id, item_idx = parsed
                ps = self._pending.get(seq_id)
                if ps is None:
                    continue
                item = ps.items[item_idx]
                if item.embedding_ready:
                    # Idempotent: duplicate notification from a re-dispatched (or
                    # zombie) write. The first one already marked it ready; the
                    # slot-free guard in _flush_ready_clones prevents re-cloning.
                    continue
                item.embedding_ready = True
                touched.add(seq_id)
        # Clone freshly-landed embeddings into the model runner (gate B). For a
        # seq not yet admitted (meta still incomplete) the clones are deferred
        # to ``_admit_meta_complete``; the slots stay reserved until then.
        for seq_id in touched:
            self._flush_ready_clones(self._pending[seq_id])

    @torch.inference_mode()
    def _flush_ready_clones(self, ps: _PendingSeq) -> None:
        """Clone ready item embeddings slot-pool -> ``disagg_embeds`` (gate B).

        Only runs for admitted seqs (``disagg_embeds`` row exists). The NIXL
        notification only signals the WRITE completed; we must synchronize this
        device before reading so the written bytes are visible to the clone
        (mirrors ``tests/test_nixl_transfer.py``'s post-notif sync). One sync
        per batch of clones is sufficient.
        """
        if not ps.admitted or ps.ordered is None:
            return
        todo = [
            (o, it)
            for o, it in enumerate(ps.ordered)
            if it.embedding_ready and not it.slot_freed
        ]
        if not todo:
            return
        torch.cuda.synchronize(self.device)
        for o, it in todo:
            emb = self.slot_pool[it.slot_id, : it.meta.num_tokens, :].clone()
            self.mr.disagg_set_embedding(ps.seq.seq_id, o, emb)
            self._free_slots.append(it.slot_id)
            it.slot_freed = True
            it.content = None  # done: drop retained payload (no re-dispatch)

    # ------------------------------------------------------------------
    # Poll: admit meta-complete seqs; reap fully-embedded ones
    # ------------------------------------------------------------------
    def _check_watchdog(self) -> None:
        """Re-dispatch in-flight items whose encoder crashed/went silent.

        Triggers (design §5.5.2): the item's encoder left the pool (orphaned,
        ``encoder_identity is None``) or no meta/embedding landed within
        ``redispatch_timeout_s``. Re-dispatch reuses the same slot and avoids the
        suspect replica when another is available. Items not yet ``done`` and not
        yet exhausted are retried; the slot stays reserved throughout."""
        if not self._encoders:
            return  # nowhere to re-dispatch to; hold until a replica appears
        now = time.monotonic()
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
                            f"{item.attempts} dispatch attempts; giving up "
                            f"(request will time out -> client retry)"
                        )
                    continue
                reason = "orphaned" if orphaned else "timeout"
                if self._send_job(seq_id, item, avoid=item.encoder_identity):
                    logger.warning(
                        f"[lm-disagg {self.lm_id}] re-dispatched seq={seq_id} "
                        f"item={item.item_idx} ({reason}, attempt "
                        f"{item.attempts}) -> {item.encoder_identity} "
                        f"(slot {item.slot_id} reused)"
                    )

    def poll(self) -> List[object]:
        self._drain_discovery()
        self._try_dispatch()
        self._drain_meta()
        self._drain_notifs()
        self._check_watchdog()

        admitted: List[object] = []
        freed_any = False
        for seq_id, ps in list(self._pending.items()):
            admit_ready = ps.meta_complete and (
                self.overlap or ps.all_embeddings_ready
            )
            if not ps.admitted and admit_ready:
                _arrived = sum(1 for it in ps.items if it.embedding_ready)
                logger.info(
                    f"[lm-disagg {self.lm_id}] admit seq={seq_id} "
                    f"(meta complete for {ps.num_items} items; "
                    f"embeddings arrived {_arrived}/{ps.num_items} at admit; "
                    f"overlap={'on' if self.overlap else 'off'})"
                )
                self._admit_meta_complete(ps)  # sets ps.admitted
                admitted.append(ps.seq)
            if ps.admitted and ps.all_embeddings_ready:
                # All embeddings cloned out + slots freed; the seq now lives
                # entirely in the scheduler / model runner.
                logger.info(
                    f"[lm-disagg {self.lm_id}] seq={seq_id} all {ps.num_items} "
                    f"embeddings landed; releasing tracking"
                )
                del self._pending[seq_id]
                freed_any = True
        if admitted or freed_any:
            # Freed slots may unblock queued admissions.
            self._try_dispatch()
        return admitted

    # ------------------------------------------------------------------
    # Admit: build expanded token-ids + grids + register gate state (== monolith)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _admit_meta_complete(self, ps: _PendingSeq) -> None:
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
        seq.prompt_len = len(expanded)
        seq.cur_length = seq.prompt_len
        seq._mm_precomputed = None

        # 7) Register gate state (ordered = image-then-video). Spans for ordered
        #    items map back through their token-order ``item_idx``.
        prompt_len = len(expanded)
        self.mr.disagg_register(
            seq.seq_id,
            DisaggSeqState(
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
            ),
        )

        # 8) Mark admitted *before* flushing so the clone path (gated on
        #    ``ps.admitted``) runs for embeddings that landed before admission
        #    -- critical when admission waits for all embeddings (overlap off),
        #    where no further notif would re-trigger the flush.
        ps.admitted = True
        self._flush_ready_clones(ps)
