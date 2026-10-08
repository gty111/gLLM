"""Encoder-side serving loop: ZMQ EncoderJob intake -> ViT -> NIXL write.

Coordinates a :class:`gllm.runtime.vision_encoder_runner.VisionEncoderRunner`
with the disaggregation control and data planes. Items flow through the
cross-request pipeline in :mod:`gllm.engine.encoder_pipeline`, one segment
(a video time slice, or a whole image) at a time:

    EncoderJob(seq, item, modality, content, remote_slots)
      plan:    grid / token count / hash (no decoding)
               push MmItemMeta(num_tokens, grid, hash) --> LM TP0 (gate A)
      decode:  segment frames -> preprocess            (NVDEC / CPU workers)
      encode:  ViT(segment) -> staging rows of send_buf (GPU thread)
      send:    nixl.write(rows -> slot rows) for every LM TP rank, then
               nixl.notify(TP0, "embp:seq:item:rows" ... "emb:seq:item")

Sending the meta before any decoding lets the LM expand its skeleton
token-ids and build the prefix-cache key while decoding, ViT and transfer are
still in flight; partial notifications let it prefill the landed prefix.
All ZMQ and NIXL calls stay on the main (serve) thread.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import List, Optional

import torch
import zmq
from logger import logger

from gllm.disagg.discovery import (
    make_discovery,
    make_payload,
    payload_nixl_metas,
)
from gllm.disagg.protocol import (
    EncoderJob,
    MmItemMeta,
    emb_notif,
    emb_partial_notif,
)
from gllm.engine.encoder_pipeline import EncoderPipeline
from gllm.runtime.vision_encoder_runner import VisionEncoderRunner
from gllm.transfer.nixl_transfer import NixlEndpoint


class Encoder:
    """High-level service for encoder-disaggregated multimodal inference."""

    def __init__(
        self,
        runner: VisionEncoderRunner,
        encoder_id: str,
        discovery_endpoint: str,
        *,
        processor_config_hash: str = "",
        advertise_host: str = "127.0.0.1",
        job_bind: str = "tcp://0.0.0.0:0",
        max_vis_tokens: int = 16384,
        nixl_backend: str = "UCX",
    ):
        self.runner = runner
        self.encoder_id = encoder_id
        self.discovery_endpoint = discovery_endpoint
        self.processor_config_hash = processor_config_hash
        self.advertise_host = advertise_host
        self.job_bind = job_bind
        self.max_vis_tokens = max_vis_tokens
        self.nixl_backend = nixl_backend

        self.feat_dim = int(
            runner.model_loader.config.vision_config.out_hidden_size
            * (
                1
                + len(
                    getattr(
                        runner.model_loader.config.vision_config,
                        "deepstack_visual_indexes",
                        [],
                    )
                )
            )
        )

        self.zmq_ctx: Optional[zmq.Context] = None
        self.job_sock: Optional[zmq.Socket] = None  # PULL: jobs in
        self.meta_sock: Optional[zmq.Socket] = None  # PUSH: meta out -> LM
        self.nixl: Optional[NixlEndpoint] = None
        self.send_buf: Optional[torch.Tensor] = None
        self.send_reg = None
        self.disc = None
        # Current LM connection (single LM node, but one NIXL agent per LM TP
        # rank under tensor parallelism). ``lm_agent_names`` is rank order
        # (index 0 == TP0, the meta + notification target); ``lm_zmq_addr`` is
        # the single TP0 meta intake.
        self.lm_identity: Optional[str] = None
        self.lm_agent_names: List[str] = []
        self.lm_zmq_addr: Optional[str] = None
        self.lm_payload: Optional[dict] = None  # last LM payload (for re-handshake)
        # NIXL write resilience: retry a failed write with a
        # re-handshake before giving up, so a transient transport hiccup (e.g.
        # mid-wireup REMOTE_DISCONNECT) does not kill the whole replica.
        self.write_max_attempts = 3
        # Test-only fault injection (Phase 8 watchdog validation): silently drop
        # the first N received jobs *before* processing, simulating an encoder
        # that took the job then crashed/hung. The LM watchdog must re-dispatch.
        # Off by default; set GLLM_ENC_FAIL_FIRST_N=k to enable.
        self._fail_first_n = int(os.environ.get("GLLM_ENC_FAIL_FIRST_N", "0"))
        self._jobs_seen = 0

    # ------------------------------------------------------------------
    def setup(self) -> None:
        self.zmq_ctx = zmq.Context.instance()
        self.job_sock = self.zmq_ctx.socket(zmq.PULL)
        self.job_sock.bind(self.job_bind)
        bound = self.job_sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        port = bound.rsplit(":", 1)[-1]
        job_addr = f"tcp://{self.advertise_host}:{port}"

        # NIXL endpoint: persistent registered send buffer (bf16, encoder GPU).
        self.nixl = NixlEndpoint(
            name=f"encoder-{self.encoder_id}", backends=(self.nixl_backend,)
        )
        self.send_buf = torch.empty(
            (self.max_vis_tokens, self.feat_dim),
            dtype=self.runner.dtype,
            device="cuda",
        )
        self.send_reg = self.nixl.register(self.send_buf)

        # Publish self into the registry + start watching for the LM. We do NOT
        # block here: the LM may come up later (any start order).
        # The serve loop drains discovery events and (re)connects dynamically.
        self.disc = make_discovery(self.discovery_endpoint)
        self.disc.publish(
            "encoder",
            self.encoder_id,
            make_payload(
                role="encoder",
                agent_name=self.nixl.name,
                nixl_meta=self.nixl.local_meta(),
                zmq_addr=job_addr,
                feat_dim=self.feat_dim,
                processor_config_hash=self.processor_config_hash,
            ),
        )
        logger.info(
            f"[encoder {self.encoder_id}] job intake at {job_addr}; "
            f"watching for LM via {self.discovery_endpoint}"
        )
        self._drain_discovery()

    # ------------------------------------------------------------------
    def _drain_discovery(self) -> None:
        """Apply ADD/UPDATE/REMOVE for the LM peer (single LM in Phase 4)."""
        try:
            evs = self.disc.poll_events("lm")
        except Exception as e:
            # A transient registry stall must not kill the encoder serve loop;
            # skip this poll and retry on the next iteration.
            logger.warning(
                f"[encoder {self.encoder_id}] discovery poll failed "
                f"({type(e).__name__}: {e}); retrying next iteration"
            )
            return
        for ev in evs:
            if ev.kind in ("ADD", "UPDATE"):
                self._connect_lm(ev.identity, ev.payload, reconnect=ev.kind == "UPDATE")
            elif ev.kind == "REMOVE":
                self._disconnect_lm(ev.identity)

    def _connect_lm(self, identity: str, payload: dict, *, reconnect: bool) -> None:
        peer_hash = payload.get("processor_config_hash", "")
        if (
            self.processor_config_hash
            and peer_hash
            and peer_hash != self.processor_config_hash
        ):
            logger.error(
                f"[encoder {self.encoder_id}] REJECT LM {identity}: "
                f"processor_config_hash mismatch ({peer_hash[:12]} != "
                f"{self.processor_config_hash[:12]})"
            )
            return
        if reconnect and self.lm_identity == identity:
            self._disconnect_lm(identity, keep_quiet=True)
        self.lm_identity = identity
        self.lm_payload = payload
        # Connect to every LM TP rank's NIXL agent (rank order); the embedding
        # is multi-written into all of them. ``add_remote_agent`` order matches
        # the slot-region order packed into each EncoderJob.
        self.lm_agent_names = [
            self.nixl.connect(meta) for meta in payload_nixl_metas(payload)
        ]
        if self.meta_sock is not None:
            self.meta_sock.close(0)
        self.meta_sock = self.zmq_ctx.socket(zmq.PUSH)
        self.meta_sock.connect(payload["zmq_addr"])
        self.lm_zmq_addr = payload["zmq_addr"]
        logger.info(
            f"[encoder {self.encoder_id}] connected to LM {identity} "
            f"agents={self.lm_agent_names} (tp={len(self.lm_agent_names)}) "
            f"meta={self.lm_zmq_addr}"
        )

    def _disconnect_lm(self, identity: str, *, keep_quiet: bool = False) -> None:
        if self.lm_identity != identity:
            return
        for name in self.lm_agent_names:
            self.nixl.disconnect(name)
        if self.meta_sock is not None:
            self.meta_sock.close(0)
            self.meta_sock = None
        if not keep_quiet:
            logger.info(f"[encoder {self.encoder_id}] LM {identity} left; idle")
        self.lm_identity = None
        self.lm_agent_names = []
        self.lm_zmq_addr = None

    # ------------------------------------------------------------------
    def _send_meta(self, job: EncoderJob, num_tokens: int, grid_thw, chash: bytes) -> None:
        """Gate A: tell the LM the item's token count / grid / hash."""
        meta = MmItemMeta(
            seq_id=job.seq_id,
            item_idx=job.item_idx,
            modality=job.modality,
            num_tokens=num_tokens,
            feat_dim=self.feat_dim,
            grid_thw=tuple(int(x) for x in grid_thw.flatten().tolist()),
            content_hash=chash,
            slot_id=job.slot_id,
        )
        self.meta_sock.send(pickle.dumps(meta))

    def _reconnect_lm(self) -> None:
        """Drop + re-add the LM remote agent to rebuild a stale UCX endpoint."""
        if self.lm_identity is None or self.lm_payload is None:
            return
        ident, payload = self.lm_identity, self.lm_payload
        try:
            self._disconnect_lm(ident, keep_quiet=True)
        except Exception:
            pass
        self._connect_lm(ident, payload, reconnect=False)

    # ------------------------------------------------------------------
    def _ensure_lm(self, attempts: int = 50, sleep_s: float = 0.1) -> bool:
        """Block briefly until the LM is connected (it must be, to dispatch)."""
        for _ in range(attempts):
            if self.meta_sock is not None and self.lm_agent_names:
                return True
            self._drain_discovery()
            if self.meta_sock is not None and self.lm_agent_names:
                return True
            time.sleep(sleep_s)
        return False

    def _recv_jobs(self) -> List[EncoderJob]:
        """Drain every currently-available job from the intake socket."""
        batch: List[EncoderJob] = []
        while True:
            try:
                raw = self.job_sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return batch
            job: EncoderJob = pickle.loads(raw)
            self._jobs_seen += 1
            if self._jobs_seen <= self._fail_first_n:
                logger.error(
                    f"[encoder {self.encoder_id}] FAULT-INJECT drop job "
                    f"seq={job.seq_id} item={job.item_idx} "
                    f"({self._jobs_seen}/{self._fail_first_n})"
                )
                continue
            if not self._ensure_lm():
                logger.error(
                    f"[encoder {self.encoder_id}] dropping job seq={job.seq_id} "
                    f"item={job.item_idx}: no LM connected"
                )
                continue
            batch.append(job)

    def serve_forever(self) -> None:
        """Main loop of the pipelined encoder: job intake, meta, and every NIXL
        operation stay on this thread; planning, decoding and the ViT run on
        :class:`EncoderPipeline` threads, overlapped across requests."""
        pipe = EncoderPipeline(
            self.runner,
            self.send_buf,
            decode_workers=self.runner.video_loader.num_workers,
        )
        poller = zmq.Poller()
        poller.register(self.job_sock, zmq.POLLIN)
        inflight: List[dict] = []
        last_disc = 0.0
        logger.info(f"[encoder {self.encoder_id}] serving jobs")
        while True:
            now = time.monotonic()
            if now - last_disc > 0.1:
                self._drain_discovery()
                last_disc = now
            timeout_ms = 1 if (inflight or pipe.busy) else 50
            socks = dict(poller.poll(timeout=timeout_ms))
            if self.job_sock in socks:
                for job in self._recv_jobs():
                    pipe.submit(job)

            for w in pipe.poll_planned():
                job = w.job
                if w.num_tokens > self.max_vis_tokens:
                    logger.error(
                        f"[encoder {self.encoder_id}] job seq={job.seq_id} "
                        f"item={job.item_idx} needs {w.num_tokens} vis tokens > "
                        f"max_vis_tokens {self.max_vis_tokens}; dropped"
                    )
                    continue
                try:
                    self._send_meta(job, w.num_tokens, w.grid_thw, w.chash)
                except Exception as e:
                    logger.error(
                        f"[encoder {self.encoder_id}] job seq={job.seq_id} "
                        f"item={job.item_idx} meta send failed: {e}"
                    )
                    continue
                pipe.start(w)

            for w, seg, off, rows in pipe.poll_staged():
                if w.failed:
                    pipe.staging.free(off, rows)
                    continue
                t = {"w": w, "seg": seg, "off": off, "rows": rows, "attempt": 0}
                if self._post_segment(t):
                    inflight.append(t)
                else:
                    self._abandon(pipe, t)

            still: List[dict] = []
            for t in inflight:
                try:
                    done = all(self.nixl.is_done(h) for h in t["handles"])
                except Exception as e:
                    self._release_handles(t)
                    logger.warning(
                        f"[encoder {self.encoder_id}] NIXL write seq="
                        f"{t['w'].job.seq_id} seg={t['seg']} failed: {e}"
                    )
                    self._reconnect_lm()
                    if self._post_segment(t):
                        still.append(t)
                    else:
                        self._abandon(pipe, t)
                    continue
                if not done:
                    still.append(t)
                    continue
                self._release_handles(t)
                w = t["w"]
                pipe.segment_written(w, t["seg"], t["off"], t["rows"])
                self._notify_progress(w)
            inflight = still

    def _post_segment(self, t: dict) -> bool:
        """Post the NIXL writes of one staged segment to every LM TP rank's
        slot (rows ``seg_rows[seg]``). False after ``write_max_attempts``."""
        w, seg = t["w"], t["seg"]
        lo, hi = w.seg_rows[seg]
        row_bytes = self.feat_dim * self.send_buf.element_size()
        src = self.send_buf[t["off"] : t["off"] + t["rows"]]
        while t["attempt"] < self.write_max_attempts:
            t["attempt"] += 1
            try:
                t["handles"] = [
                    self.nixl.write(src, rs.with_offset(lo * row_bytes, (hi - lo) * row_bytes))
                    for rs in w.job.remote_slots
                ]
                return True
            except Exception as e:
                logger.warning(
                    f"[encoder {self.encoder_id}] NIXL post seq={w.job.seq_id} "
                    f"seg={seg} attempt {t['attempt']}/{self.write_max_attempts}: {e}"
                )
                time.sleep(0.1 * t["attempt"])
                self._reconnect_lm()
        return False

    def _release_handles(self, t: dict) -> None:
        for h in t.get("handles", []):
            self.nixl.release(h)
        t["handles"] = []

    def _abandon(self, pipe: "EncoderPipeline", t: dict) -> None:
        w = t["w"]
        pipe.staging.free(t["off"], t["rows"])
        if not w.failed:
            w.failed = True
            logger.error(
                f"[encoder {self.encoder_id}] job seq={w.job.seq_id} "
                f"item={w.job.item_idx} dropped: NIXL write failed; LM watchdog "
                "will re-dispatch"
            )

    def _notify_progress(self, w) -> None:
        """Announce the item's landed rows to LM TP0 as a growing prefix."""
        rows = w.done_prefix_rows()
        if rows <= w.notified_rows or w.failed:
            return
        job = w.job
        target = (
            job.lm_agent_names[0] if job.lm_agent_names else job.remote_slots[0].agent_name
        )
        msg = (
            emb_notif(job.seq_id, job.item_idx)
            if rows == w.num_tokens
            else emb_partial_notif(job.seq_id, job.item_idx, rows)
        )
        self.nixl.notify(target, msg)
        w.notified_rows = rows
