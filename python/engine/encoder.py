"""Encoder-side serving loop: ZMQ EncoderJob intake -> ViT -> NIXL write.

Coordinates a :class:`gllm.runtime.vision_encoder_runner.VisionEncoderRunner`
with the disaggregation control and data planes. Items flow through the
cross-request pipeline in :mod:`gllm.engine.encoder_pipeline`, one segment
(a video time slice, or a whole image) at a time:

    EncoderJob(seq, item, modality, content, LM arena regions)
      plan:    grid / token count / hash (no decoding)
               push MmItemMeta(num_tokens, grid, hash) --> LM TP0 (gate A)
               <-- EmbTarget(LM arena pages for the item's rows)
      decode:  segment frames -> preprocess            (NVDEC / CPU workers)
      encode:  ViT(segment) -> staging pages of send_buf (GPU thread)
      send:    nixl.write(staging pages -> LM pages) for every LM TP rank,
               then nixl.notify(TP0, "embp:seq:item:rows" ... "emb:seq:item")

Sending the meta before any decoding lets the LM expand its skeleton
token-ids and build the prefix-cache key while decoding, ViT and transfer are
still in flight; partial notifications let it prefill the landed prefix.
All ZMQ and NIXL calls stay on the main (serve) thread.
"""

from __future__ import annotations

import os
import pickle
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import zmq
from logger import logger

from gllm.disagg.discovery import (
    make_discovery,
    make_payload,
    payload_nixl_metas,
)
from gllm.disagg.paging import copy_runs, num_pages, rows_per_page
from gllm.disagg.protocol import (
    EmbCancel,
    EmbTarget,
    EncoderJob,
    MmItemMeta,
    emb_fail_notif,
    emb_notif,
    emb_partial_notif,
)
from gllm.engine.encoder_pipeline import EncoderPipeline, PagedStaging
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
        self.staging: Optional[PagedStaging] = None
        self.send_reg = None
        # LM arena pages per (seq_id, item_idx), from EmbTarget messages.
        self._targets: "OrderedDict[Tuple[int, int], List[int]]" = OrderedDict()
        # Items in progress (``None`` while planning) and recently finished:
        # a re-dispatched job for either is a duplicate. Running it again
        # would hold staging pages and could write pages after the LM freed
        # them.
        self._active: Dict[Tuple[int, int], object] = {}
        self._finished: "OrderedDict[Tuple[int, int], None]" = OrderedDict()
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
        # Staged rows wait this long for the LM's page target.
        self.target_timeout_s = 120.0
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

        # NIXL endpoint: persistent registered send buffer (encoder GPU),
        # paged into ~1 MiB staging pages.
        self.nixl = NixlEndpoint(
            name=f"encoder-{self.encoder_id}", backends=(self.nixl_backend,)
        )
        dtype = self.runner.dtype
        per_page = rows_per_page(1 << 20, self.feat_dim, dtype)
        pages = num_pages(self.max_vis_tokens, per_page)
        row_bytes = self.feat_dim * torch.empty((), dtype=dtype).element_size()
        send_buf = torch.empty(pages * per_page * row_bytes, dtype=torch.uint8, device="cuda")
        self.staging = PagedStaging(send_buf, dtype, self.feat_dim, per_page)
        self.send_reg = self.nixl.register(send_buf)

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
        # the region order packed into each EncoderJob.
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
        """Drain the intake socket: jobs are returned, page targets recorded."""
        batch: List[EncoderJob] = []
        while True:
            try:
                raw = self.job_sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return batch
            msg = pickle.loads(raw)
            if isinstance(msg, EmbTarget):
                self._targets[(msg.seq_id, msg.item_idx)] = msg.pages
                # Targets of items this replica never finishes (dropped jobs,
                # re-dispatch elsewhere) must not accumulate.
                while len(self._targets) > 4096:
                    self._targets.popitem(last=False)
                continue
            if isinstance(msg, EmbCancel):
                key = (msg.seq_id, msg.item_idx)
                self._targets.pop(key, None)
                w = self._active.pop(key, None)
                if w is not None:
                    w.failed = True  # drops its staged / queued segments
                continue
            job: EncoderJob = msg
            key = (job.seq_id, job.item_idx)
            w = self._active.get(key, False)
            if key in self._finished or (w is not False and not getattr(w, "failed", False)):
                continue  # duplicate of an item in progress or done
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
            self._active[key] = None
            batch.append(job)

    def serve_forever(self) -> None:
        """Main loop of the pipelined encoder: job intake, meta, and every NIXL
        operation stay on this thread; planning, decoding and the ViT run on
        :class:`EncoderPipeline` threads, overlapped across requests."""
        pipe = EncoderPipeline(
            self.runner,
            self.staging,
            decode_workers=self.runner.video_loader.num_workers,
        )
        poller = zmq.Poller()
        poller.register(self.job_sock, zmq.POLLIN)
        waiting: List[dict] = []  # staged, LM pages not known yet
        inflight: List[dict] = []
        last_disc = 0.0
        logger.info(f"[encoder {self.encoder_id}] serving jobs")
        while True:
            now = time.monotonic()
            if now - last_disc > 0.1:
                self._drain_discovery()
                last_disc = now
            timeout_ms = 1 if (inflight or waiting or pipe.busy) else 50
            socks = dict(poller.poll(timeout=timeout_ms))
            if self.job_sock in socks:
                for job in self._recv_jobs():
                    pipe.submit(job)

            planned = pipe.poll_planned()
            for job in pipe.dropped:
                self._active.pop((job.seq_id, job.item_idx), None)
                self._report_failure(job)
            pipe.dropped.clear()
            while not pipe.failed.empty():
                w = pipe.failed.get_nowait()
                if self._active.get((w.job.seq_id, w.job.item_idx)) is w:
                    self._forget(w)
                    self._report_failure(w.job)
            for w in planned:
                job = w.job
                key = (job.seq_id, job.item_idx)
                if key not in self._active:
                    continue  # cancelled while planning
                self._active[key] = w
                unit = max(hi - lo for lo, hi in w.seg_rows)
                if unit > self.staging.capacity:
                    logger.error(
                        f"[encoder {self.encoder_id}] job seq={job.seq_id} "
                        f"item={job.item_idx} stages {unit} rows at once > "
                        f"staging capacity {self.staging.capacity}; dropped"
                    )
                    self._active.pop(key, None)
                    self._report_failure(job)
                    continue
                try:
                    self._send_meta(job, w.num_tokens, w.grid_thw, w.chash)
                except Exception as e:
                    logger.error(
                        f"[encoder {self.encoder_id}] job seq={job.seq_id} "
                        f"item={job.item_idx} meta send failed: {e}"
                    )
                    self._active.pop(key, None)
                    self._report_failure(job)
                    continue
                pipe.start(w)

            for w, seg, pages, rows in pipe.poll_staged():
                waiting.append({"w": w, "seg": seg, "pages": pages, "rows": rows,
                                "attempt": 0, "since": time.monotonic()})
            still_waiting: List[dict] = []
            for t in waiting:
                w = t["w"]
                if w.failed:
                    self.staging.free(t["pages"])
                    continue
                t["dst"] = self._targets.get((w.job.seq_id, w.job.item_idx))
                if t["dst"] is None:
                    if time.monotonic() - t["since"] > self.target_timeout_s:
                        # The LM dropped the request (or is starved for pages):
                        # free the staging pages; its watchdog re-dispatches.
                        logger.warning(
                            f"[encoder {self.encoder_id}] job seq={w.job.seq_id} "
                            f"item={w.job.item_idx}: no LM pages after "
                            f"{self.target_timeout_s:.0f}s; dropped"
                        )
                        w.failed = True
                        self.staging.free(t["pages"])
                        self._forget(w)
                        self._report_failure(w.job)
                        continue
                    still_waiting.append(t)
                elif self._post_segment(t):
                    inflight.append(t)
                else:
                    self._abandon(t)
            waiting = still_waiting

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
                        self._abandon(t)
                    continue
                if not done:
                    still.append(t)
                    continue
                self._release_handles(t)
                w = t["w"]
                pipe.segment_written(w, t["seg"], t["pages"])
                self._notify_progress(w)
            inflight = still

    def _write_descs(self, t: dict) -> Tuple[list, Dict[str, list]]:
        """Paired (addr, nbytes, dev) pieces from the segment's staging pages
        to its rows in every LM TP rank's arena pages, merged where both sides
        are contiguous."""
        w, seg = t["w"], t["seg"]
        job = w.job
        lo, hi = w.seg_rows[seg]
        st = self.staging
        rb = st.row_bytes
        runs = copy_runs(
            t["pages"], st.rows_per_page, 0, t["dst"], job.dst_rows_per_page, lo, hi - lo
        )
        src_base, src_dev = self.send_reg.base_addr, self.send_reg.dev_id
        local: list = []
        remote: Dict[str, list] = {r.agent_name: [] for r in job.dst_regions}

        for sp, so, dp, do, n in runs:
            src = src_base + (sp * st.rows_per_page + so) * rb
            dst = {
                r.agent_name: r.base_addr + dp * job.dst_page_stride + do * rb
                for r in job.dst_regions
            }
            # Merge with the previous run when adjacent on every side.
            if local and local[-1][0] + local[-1][1] == src and all(
                remote[a][-1][0] + remote[a][-1][1] == dst[a] for a in remote
            ):
                local[-1] = (local[-1][0], local[-1][1] + n * rb, src_dev)
                for r in job.dst_regions:
                    last = remote[r.agent_name][-1]
                    remote[r.agent_name][-1] = (last[0], last[1] + n * rb, last[2])
            else:
                local.append((src, n * rb, src_dev))
                for r in job.dst_regions:
                    remote[r.agent_name].append((dst[r.agent_name], n * rb, r.dev_id))
        return local, remote

    def _post_segment(self, t: dict) -> bool:
        """Post the NIXL writes of one staged segment into every LM TP rank's
        pages for rows ``seg_rows[seg]``. False after ``write_max_attempts``."""
        w, seg = t["w"], t["seg"]
        local, remote = self._write_descs(t)
        while t["attempt"] < self.write_max_attempts:
            t["attempt"] += 1
            try:
                t["handles"] = [
                    self.nixl.write_descs(local, pieces, agent)
                    for agent, pieces in remote.items()
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

    def _report_failure(self, job) -> None:
        """Tell LM TP0 to re-dispatch now instead of waiting for a timeout."""
        target = job.lm_agent_names[0] if job.lm_agent_names else job.dst_regions[0].agent_name
        try:
            self.nixl.notify(target, emb_fail_notif(job.seq_id, job.item_idx))
        except Exception as e:
            logger.warning(f"[encoder {self.encoder_id}] failure notif lost: {e}")

    def _forget(self, w) -> None:
        """Stop tracking a failed item so a re-dispatched job runs again."""
        key = (w.job.seq_id, w.job.item_idx)
        if self._active.get(key) is w:
            del self._active[key]
            self._targets.pop(key, None)

    def _abandon(self, t: dict) -> None:
        w = t["w"]
        self.staging.free(t["pages"])
        if not w.failed:
            w.failed = True
            self._forget(w)
            self._report_failure(w.job)
            logger.error(
                f"[encoder {self.encoder_id}] job seq={w.job.seq_id} "
                f"item={w.job.item_idx} dropped: NIXL write failed; LM will "
                "re-dispatch"
            )

    def _notify_progress(self, w) -> None:
        """Announce the item's landed rows to LM TP0 as a growing prefix."""
        rows = w.done_prefix_rows()
        if rows <= w.notified_rows or w.failed:
            return
        job = w.job
        target = job.lm_agent_names[0] if job.lm_agent_names else job.dst_regions[0].agent_name
        if rows == w.num_tokens:
            msg = emb_notif(job.seq_id, job.item_idx)
            key = (job.seq_id, job.item_idx)
            self._targets.pop(key, None)
            self._active.pop(key, None)
            self._finished[key] = None
            while len(self._finished) > 4096:
                self._finished.popitem(last=False)
        else:
            msg = emb_partial_notif(job.seq_id, job.item_idx, rows)
        self.nixl.notify(target, msg)
        w.notified_rows = rows
