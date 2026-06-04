"""Encoder-side serving loop: ZMQ EncoderJob intake -> ViT -> NIXL write.

Drives a :class:`gllm.encoder_engine.EncoderEngine` from the disaggregation
control plane (design §4.2 / §5):

    for each EncoderJob(seq, item, modality, content, remote_slot):
        mm_input, grid = processor(content)            # CPU pixel IO
        push MmItemMeta(num_tokens, grid, hash) -----> LM PP0  (before ViT!)
        vis = ViT(mm_input)                            # GPU [N_vis, feat_dim]
        send_buf[:N_vis].copy_(vis)
        nixl.write(send_buf[:N_vis] -> remote_slot, notif="emb:seq:item")
        nixl.wait(handle)                              # reuse send_buf safely

Sending the meta *before* the ViT is the whole point of the per-item channel:
it lets the LM expand its skeleton token-ids and build the prefix-cache key
while the (slower) ViT + transfer are still in flight (design §5.4 / §6).

Phase 3b processes one job at a time (single in-flight transfer per encoder),
which keeps the persistent send buffer race-free without a ring. Per-item
pipelining + Encoder DP is Phase 5.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import List, Optional

import torch
import zmq
from logger import logger

from gllm.disagg.discovery import make_discovery, make_payload, payload_nixl_meta
from gllm.disagg.protocol import EncoderJob, MmItemMeta, emb_notif
from gllm.encoder_engine import EncoderEngine
from gllm.transfer.nixl_transfer import NixlEndpoint


class EncoderRuntime:
    def __init__(
        self,
        engine: EncoderEngine,
        encoder_id: str,
        discovery_endpoint: str,
        *,
        processor_config_hash: str = "",
        advertise_host: str = "127.0.0.1",
        job_bind: str = "tcp://0.0.0.0:0",
        max_vis_tokens: int = 16384,
        nixl_backend: str = "UCX",
    ):
        self.engine = engine
        self.encoder_id = encoder_id
        self.discovery_endpoint = discovery_endpoint
        self.processor_config_hash = processor_config_hash
        self.advertise_host = advertise_host
        self.job_bind = job_bind
        self.max_vis_tokens = max_vis_tokens
        self.nixl_backend = nixl_backend

        self.feat_dim = int(
            engine.model_loader.config.vision_config.out_hidden_size
            * (
                1
                + len(
                    getattr(
                        engine.model_loader.config.vision_config,
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
        # Current LM connection (single LM in Phase 4; keyed by discovery id).
        self.lm_identity: Optional[str] = None
        self.lm_agent_name: Optional[str] = None
        self.lm_zmq_addr: Optional[str] = None
        self.lm_payload: Optional[dict] = None  # last LM payload (for re-handshake)
        # NIXL write resilience (design §5.5): retry a failed write with a
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
            dtype=self.engine.dtype,
            device="cuda",
        )
        self.send_reg = self.nixl.register(self.send_buf)

        # Publish self into the registry + start watching for the LM. We do NOT
        # block here: the LM may come up later (any start order, design §7.3.4).
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
        for ev in self.disc.poll_events("lm"):
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
        self.lm_agent_name = self.nixl.connect(payload_nixl_meta(payload))
        if self.meta_sock is not None:
            self.meta_sock.close(0)
        self.meta_sock = self.zmq_ctx.socket(zmq.PUSH)
        self.meta_sock.connect(payload["zmq_addr"])
        self.lm_zmq_addr = payload["zmq_addr"]
        logger.info(
            f"[encoder {self.encoder_id}] connected to LM {identity} "
            f"agent={self.lm_agent_name} meta={self.lm_zmq_addr}"
        )

    def _disconnect_lm(self, identity: str, *, keep_quiet: bool = False) -> None:
        if self.lm_identity != identity:
            return
        if self.lm_agent_name is not None:
            self.nixl.disconnect(self.lm_agent_name)
        if self.meta_sock is not None:
            self.meta_sock.close(0)
            self.meta_sock = None
        if not keep_quiet:
            logger.info(f"[encoder {self.encoder_id}] LM {identity} left; idle")
        self.lm_identity = None
        self.lm_agent_name = None
        self.lm_zmq_addr = None

    # ------------------------------------------------------------------
    # Per-item processing is split into two phases so a *batch* of drained
    # jobs can emit ALL their metadata (gate A) before ANY ViT runs (design
    # §6.2). Metadata only needs the cheap CPU processor (grid_thw -> num_tokens
    # + content_hash), so completing gate A for the whole batch up front lets
    # the LM start prefilling the ready prefix while the (heavy, serialized)
    # ViTs for the remaining items are still running -- widening the
    # encode/prefill overlap window from "one ViT" to the full encode spread.
    @torch.inference_mode()
    def _prepare_job(self, job: EncoderJob) -> dict:
        """Phase A: CPU processor + send :class:`MmItemMeta` (gate A).

        Returns the state needed by :meth:`_encode_and_write` (phase B).
        """
        mm_input, grid_thw = self.engine.run_processor(job.content, job.modality)
        num_tokens = self.engine.num_vis_tokens(grid_thw)
        chash = self.engine.content_hash(mm_input, grid_thw)
        logger.debug(
            f"[encoder {self.encoder_id}] handling seq={job.seq_id} "
            f"item={job.item_idx} modality={job.modality} "
            f"slot={job.slot_id} num_tokens={num_tokens}"
        )

        if num_tokens > self.max_vis_tokens:
            raise ValueError(
                f"item needs {num_tokens} vis tokens > max_vis_tokens "
                f"{self.max_vis_tokens}; raise --max-vis-tokens"
            )

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
        return {
            "job": job,
            "mm_input": mm_input,
            "chash": chash,
            "num_tokens": num_tokens,
        }

    @torch.inference_mode()
    def _encode_and_write(self, prep: dict) -> None:
        """Phase B: ViT (with dedup cache) -> staging buffer -> NIXL WRITE."""
        job: EncoderJob = prep["job"]
        num_tokens: int = prep["num_tokens"]

        # ViT (with per-replica dedup cache), then stage into the send buf.
        vis = self.engine.encode(prep["mm_input"], prep["chash"])  # [N, feat_dim]
        assert vis.shape[0] == num_tokens, (
            f"ViT rows {vis.shape[0]} != predicted {num_tokens}"
        )
        assert vis.shape[1] == self.feat_dim, (
            f"ViT feat_dim {vis.shape[1]} != slot feat_dim {self.feat_dim}"
        )
        src = self.send_buf[:num_tokens]
        src.copy_(vis.to(self.send_buf.dtype))
        # The NIXL WRITE below is a UCX RDMA read of ``src`` issued OUTSIDE the
        # CUDA stream (``agent.transfer`` reads the raw device pointer), so it
        # does not honor stream ordering against the async ``copy_`` above.
        # Without this barrier the transport can ship stale/partial send-buffer
        # bytes whenever writes are issued back-to-back (batched phase B / high
        # concurrency) -- landing the WRONG image's embedding in the LM slot
        # (cross-request visual contamination). Make the copy fully visible to
        # the transport before launching the transfer.
        torch.cuda.synchronize()

        # NIXL WRITE into the LM's reserved slot, sub-sized to the actual item,
        # with the data-ready notif. Retried with a re-handshake on transient
        # transport failure (design §5.5).
        nbytes = num_tokens * self.feat_dim * self.send_buf.element_size()
        remote = job.remote_slot.with_offset(0, nbytes)
        self._write_with_retry(src, remote, job)

    def _write_with_retry(self, src, remote, job: EncoderJob) -> None:
        notif = emb_notif(job.seq_id, job.item_idx)
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.write_max_attempts + 1):
            try:
                handle = self.nixl.write(src, remote, notif_msg=notif)
                self.nixl.wait(handle)
                self.nixl.release(handle)
                return
            except Exception as e:  # transport hiccup: re-handshake + retry
                last_err = e
                logger.warning(
                    f"[encoder {self.encoder_id}] NIXL write seq={job.seq_id} "
                    f"item={job.item_idx} attempt {attempt}/"
                    f"{self.write_max_attempts} failed: {type(e).__name__}: {e}"
                )
                if attempt < self.write_max_attempts:
                    time.sleep(0.1 * attempt)  # linear backoff
                    self._reconnect_lm()
        raise RuntimeError(
            f"NIXL write seq={job.seq_id} item={job.item_idx} failed after "
            f"{self.write_max_attempts} attempts: {last_err!r}"
        )

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
            if self.meta_sock is not None and self.lm_agent_name is not None:
                return True
            self._drain_discovery()
            if self.meta_sock is not None and self.lm_agent_name is not None:
                return True
            time.sleep(sleep_s)
        return False

    def serve_forever(self) -> None:
        poller = zmq.Poller()
        poller.register(self.job_sock, zmq.POLLIN)
        logger.info(f"[encoder {self.encoder_id}] serving jobs")
        while True:
            # Pick up LM (re)connections / departures before touching jobs.
            self._drain_discovery()
            socks = dict(poller.poll(timeout=1000))
            if self.job_sock not in socks:
                continue
            # Drain every currently-available job into one batch so we can run
            # the cheap CPU/meta phase for ALL of them before any heavy ViT.
            batch: List[EncoderJob] = []
            while True:
                try:
                    raw = self.job_sock.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
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

            # Phase A: processor + emit meta (gate A) for the whole batch first.
            preps: List[dict] = []
            for job in batch:
                try:
                    preps.append(self._prepare_job(job))
                except Exception as e:  # pragma: no cover - operational guard
                    logger.error(
                        f"[encoder {self.encoder_id}] job seq={job.seq_id} "
                        f"item={job.item_idx} dropped in prepare: "
                        f"{type(e).__name__}: {e}"
                    )

            # Phase B: ViT + NIXL write (gate B), serialized (shared send buf).
            for prep in preps:
                job = prep["job"]
                try:
                    self._encode_and_write(prep)
                except Exception as e:  # pragma: no cover - operational guard
                    # Drop this item but KEEP serving: one bad item (e.g. a
                    # dead transport to the LM) must not take the whole replica
                    # out of the DP pool. The LM's in-flight job stays unacked;
                    # re-dispatch to a healthy replica is the Phase 8 watchdog.
                    logger.error(
                        f"[encoder {self.encoder_id}] job seq={job.seq_id} "
                        f"item={job.item_idx} dropped after error: "
                        f"{type(e).__name__}: {e}"
                    )
