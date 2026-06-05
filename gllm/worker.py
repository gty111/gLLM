"""Per-rank worker driver loops (non-overlap path).

Distributed scheduler design (v2)
=================================

Every PP=0 TP rank is now a *column driver*: it owns a real
:class:`Scheduler`, NCCL-broadcasts new front-end work to its peers
on the same PP stage, and only ships the resulting
:class:`SchedulePayload` over zmq to *its own column's* PP-other
followers (column ``k`` = ranks ``{k, tp_size+k, 2*tp_size+k, ...}``).

The motivation is the same as the overlap path's: the old design
had rank 0 as the sole scheduler and pushed a delta payload to every
TP follower over zmq each iteration. That broadcast was the dominant
source of inter-rank skew in TP=4 decode (~1.7 ms / iter on H100),
and even after the delta refactor still ate 50-200 us of wire
bookkeeping per iter. Moving the scheduler onto each column driver
keeps determinism (FIFO IDAllocator + identical broadcast input)
and removes both costs from the critical path.

PP>1 specifics
--------------

* Tokens still flow ``output_rank -> rank 0`` over the existing single
  PULL socket; rank 0 then NCCL-broadcasts the list to its PP=0 TP
  peers via :meth:`zmqComm.broadcast_tokens_to_tp` so each column's
  scheduler can ``add_next_tokens`` independently.
* PP-other ranks (``pp_rank > 0``) keep the :class:`FollowerSeqStore`
  mirror; they receive ``SchedulePayload`` deltas from their column's
  PP=0 TP=k driver.
* Sampling: every last-PP TP rank computes logits + samples in
  :meth:`ModelRunner.step_once`, but only the ``output_rank`` token
  list is shipped back. The other last-PP TP ranks do redundant
  sampling work that the design tolerates (it was already this way
  pre-refactor).
"""

import logging
import os
import traceback
from collections import deque
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from logger import logger

from gllm.comm import IPCPackage, zmqComm
from gllm.dist_schedule import (
    DriverPayloadBuilder,
    FollowerSeqStore,
    SchedulePayload,
)
from gllm.dist_utils import (
    get_last_pp_rank,
    get_next_pp_rank,
    get_pp_rank,
    get_pp_size,
    get_rank,
    get_tp_rank,
    get_tp_size,
    get_world_size,
    init_dist,
    is_first_pp_rank,
    is_last_pp_rank,
    is_output_rank,
    recv_pp_data,
    send_pp_data,
)
from gllm.input_data import InputData
from gllm.model_runner import ModelRunner, OverlapModelRunner
from gllm.profiler_mixin import TorchProfilerMixin
from gllm.scheduler import Scheduler


# Used with PipeAsyncLLM
class Worker(TorchProfilerMixin):

    def __init__(
        self,
        model_runner: Union[ModelRunner, OverlapModelRunner],
        local_rank,
        pp_rank,
        tp_rank,
        pp_size,
        tp_size,
        use_ep,
        master_addr,
        master_port,
        comm: zmqComm,
        mp_alive,
        mp_load_progress,
        assigned_layers,
        schedule_method,
        disagg_config=None,
    ):
        self.model_runner = model_runner
        self.local_rank = local_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.use_ep = use_ep
        self.master_addr = master_addr
        self.master_port = master_port
        self.comm = comm
        self.mp_alive = mp_alive
        self.mp_load_progress = mp_load_progress
        self.assigned_layers = assigned_layers
        self.schedule_method = schedule_method
        self.use_mla = model_runner.model_loader.use_mla
        # Encoder-disaggregation config (gllm.disagg.config.DisaggConfig),
        # pickled here from the parent across the spawn boundary; ``None`` on the
        # monolith path.
        self.disagg_config = disagg_config
        # Encoder-disaggregation LM-side state, built lazily in :meth:`init` when
        # ``disagg_config.is_lm`` is set. ``_disagg_recv`` (the per-rank NIXL
        # slot pool) lives on *every* PP0 TP rank; ``_disagg_coord`` (the TP0
        # control plane) lives only on TP0 (== rank 0). ``_is_disagg_lm`` is the
        # cheap per-iter routing flag (true on every column). All ``None`` /
        # False on the monolith.
        self._disagg_recv = None
        self._disagg_coord = None
        self._is_disagg_lm = False
        self.init_profiler_state()

    def init_logger(self):
        tp_ep_log = "TP" if not self.use_ep or self.tp_size == 1 else "TP/EP"
        formatter = logging.Formatter(
            f"[%(asctime)s %(filename)s:%(lineno)d Worker{self.pp_rank*self.tp_size+self.tp_rank} "
            f"PP{self.pp_rank} {tp_ep_log}{self.tp_rank}] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        for handler in logger.handlers:
            handler.setFormatter(formatter)

    def init(self):
        self.init_logger()
        if self.pp_size > 1 or self.tp_size > 1:
            init_dist(
                self.pp_size,
                self.tp_size,
                self.use_ep,
                self.local_rank,
                self.pp_rank,
                self.tp_rank,
                self.master_addr,
                self.master_port,
                self.assigned_layers,
            )
        self.rank = get_rank()
        torch.cuda.set_device(f"cuda:{self.local_rank}")

        self.comm.init()

        # Bring up the custom NVLink-P2P all-reduce path before the model
        # runner builds CUDA graphs -- graph capture has to see the same
        # AR implementation that replay will use, otherwise the captured
        # NCCL kernel and the eager custom kernel disagree on streams /
        # buffers and we get gradual KV drift between TP ranks (we hit
        # this exact failure mode when overlap scheduling captured on the
        # wrong stream; see ``OverlapModelRunner.capture_graph``).
        if self.tp_size > 1:
            from gllm.distributed import init_custom_allreduce
            from gllm.dist_utils import get_tp_group

            init_custom_allreduce(
                device=torch.device(f"cuda:{self.local_rank}"),
                group=get_tp_group(),
                rank=self.tp_rank,
                world_size=self.tp_size,
            )

        self.model_runner.init(self.mp_load_progress)
        self.hidden_size = self.model_runner.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual

        self._init_role_state()

        self._maybe_init_disagg()

        self.mp_alive[self.local_rank] = 1

        logger.info(f"Initialization complete")

    def _maybe_init_disagg(self):
        """Bring up the LM-side disagg state on every PP0 TP rank.

        Gated by ``disagg_config.is_lm`` (the entrypoint builds the config; see
        ``gllm.disagg.config.DisaggConfig``). Each PP0 TP rank builds a
        :class:`DisaggReceiver` (its own NIXL slot pool); the per-rank NIXL
        handshakes are gathered to TP0, which additionally builds the
        :class:`DisaggCoordinator` (control plane). TP>1 is supported natively:
        the encoder multi-writes the full embedding into every rank's pool and
        the coordinator fans its decisions out as per-iter events (see
        :meth:`recv_ipc_package`).
        """
        cfg = self.disagg_config
        if cfg is None or not cfg.is_lm:
            return
        if not is_first_pp_rank():
            return
        from gllm.disagg.lm_manager import DisaggCoordinator, DisaggReceiver

        self._is_disagg_lm = True
        lm_id = cfg.lm_id if cfg.lm_id is not None else "lm0"

        # LMDisaggManager defaults (num_slots=32, max_vis_tokens=16384) stay the
        # single source of truth when the operator didn't pin them.
        num_slots = cfg.num_slots if cfg.num_slots is not None else 32
        max_vis_tokens = cfg.max_vis_tokens if cfg.max_vis_tokens is not None else 16384

        recv = DisaggReceiver(
            self.model_runner,
            lm_id=lm_id,
            tp_rank=self.tp_rank,
            num_slots=num_slots,
            max_vis_tokens=max_vis_tokens,
            nixl_backend=cfg.nixl_backend,
        )
        recv.setup()
        self._disagg_recv = recv

        # Gather every PP0 TP rank's NIXL handshake (agent meta + slot-pool
        # region) to TP0 so the coordinator can pack one slot region per rank
        # into each EncoderJob and publish all agent metas to discovery. This is
        # a one-time, non-hot-path all-gather over the TP group.
        handshake = recv.handshake()
        if self.tp_size > 1:
            from gllm.dist_utils import get_tp_group

            gathered: List[Optional[dict]] = [None] * self.tp_size
            dist.all_gather_object(gathered, handshake, group=get_tp_group())
            rank_handshakes = [gathered[r] for r in range(self.tp_size)]
        else:
            rank_handshakes = [handshake]

        if self.tp_rank == 0:
            self._disagg_coord = DisaggCoordinator(
                self.model_runner,
                recv,
                rank_handshakes=rank_handshakes,
                lm_id=lm_id,
                discovery_endpoint=cfg.discovery_endpoint,
                processor_config_hash=cfg.processor_config_hash,
                advertise_host=cfg.advertise_host,
                meta_bind=cfg.meta_bind,
                num_slots=num_slots,
                nixl_backend=cfg.nixl_backend,
                encoder_dp=cfg.encoder_dp,
            )
            self._disagg_coord.setup()

    def _admit_requests(self, seqs):
        """Route new front-end seqs: disagg mm seqs -> coordinator, else scheduler.

        A disaggregated multimodal request carries ``seq.mm_items`` (the raw
        per-item content the encoder needs) and a *skeleton* token-id list. It
        must not enter the scheduler until its visual embeddings have arrived.
        On TP0 it is handed to the coordinator; on the other TP ranks it is
        simply dropped -- the expanded, ready seq arrives later as a fanned-out
        ADMIT event (see :meth:`_apply_disagg_events`). Everything else (text,
        and the monolith path) goes straight to the scheduler on every column.
        """
        if not self._is_disagg_lm:
            self.scheduler.add_new_requests(seqs)
            return
        direct = []
        for seq in seqs:
            if getattr(seq, "mm_items", None):
                if self._disagg_coord is not None:
                    self._disagg_coord.submit(seq)
                # non-TP0: drop; the expanded seq returns via an ADMIT event.
            else:
                direct.append(seq)
        if direct:
            self.scheduler.add_new_requests(direct)

    def _apply_disagg_events(self, events) -> None:
        """Apply one iteration's fanned-out disagg events on this column.

        Runs identically on every PP0 TP rank (TP0 applies the authoritative
        objects it built; peers apply their freshly-pickled copies). ``ADMIT``
        registers the gate state + adds the expanded seq to this column's
        scheduler; ``EMB_READY`` clones the embedding out of *this rank's* local
        slot pool. Keeping both on the same fanned-out stream guarantees every
        column mutates its scheduler / model runner in lockstep.
        """
        if not events:
            return
        for seq, state in events.admits:
            self.model_runner.disagg_register(seq.seq_id, state)
            self.scheduler.add_new_requests([seq])
        if events.emb_ready:
            self._disagg_recv.sync()
            for seq_id, ordered_idx, slot_id, num_tokens in events.emb_ready:
                emb = self._disagg_recv.clone_slot(slot_id, num_tokens)
                self.model_runner.disagg_set_embedding(seq_id, ordered_idx, emb)
        if events.aborts:
            # An admitted seq whose encode failed unrecoverably (coordinator
            # watchdog gave up). Drop it from every column's scheduler in this
            # same iteration; ``check_abort_seqs`` -> ``model_runner.free``
            # reclaims its page / SSM slot + disagg state on each rank.
            self.scheduler.add_abort_ids(events.aborts)

    def _init_role_state(self):
        """Build per-rank scheduler / follower-store state.

        Every PP=0 TP rank gets a :class:`Scheduler` + a
        :class:`DriverPayloadBuilder` (one builder per column; cursors
        are independent of other columns'). PP-other ranks get a
        :class:`FollowerSeqStore` mirroring their column's PP=0
        driver, plus a queue of pre-built :class:`InputData`. The
        old design constructed this only on rank 0 and relied on a
        rank-0-centric zmq broadcast; the new column-driver design
        means every PP=0 TP rank looks like rank 0 here, and PP-other
        ranks pull from their column's driver instead of from rank 0.
        """
        if is_first_pp_rank():
            self.scheduler = Scheduler(
                self.pp_size,
                self.model_runner,
                self.schedule_method,
            )
            # One DriverPayloadBuilder per column. We do NOT share a
            # single builder across columns -- each column has its own
            # zmq fanout and its own ``_pending_follower_frees``
            # accumulator, so the cursors must be per-rank.
            self.payload_builder = DriverPayloadBuilder()
        else:
            # Per-rank state mirror. Replaces the stateless "rebuild
            # InputData from a freshly-pickled Sequence list every iter"
            # pattern with an incremental delta application.
            # VL models need ``seq.token_ids`` on the follower for *every*
            # prefill seq (even text-only ones) because ``_mm_prepare_cpu``'s
            # ``MRotaryEmbedding.get_input_positions`` + ``torch.isin`` paths
            # read it unconditionally; see :class:`FollowerSeq` docstring.
            self.follower_store = FollowerSeqStore(
                mm_needs_token_ids=self.model_runner.use_mm,
            )
            # Input data for each rank except 0
            self.schedule_queue = deque()

    # ------------------------------------------------------------------
    # PP-other receive / forward (unchanged behaviour, sockets per-column)
    # ------------------------------------------------------------------

    def recv_schedule_payload(self) -> None:
        """Poll for one :class:`SchedulePayload`, apply it, queue InputData.

        Non-blocking single recv. The follower-store apply is a few dict
        ops + a list comprehension; the heavy work (``cal_input``)
        happens here on the main thread, identical to the pre-refactor
        path. We deliberately apply frees *after* materializing the
        batch so a payload that frees a seq it also schedules (currently
        impossible, defensively handled) does the right thing.
        """
        payload = self.comm.recv_schedule_payload()
        if payload is None:
            return

        if payload.control_cmd != 0:
            self._apply_control_cmd(payload.control_cmd, payload.control_data)

        seqs = self.follower_store.apply_payload(payload)

        # Clean up follower-local caches (VL ``embedding_cache``) for
        # seqs the driver just freed.  Important to do this *after*
        # ``apply_payload`` so we use the same eviction order rank-0
        # used (free-after-update => mirror is in the post-batch
        # state).
        if payload.frees:
            for sid in payload.frees:
                self.model_runner.free_follower_state(sid)

        if not seqs:
            return

        input_data = InputData(
            use_buffer=False,
            memory_manager=self.model_runner.memory_manager,
            max_seq_length=self.model_runner.model_max_length,
        )
        input_data.cal_input(seqs)
        if payload.mrope_positions is not None:
            input_data.set_mrope_position(payload.mrope_positions)
        self.schedule_queue.append(input_data)

    def forward_pp(self):
        if len(self.schedule_queue) != 0:
            input_data: InputData = self.schedule_queue.popleft()
            # pp last rank => pp next rank
            recv_pp_data(
                get_last_pp_rank(),
                input_data.tokens_cpu.shape[0],
                self.model_runner.input_hidden_states,
                self.model_runner.input_residual,
                self.ret_residual,
            )
            self.model_runner.prepare_input(input_data=input_data)
            output = self.model_runner.step_once()
            if is_output_rank():
                self.comm.send_tokens(output)
            elif not is_last_pp_rank():
                send_pp_data(output, get_next_pp_rank())

    # ------------------------------------------------------------------
    # PP=0 column driver: input distribution, scheduling, forward, output
    # ------------------------------------------------------------------

    def recv_ipc_package(self):
        """Drain frontend zmq on rank 0, fan out to PP=0 TP peers.

        Every PP=0 TP rank invokes this each iter; only rank 0
        actually polls the frontend socket. The aggregated
        :class:`IPCPackage` (or ``None`` if nothing was waiting) is
        shipped to peer column drivers over a CPU-side zmq fan-out
        (see :meth:`zmqComm.broadcast_input_to_tp`) so every column
        driver adds the same requests / aborts / control commands to
        its local scheduler in lockstep, without consuming NVLink
        bandwidth that would otherwise contend with the model's
        per-layer all-reduce.
        """
        cum: Optional[IPCPackage] = None
        if self.rank == 0:
            cum = IPCPackage([])
            saw_log_override = False
            while True:
                ipc_package = self.comm.recv_ipc_package()
                if ipc_package is None:
                    break
                cum.schedule_lists.extend(ipc_package.schedule_lists)
                cum.abort_ids.extend(ipc_package.abort_ids)
                if ipc_package.log is not None:
                    cum.log = ipc_package.log
                    saw_log_override = True
                if ipc_package.control_cmd is not None:
                    code, data = self._translate_control_cmd(
                        ipc_package.control_cmd
                    )
                    if code != 0:
                        cum.control_cmd_code = code
                        cum.control_data = data
            if not saw_log_override:
                cum.log = None

        # TP0 control plane: drive the disagg coordinator (discovery / meta /
        # notif / dispatch / watchdog) once per iter and attach the resulting
        # ADMIT / EMB_READY events to the package so they fan out with the input
        # and every column applies them in this same iteration. Must run before
        # the empty-elision (events alone are enough to keep ``cum`` alive) and
        # before the broadcast so peers receive them.
        if self.rank == 0 and self._disagg_coord is not None:
            events = self._disagg_coord.poll()
            if events:
                if cum is None:
                    cum = IPCPackage([])
                    cum.log = None
                cum.disagg_events = events

        if self.rank == 0:
            if (
                cum is not None
                and cum.is_input_empty()
                and cum.log is None
                and cum.disagg_events is None
            ):
                cum = None

        if get_tp_size() > 1:
            cum = self.comm.broadcast_input_to_tp(cum)

        if cum is None:
            return

        if cum.schedule_lists:
            self._admit_requests(cum.schedule_lists)
        if cum.disagg_events is not None:
            self._apply_disagg_events(cum.disagg_events)
        if cum.abort_ids:
            self.scheduler.add_abort_ids(cum.abort_ids)
            # TP0 also reclaims any coordinator-held NIXL receive slots for
            # aborted seqs still pending pre-admission (the scheduler-side
            # teardown above already covers admitted seqs on every column).
            if self._disagg_coord is not None:
                self._disagg_coord.abort(cum.abort_ids)
        if cum.log is not None:
            self.scheduler.set_log(cum.log)
        if cum.control_cmd_code != 0:
            # Apply locally on this column driver, then forward to
            # PP-other followers via the SchedulePayload mechanism so
            # they enable / disable the profiler in the same iter.
            self._apply_control_cmd(cum.control_cmd_code, cum.control_data)
            if get_pp_size() > 1:
                self.comm.broadcast_control_cmd(
                    cum.control_cmd_code, cum.control_data
                )

    def _translate_control_cmd(self, control_cmd: str):
        """rank-0-only string -> (code, data) translation for broadcast."""
        if control_cmd == "start_profile":
            import os
            import time

            start_ts = int(time.time())
            profile_session_dir = os.path.join(
                self.profile_output_dir,
                f"trace_session_{start_ts}",
            )
            return 1, profile_session_dir
        if control_cmd == "stop_profile":
            return 2, None
        return 0, None

    def recv_next_tokens(self):
        """Pull tokens off the output_rank => rank 0 socket and broadcast.

        For PP=1 the broadcast is a no-op: the sample tokens are
        already shared across the TP group inside ``run_batch_async``
        / ``step_once`` (every TP rank runs the sampler with the same
        all-reduced logits). For PP>1 only the ``output_rank`` (last
        PP, TP=0) samples and ships the int list back to rank 0; we
        then NCCL-fan it out to every PP=0 TP rank so each column
        driver's scheduler can ``add_next_tokens`` independently.
        """
        if get_pp_size() == 1:
            return

        next_tokens: Optional[List[int]] = None
        if self.rank == 0:
            next_tokens = self.comm.recv_tokens()

        if get_tp_size() > 1:
            next_tokens = self.comm.broadcast_tokens_to_tp(next_tokens)

        if next_tokens is not None:
            self.scheduler.add_next_tokens(next_tokens)

    def check_abort_seqs(self):
        """Process aborts on every column driver; only rank 0 replies."""
        ipc_package = self.scheduler.check_abort_seqs()
        if ipc_package is not None and self.rank == 0:
            self.comm.send_output(ipc_package)

    def process_output(self):
        """Finalize this column's batch; only rank 0 replies to frontend."""
        ipc_package = self.scheduler.process_output()
        if ipc_package is not None and self.rank == 0:
            self.comm.send_output(ipc_package)

    def _build_schedule_payload(
        self,
        schedule_seqs: List,
    ) -> Optional[SchedulePayload]:
        """Build the per-column delta payload for this scheduling iteration.

        Each column driver maintains its own ``payload_builder``
        cursors (``_known``, ``_last_pages_len``) and its scheduler's
        ``_pending_follower_frees`` accumulator -- the column drivers
        run identical schedules, but the payloads are still local
        objects pickled by independent zmq sender threads. Returns
        ``None`` for ``world_size == 1`` (single-rank shortcut).
        """
        if get_world_size() <= 1:
            return None
        return self.payload_builder.build(
            scheduled_seqs=schedule_seqs,
            frees=self.scheduler.consume_pending_follower_frees(),
            mrope_positions=None,
            use_mm=self.model_runner.use_mm,
        )

    def schedule_forward(self):
        schedule_seqs = self.scheduler.schedule_once()
        if len(schedule_seqs) == 0:
            return
        # Every PP-0 column driver builds its own per-column delta
        # payload (cursors are per-rank). For ``pp_size > 1`` the
        # payload also carries ``mrope_positions`` for VL models,
        # which only get computed inside ``prepare_input``, so the
        # send to PP-other followers must happen *after* local input
        # prep. (Pre-refactor we used to do an extra "early send" to
        # TP followers without ``mrope_positions`` so their
        # ``cal_input`` could overlap with ours, but with the
        # per-column scheduler design TP followers no longer exist.)
        # For ``pp_size == 1`` ``send_schedule_payload`` is an inline
        # no-op since this column has no PP-other followers.
        payload = self._build_schedule_payload(schedule_seqs)
        self.model_runner.prepare_input(schedule_seqs)
        if payload is not None and get_pp_size() > 1:
            # Subsequent PP stages don't run ``_mm_prepare_cpu``,
            # so we ship the m-rope positions we just built.
            import dataclasses

            pp_payload = dataclasses.replace(
                payload,
                mrope_positions=(
                    self.model_runner.input_data.mrope_positions_cpu
                ),
            )
            self.comm.send_schedule_payload(pp_payload)
        output = self.model_runner.step_once()

        if is_last_pp_rank():
            # ``step_once`` returns a List[int] of sampled tokens on
            # every last-PP TP rank.
            next_tokens = output
            if get_pp_size() == 1:
                # PP=1: every TP rank is also a column driver and
                # needs these tokens for ``add_next_tokens``. Random
                # samplers (top_k != 1, multinomial) are not
                # guaranteed to produce TP-identical tokens, so we
                # broadcast from output_rank within the TP group --
                # OverlapModelRunner does this on the GPU side, but
                # the eager non-overlap path doesn't, so we do it
                # explicitly here. For greedy decoding the broadcast
                # is harmless (~10us / iter).
                if get_tp_size() > 1:
                    next_tokens = self.comm.broadcast_tokens_to_tp(
                        next_tokens if is_output_rank() else None
                    )
                if next_tokens is not None:
                    self.scheduler.add_next_tokens(next_tokens)
            elif is_output_rank():
                # PP>1: only output_rank's tokens flow back to
                # rank 0 via the existing token zmq pair; the other
                # column drivers pick them up in next iter's
                # ``recv_next_tokens`` (which NCCL-fans them out
                # across the PP-0 TP group).
                self.comm.send_tokens(next_tokens)
            # last-PP TP>0 ranks for PP>1: discard ``next_tokens``;
            # they don't drive a scheduler.
        else:
            # PP-0 / mid-PP ranks: send hidden states downstream.
            send_pp_data(output, get_next_pp_rank())

    def run_pp0(self):
        """Per-iter loop run by every PP=0 TP rank."""
        self.check_abort_seqs()
        # ``recv_ipc_package`` also drives the disagg coordinator (TP0) and
        # applies its fanned-out ADMIT / EMB_READY events on every column.
        self.recv_ipc_package()
        self.recv_next_tokens()
        self.schedule_forward()
        self.process_output()

    def run_other(self):
        self.recv_schedule_payload()
        self.forward_pp()

    def handle_keyboardInterrupt(self):
        self.mp_alive[self.local_rank] = -1
        logger.info(f"Exit")
        if dist.is_initialized():
            dist.destroy_process_group()

    def handle_exception(self, e):
        logger.error(e)
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.mp_alive[self.local_rank] = -1


def run_worker(worker: Worker):
    try:
        worker.init()
        while True:
            if worker.pp_rank == 0:
                worker.run_pp0()
            else:
                worker.run_other()
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
