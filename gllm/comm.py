import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import zmq

from gllm.dist_schedule import SchedulePayload
from gllm.dist_utils import (
    get_ipc_tp_group,
    get_output_rank,
    get_pp_rank,
    get_pp_size,
    get_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
    get_world_size,
    is_output_rank,
    recv_obj_list,
    send_obj_list,
)
from gllm.sequence import Sequence
from gllm.utils import make_socket


_SHUTDOWN = object()  # sentinel pushed onto a sender queue to drain it


class IPCPackage:
    def __init__(self, schedule_lists: List[Sequence]):
        # front-end => worker
        self.log = True
        self.schedule_lists = schedule_lists
        self.abort_ids = []  # seq_ids to abort
        self.control_cmd = None  # optional control command (e.g., start/stop profile)
        # ``control_cmd_code`` and ``control_data`` are populated by the
        # rank-0 worker before it broadcasts an :class:`IPCPackage` to its
        # TP peers via :meth:`zmqComm.broadcast_input_to_tp`. They carry
        # the post-translation form of ``control_cmd`` (e.g. the
        # profile_session_dir) so that every TP rank applies an
        # *identical* command -- otherwise each rank would mint its own
        # ``time.time()``-based dir and the per-rank traces would
        # scatter across N session folders. ``control_cmd`` (the raw
        # string) is dropped on the worker after translation; only the
        # code/data pair survives the broadcast.
        self.control_cmd_code = 0
        self.control_data = None
        # worker => front-end
        self.free_ids = []  # seq_ids to free
        self.act_schedule_ids = []
        self.next_tokens = []

    def is_input_empty(self) -> bool:
        """True iff this package carries no front-end => worker work.

        Used by the TP broadcast fast-path to skip the
        ``broadcast_object_list`` call on iters with no new requests /
        aborts / control commands (the steady-state decode case, where
        99 %+ of iters have nothing to send).
        """
        return (
            len(self.schedule_lists) == 0
            and len(self.abort_ids) == 0
            and self.control_cmd_code == 0
        )


class zmqComm:
    def __init__(
        self,
        host_addr,
        port_base,
        launch_mode,
        master_addr,
        schedule_path,
        output_path,
        token_path,
        frontend=False,
    ):
        self.host_addr = host_addr
        self.port_base = port_base
        self.master_addr = master_addr
        self.launch_mode = launch_mode
        self.schedule_path = schedule_path
        self.output_path = output_path
        self.token_path = token_path
        self.frontend = frontend

    def init(self):
        self.ctx = zmq.Context()
        # Persistent zmq-sender threads keyed by socket. See ``_get_sender``
        # for why we avoid the prior fresh-thread-per-send pattern.
        self._senders: Dict["zmq.Socket", "queue.SimpleQueue"] = {}

        if self.frontend:  # front-end process
            self.request_socket = make_socket(self.ctx, self.schedule_path, zmq.PUSH)
            self.output_socket = make_socket(self.ctx, self.output_path, zmq.PULL)
            return

        # ------------------------------------------------------------------
        # Worker-process socket layout (per-column scheduler design)
        # ------------------------------------------------------------------
        #
        # Pre-refactor topology (rank-0-centric):
        #   * rank 0 ran the only Scheduler and pushed ``SchedulePayload``
        #     to every other rank: one per TP follower on PP-0 plus one
        #     per PP-other rank.
        #   * Tokens flowed output_rank -> rank 0 over a single PULL.
        #
        # New topology: every PP-0 TP rank is a *column driver*. Column
        # ``k`` consists of (PP=0,TP=k), (PP=1,TP=k), ..., (PP=N-1,TP=k);
        # the driver runs its own deterministic scheduler and only sends
        # ``SchedulePayload`` to *its own column's* PP-other ranks. New
        # requests / aborts / control commands arrive at rank 0 from the
        # frontend and are NCCL-broadcast within the PP-0 TP group via
        # :meth:`broadcast_input_to_tp` (no zmq fan-out between TP ranks).
        # Tokens still funnel through rank 0 (output_rank still uses a
        # single PULL into rank 0); rank 0 NCCL-broadcasts the result
        # within the PP-0 TP group via :meth:`broadcast_tokens_to_tp`.
        #
        # For PP=1 (overlap path) the per-column structure collapses
        # cleanly: ``schedule_other_sockets`` is empty, no tokens leg,
        # and NCCL alone handles every cross-rank message.
        rank = get_rank()
        pp_rank = get_pp_rank()
        tp_rank = get_tp_rank()
        pp_size = get_pp_size()
        tp_size = get_tp_size()

        if rank == 0:
            # rank 0 is the only worker that talks to the frontend.
            self.request_socket = make_socket(self.ctx, self.schedule_path, zmq.PULL)
            self.output_socket = make_socket(self.ctx, self.output_path, zmq.PUSH)
            if pp_size > 1:
                # output_rank => rank 0 : next tokens (single PULL,
                # broadcast inside TP group below)
                if self.launch_mode == "normal":
                    self.token_socket = make_socket(
                        self.ctx, self.token_path, zmq.PULL
                    )
                else:
                    port_token = self.port_base + get_world_size()
                    self.token_socket = make_socket(
                        self.ctx,
                        f"tcp://{self.master_addr}:{port_token}",
                        zmq.PULL,
                    )
                    send_obj_list([port_token], get_output_rank())

        # Every PP-0 rank becomes a column driver. For ``pp_size == 1``
        # the loop is empty, ``schedule_other_sockets`` stays an empty
        # list, and the schedule send paths short-circuit -- exactly
        # what we want for the overlap path.
        if pp_rank == 0:
            self.schedule_other_sockets: List[zmq.Socket] = []
            if pp_size > 1:
                if self.launch_mode == "normal":
                    for pp in range(1, pp_size):
                        target_rank = pp * tp_size + tp_rank
                        socket = make_socket(
                            self.ctx,
                            f"{self.schedule_path}_{target_rank}",
                            zmq.PUSH,
                        )
                        self.schedule_other_sockets.append(socket)
                else:
                    for pp in range(1, pp_size):
                        target_rank = pp * tp_size + tp_rank
                        port_each = self.port_base + target_rank
                        send_obj_list([port_each], target_rank)
                        addr_each = [None]
                        recv_obj_list(addr_each, target_rank)
                        socket = make_socket(
                            self.ctx,
                            f"tcp://{addr_each[0]}:{port_each}",
                            zmq.PUSH,
                        )
                        self.schedule_other_sockets.append(socket)
        else:
            # PP-other rank: pull from its own column driver
            # (rank ``tp_rank`` on PP=0).
            if self.launch_mode == "normal":
                self.schedule_socket = make_socket(
                    self.ctx, f"{self.schedule_path}_{rank}", zmq.PULL
                )
            else:
                port_schedule = [None]
                recv_obj_list(port_schedule, tp_rank)  # column driver
                send_obj_list([self.host_addr], tp_rank)
                self.schedule_socket = make_socket(
                    self.ctx,
                    f"tcp://{self.host_addr}:{port_schedule[0]}",
                    zmq.PULL,
                )

        if is_output_rank() and pp_size != 1:
            # output_rank (last-PP TP=0) => rank 0 : next tokens.
            if self.launch_mode == "normal":
                self.token_socket = make_socket(self.ctx, self.token_path, zmq.PUSH)
            else:
                port_token = [None]
                recv_obj_list(port_token, 0)
                self.token_socket = make_socket(
                    self.ctx,
                    f"tcp://{self.master_addr}:{port_token[0]}",
                    zmq.PUSH,
                )

    def send_tokens(self, tokens):
        assert type(tokens) == list
        self.token_socket.send_pyobj(tokens)

    def recv_tokens(self):
        if self.token_socket.poll(timeout=0) != 0:
            next_tokens = self.token_socket.recv_pyobj()
            return next_tokens
        else:
            return None

    def send_output(self, output):
        self.output_socket.send_pyobj(output)

    def recv_output(self):
        if self.output_socket.poll(timeout=0) != 0:
            output = self.output_socket.recv_pyobj()
            return output
        else:
            return None

    def _get_sender(self, socket: "zmq.Socket") -> "queue.SimpleQueue":
        """Return a persistent FIFO that ships pyobjs to ``socket``.

        Originally we spun up a fresh ``threading.Thread(target=socket.send_pyobj)``
        for every send. Profiler showed ~205 us per ``threading.start()`` call
        and ~28 ms / run of pure thread-creation overhead (Qwen3-0.6B TP=2,
        137 batches, two sends per batch). Replacing those one-shot threads
        with one long-lived sender per socket drops each send to a
        ``SimpleQueue.put`` (~1 us) and also makes the zmq usage thread-safe
        by construction -- zmq sockets are not safe to share across threads
        and the previous design relied on each one-shot send finishing before
        the next batch's send happened, which was racy under load.

        The sender thread is daemon=True so it dies with the process; we
        deliberately don't bother with a graceful shutdown path because
        worker processes exit via SIGTERM today.
        """
        sender = self._senders.get(socket)
        if sender is not None:
            return sender
        q: "queue.SimpleQueue" = queue.SimpleQueue()

        def _run() -> None:
            send_pyobj = socket.send_pyobj
            while True:
                payload = q.get()
                if payload is _SHUTDOWN:
                    return
                try:
                    send_pyobj(payload)
                except Exception:
                    # Mirror the prior fire-and-forget behaviour: a failing
                    # send used to crash a one-shot thread silently. Don't
                    # take down the whole sender on a single bad send.
                    pass

        t = threading.Thread(target=_run, daemon=True, name="zmq-sender")
        t.start()
        self._senders[socket] = q
        return q

    def send_schedule_payload(
        self,
        payload: SchedulePayload,
    ):
        """Ship one :class:`SchedulePayload` to this column's PP-other ranks.

        With the per-column scheduler design TP synchronization no
        longer goes through zmq -- each PP-0 TP rank runs its own
        deterministic scheduler and broadcasts new front-end work via
        NCCL (:meth:`broadcast_input_to_tp`). The only zmq schedule
        traffic that remains is the PP=0 TP=k -> PP=p TP=k path for
        ``p > 0``, which still benefits from the delta-style payload
        because we don't have a CPU-side group covering "this column"
        cheaply.

        Callers must be PP=0 ranks (see :meth:`init`); we no longer
        differentiate "first PP" vs "other PP" follower groups because
        each PP-0 rank only owns one set of sockets (its own column's
        PP-other followers).
        """
        if payload.is_empty():
            return
        if not getattr(self, "schedule_other_sockets", None):
            return
        for socket in self.schedule_other_sockets:
            self._get_sender(socket).put(payload)

    def broadcast_control_cmd(
        self, control_cmd_code: int, profile_session_dir: Optional[str] = None
    ):
        """Ship an empty-schedule payload to this column's PP-other ranks.

        Used by the profiler-start/stop plumbing on each PP-0 TP rank
        (every column driver fires this independently after the
        :meth:`broadcast_input_to_tp` step in the schedule loop has
        agreed that a control command needs forwarding to PP-other
        followers). For PP=1 the socket list is empty and this is a
        no-op.
        """
        payload = SchedulePayload(
            control_cmd=control_cmd_code,
            control_data=profile_session_dir,
        )
        if not getattr(self, "schedule_other_sockets", None):
            return
        for socket in self.schedule_other_sockets:
            self._get_sender(socket).put(payload)

    # ------------------------------------------------------------------
    # NCCL-backed TP-group input broadcast (overlap-scheduling path)
    # ------------------------------------------------------------------
    #
    # The overlap path moves scheduling onto every PP0 TP rank, so each
    # rank needs the exact same stream of front-end => worker messages
    # (new requests, aborts, control commands). Rank-0 polls zmq, then
    # publishes the aggregated :class:`IPCPackage` to its TP peers with
    # the helpers below.
    #
    # Determinism rule: every PP0 TP rank MUST call this every
    # iteration in the same order; otherwise the NCCL broadcast group
    # desyncs and the scheduler states fork. Skipping the call on
    # "obviously empty" iters is therefore not allowed -- we fold the
    # fast path *inside* the method (a single 1-element broadcast that
    # signals whether the heavy ``broadcast_object_list`` follows).

    def _ensure_tp_broadcast_state(self) -> None:
        """Lazy init of the per-iter scratch tensor used by the broadcast.

        The tensor lives on the worker's CUDA device (NCCL group is
        device-bound); we keep one shared instance to avoid a fresh
        allocation + .item() pinned-memory dance every iter.
        """
        if getattr(self, "_tp_bcast_flag_gpu", None) is None:
            self._tp_bcast_flag_gpu = torch.zeros(1, dtype=torch.long, device="cuda")
            # Pinned host buffer so the .item() readback after the
            # broadcast goes through pinned-memory DMA instead of the
            # default malloc-cudaMemcpyAsync-sync dance. ~1us savings
            # per iter, but it's on the critical path so it adds up.
            self._tp_bcast_flag_cpu = torch.zeros(
                1, dtype=torch.long, device="cpu", pin_memory=True
            )
            # Source rank inside the TP group: tp_rank == 0 of this PP
            # stage. Pre-compute once so the per-iter broadcast doesn't
            # have to recompute it. For pp_size=1 this is global rank 0.
            self._tp_bcast_src_rank = get_pp_rank() * get_tp_size()

    def broadcast_input_to_tp(
        self, ipc_package: Optional["IPCPackage"]
    ) -> Optional["IPCPackage"]:
        """Rank-0-driven broadcast of an :class:`IPCPackage` to TP peers.

        Two-phase protocol:

        1. **Header phase** -- a 1-element ``int64`` broadcast over the
           TP NCCL group. ``0`` means "no payload follows", any other
           value means "expect a ``broadcast_object_list``". This is
           cheap (~5 us on H100 NVLink) and lets us skip the heavier
           pickle path on iters with nothing new (the common steady
           state).

        2. **Payload phase** (only when header != 0) --
           :func:`torch.distributed.broadcast_object_list` carries the
           pickled ``IPCPackage`` over the same TP NCCL group. The
           payload includes the (post-translation) control command
           code + data, so every rank applies an identical command.

        ``ipc_package`` is honored on rank-0 only; on other TP ranks it
        is ignored (we always return what came over the wire).

        Caller contract:
        * Must be called on every PP0 TP rank every iteration in the
          same order (lock-step with the schedule loop).
        * Must NOT be called from PP-other ranks (their TP group lives
          on a different PP stage; we only define the broadcast for
          the first PP stage).
        """
        if get_tp_size() <= 1:
            return ipc_package

        self._ensure_tp_broadcast_state()
        flag_gpu = self._tp_bcast_flag_gpu
        src = self._tp_bcast_src_rank
        # Use the dedicated IPC NCCL group (separate communicator) so
        # the broadcast kernel doesn't queue behind the forward path's
        # per-layer all_reduces -- which would otherwise force the
        # .item() readback below to wait for the previous iter's
        # entire forward to drain (we measured ~6 ms / iter on
        # ``_TP_GROUP`` vs ~10-20 us on ``_IPC_TP_GROUP``).
        tp_group = get_ipc_tp_group()

        if get_rank() == src:
            # Caller is responsible for passing ``None`` when there's
            # nothing to ship (the ``recv_ipc_package`` path collapses
            # an "empty input + no log override" cum to ``None``
            # before reaching us). This keeps log-only updates -- e.g.
            # the per-iter ``log=True/False`` toggle -- on the wire.
            flag_gpu.fill_(1 if ipc_package is not None else 0)
        # Rank-0 fills, all ranks broadcast. Non-source ranks pass in
        # whatever they have (they overwrite from the wire anyway).
        dist.broadcast(flag_gpu, src=src, group=tp_group)
        # Pinned readback => fast .item(). For the common header==0
        # case this is the only sync we pay per iter on followers.
        self._tp_bcast_flag_cpu.copy_(flag_gpu, non_blocking=False)
        if int(self._tp_bcast_flag_cpu.item()) == 0:
            return None

        obj_list = [ipc_package if get_rank() == src else None]
        # ``device`` must be a CUDA device for NCCL backend; otherwise
        # PyTorch falls back to a temporary process group on CPU which
        # serializes the broadcast onto the gloo store and hurts perf.
        dist.broadcast_object_list(
            obj_list,
            src=src,
            group=tp_group,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        return obj_list[0]

    def broadcast_tokens_to_tp(
        self, next_tokens: Optional[List[int]]
    ) -> Optional[List[int]]:
        """Rank-0-driven broadcast of sampled tokens to PP-0 TP peers.

        Used by the non-overlap (Worker) path when ``pp_size > 1``: the
        last-PP TP=0 rank pushes a list of integer token ids into
        rank 0 over zmq, and rank 0 then has to make that list visible
        to every column driver on PP=0 so each driver's scheduler can
        ``add_next_tokens`` and process the iteration's output. We
        carry the list as an int64 GPU tensor so that the broadcast
        rides the existing TP NCCL communicator (no second backend, no
        gloo round trip).

        For the overlap path (PP=1) this method is unnecessary: the
        token broadcast already happens GPU-side inside
        :meth:`OverlapModelRunner.run_batch_async` on the same TP NCCL
        group, and every PP-0 TP rank D2H-copies it locally.

        Returns ``None`` (on every TP rank in lockstep) when the source
        rank had no tokens to broadcast this iter -- callers can then
        use ``is not None`` to decide whether to enqueue, which keeps
        "no message" distinct from a hypothetical empty token list.
        """
        if get_tp_size() <= 1:
            return next_tokens

        self._ensure_tp_broadcast_state()
        src = self._tp_bcast_src_rank
        # Tokens are small; reuse the dedicated IPC group so we don't
        # serialize behind forward AR (same reasoning as
        # :meth:`broadcast_input_to_tp`).
        tp_group = get_ipc_tp_group()
        device = torch.device("cuda", torch.cuda.current_device())

        # Phase 1: broadcast the length so non-source ranks know how
        # much to allocate.
        len_gpu = self._tp_bcast_flag_gpu  # reused 1-element scratch
        if get_rank() == src:
            len_gpu.fill_(len(next_tokens) if next_tokens else 0)
        dist.broadcast(len_gpu, src=src, group=tp_group)
        self._tp_bcast_flag_cpu.copy_(len_gpu, non_blocking=False)
        n = int(self._tp_bcast_flag_cpu.item())
        if n == 0:
            return None

        # Phase 2: broadcast the int64 token tensor of length n.
        tok_tensor = torch.empty(n, dtype=torch.long, device=device)
        if get_rank() == src:
            tok_tensor.copy_(
                torch.as_tensor(next_tokens, dtype=torch.long, device="cpu"),
                non_blocking=True,
            )
        dist.broadcast(tok_tensor, src=src, group=tp_group)
        return tok_tensor.cpu().tolist()

    def recv_schedule_payload(self) -> Optional[SchedulePayload]:
        if self.schedule_socket.poll(timeout=0) != 0:
            payload = self.schedule_socket.recv_pyobj()
            assert isinstance(payload, SchedulePayload), (
                f"unexpected schedule payload {type(payload).__name__!r}: "
                "all schedule sends should go through send_schedule_payload"
            )
            return payload
        return None

    def send_ipc_package(self, ipc_package):
        self.request_socket.send_pyobj(ipc_package)

    def recv_ipc_package(self):
        if self.request_socket.poll(timeout=0) != 0:
            ipc_package = self.request_socket.recv_pyobj()
            return ipc_package
        else:
            return None
