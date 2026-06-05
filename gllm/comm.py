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
        # Encoder-disaggregation control events (gllm.disagg.lm_manager.
        # DisaggEvents) generated authoritatively on TP0 and fanned out with the
        # input so every PP=0 TP rank admits / advances disagg seqs in the same
        # iteration. ``None`` on every non-disagg iteration / the monolith.
        self.disagg_events = None
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
        # frontend and are fanned out to PP=0 TP peers via zmq PUSH/PULL
        # (:meth:`broadcast_input_to_tp`). The earlier NCCL flag-broadcast
        # implementation contended with the model's per-layer all-reduce
        # for NVLink and inflated decode-AR tail latency by ~70 ms /
        # decode-heavy profile; profile shows ~1 % of decode iters had
        # a 5-9 ms NCCL-AR spike that disappears with the zmq path
        # since zmq stays on the CPU and never touches NVLink.
        # Tokens still funnel through rank 0 (output_rank still uses a
        # single PULL into rank 0); rank 0 NCCL-broadcasts the result
        # within the PP-0 TP group via :meth:`broadcast_tokens_to_tp`.
        #
        # For PP=1 (overlap path) the per-column structure collapses
        # cleanly: ``schedule_other_sockets`` is empty, no tokens leg,
        # and the per-iter input zmq fan-out alone handles every
        # cross-rank message.
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
        #
        # On top of the column-driver fan-out (rank 0 -> PP-other in
        # the same column) we also need a *PP-0 TP fan-out* so that
        # rank 0 can ship the front-end :class:`IPCPackage` to every
        # peer column driver (PP=0 TP=k for k>0). That used to ride
        # NCCL but now goes over zmq for the NVLink-contention reason
        # documented above. ``_input_tp_sockets`` is the rank-0 send
        # side; ``_input_tp_recv_socket`` is the per-peer recv side.
        self._input_tp_sockets: List[zmq.Socket] = []
        self._input_tp_recv_socket: Optional[zmq.Socket] = None
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
            # PP-0 TP fan-out (replaces the NCCL flag broadcast in
            # :meth:`broadcast_input_to_tp`). For tp_size==1 the
            # broadcast short-circuits and we never touch any of
            # these sockets, so skip the setup entirely.
            if tp_size > 1:
                if rank == 0:
                    for peer_tp in range(1, tp_size):
                        peer_rank = peer_tp  # PP=0 TP=peer_tp == global rank peer_tp
                        if self.launch_mode == "normal":
                            socket = make_socket(
                                self.ctx,
                                f"{self.schedule_path}_tpinput_{peer_rank}",
                                zmq.PUSH,
                            )
                        else:
                            # Skip the world_size + token (==world_size)
                            # offsets used above; +1 keeps a free slot
                            # in case a future feature wants
                            # port_base + world_size + 1 - peer_rank
                            # encoded somewhere.
                            port_each = (
                                self.port_base + get_world_size() + 1 + peer_rank
                            )
                            send_obj_list([port_each], peer_rank)
                            addr_each = [None]
                            recv_obj_list(addr_each, peer_rank)
                            socket = make_socket(
                                self.ctx,
                                f"tcp://{addr_each[0]}:{port_each}",
                                zmq.PUSH,
                            )
                        self._input_tp_sockets.append(socket)
                else:
                    if self.launch_mode == "normal":
                        self._input_tp_recv_socket = make_socket(
                            self.ctx,
                            f"{self.schedule_path}_tpinput_{rank}",
                            zmq.PULL,
                        )
                    else:
                        port_input = [None]
                        recv_obj_list(port_input, 0)
                        send_obj_list([self.host_addr], 0)
                        self._input_tp_recv_socket = make_socket(
                            self.ctx,
                            f"tcp://{self.host_addr}:{port_input[0]}",
                            zmq.PULL,
                        )
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
    # zmq-backed PP=0 TP-group input broadcast (column-driver path)
    # ------------------------------------------------------------------
    #
    # Every PP=0 TP rank runs its own scheduler (column driver) and
    # therefore needs the exact same stream of front-end => worker
    # messages (new requests, aborts, control commands). Rank-0 polls
    # the front-end zmq socket, aggregates whatever is waiting into a
    # single :class:`IPCPackage`, and ships that to its peer column
    # drivers via the dedicated zmq fan-out set up in :meth:`init`.
    #
    # The earlier implementation used a NCCL ``broadcast`` on the
    # dedicated IPC group ( ``_IPC_TP_GROUP`` ): cheap on average (~5
    # us) but it shared NVLink with the model's per-layer all-reduce,
    # which forced occasional 5-9 ms tail spikes when the broadcast
    # collided with a forward-path AR. The zmq fan-out below stays on
    # the CPU and never touches NVLink, eliminating that contention.
    #
    # Determinism rule (unchanged from the NCCL version): every PP=0
    # TP rank MUST call this every iteration in the same order. Rank
    # 0 sends EXACTLY ONE pyobj per iter (possibly ``None``); peers
    # block-recv exactly once. Skipping the call would desync the
    # column-driver schedulers across TP ranks.

    def _ensure_tp_broadcast_state(self) -> None:
        """Lazy init of state used by :meth:`broadcast_tokens_to_tp`.

        ``broadcast_tokens_to_tp`` still rides NCCL (it's only used
        on the PP>1 path with one call per iter, where forward-path
        AR contention is irrelevant), so we keep the small CUDA
        scratch tensors used by the length+payload protocol there.
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
        """Rank-0-driven fan-out of an :class:`IPCPackage` to PP=0 TP peers.

        Single-shot zmq PUSH/PULL: rank 0 ``send_pyobj`` once per
        peer, every other PP=0 TP rank ``recv_pyobj``s exactly once.
        ``ipc_package`` may be ``None`` -- that's the steady-state
        decode case where the front-end has nothing pending; rank 0
        still sends the ``None`` so peers stay in lockstep without
        any per-iter NCCL traffic.

        Caller contract:
        * Must be called on every PP=0 TP rank every iteration in the
          same order (lock-step with the schedule loop).
        * Must NOT be called from PP-other ranks (their column gets
          updates over :meth:`send_schedule_payload` instead).

        See module-level comment above for why this no longer rides
        NCCL.
        """
        if get_tp_size() <= 1:
            return ipc_package

        if get_rank() == 0:
            # ``send_pyobj`` pickles + sends; we accept the per-peer
            # pickle cost (3 pickles for tp_size=4) because pickling
            # ``None`` or an empty IPCPackage is < 1 us each and the
            # send itself is ~1 us over ipc://. zmq sockets are not
            # thread-safe across threads, but each socket is only
            # touched from the main worker thread here so the direct
            # ``send_pyobj`` is safe (no need for the
            # ``_get_sender`` background-thread pattern that the
            # PP-other schedule fan-out uses).
            for sock in self._input_tp_sockets:
                sock.send_pyobj(ipc_package)
            return ipc_package

        # PP=0 TP=k>0: blocking recv. Rank 0 sends every iter so this
        # is bounded by zmq ipc round-trip latency (~1 us once the
        # socket is warm).
        return self._input_tp_recv_socket.recv_pyobj()

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
