import pickle
import queue
import threading
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import zmq

from gllm.distributed.parallel_state import (
    get_ipc_tp_group,
    get_output_rank,
    get_pp_rank,
    get_pp_size,
    get_rank,
    get_tp_rank,
    get_tp_size,
    is_output_rank,
    recv_obj_list,
    send_obj_list,
)
from gllm.runtime.sequence import GenerationSequence
from gllm.scheduling.distributed import SchedulePayload
from gllm.utils import make_pull_bind, make_pull_random, make_socket

_SHUTDOWN = object()  # sentinel pushed onto a sender queue to drain it

# TCP keepalive for decoupled (tcp://) transports: a half-open connection
# (machine/network partition -- no FIN, hours-long default TCP timeout)
# otherwise keeps stealing a share of the worker's load-balanced output
# PUSH into dead kernel buffers, and the replacement frontend's requests
# hang with no error. With keepalive the kernel tears the leg down
# (~idle + retries) so the partition is detected in minutes, not hours.
_TCP_KEEPALIVE_IDLE = 60   # seconds of idle before probing
_TCP_KEEPALIVE_INTERVAL = 10
_TCP_KEEPALIVE_COUNT = 3


def _apply_tcp_keepalive(socket) -> None:
    """Enable kernel TCP keepalive on a tcp:// socket (no-op for ipc://)."""
    try:
        socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, _TCP_KEEPALIVE_IDLE)
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTERVAL, _TCP_KEEPALIVE_INTERVAL)
        socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, _TCP_KEEPALIVE_COUNT)
    except (zmq.ZMQError, AttributeError):
        pass  # non-tcp transport or libzmq without the options


class IPCPackage:
    """One tick of frontend<->worker traffic, in a single pickle.

    FRONTEND SESSION PROTOCOL (decoupled deployment)
    ================================================

    Four fields carry session identity across the wire. THREE of them
    are STAMP LISTS that must stay POSITIONALLY aligned with their id
    list for the entire lifetime of the package (construction, drain
    merges, pickling, translation):

    ============================  ========  =====================  ======
    field                         direction aligns with            set by
    ============================  ========  =====================  ======
    ``seq.frontend_session``      req       (per seq object)       frontend
    ``abort_sessions``            req       ``abort_ids``          frontend
    ``sessions``                  output    ``act_schedule_ids``   worker
    ``free_sessions``             output    ``free_ids``           worker
    ============================  ========  =====================  ======

    * ``None`` on a stamp field means LEGACY/unstamped (monolith path);
      an empty list is only legal alongside an empty id list.
    * A batch may legally carry the SAME client id more than once
      (once per session), so stamps must never be collapsed into a
      dict keyed by id -- positional order is the contract.
    * Helpers below (:meth:`merge_aligned`, :meth:`abort_stamps_valid`,
      :meth:`output_stamps_valid`) are the only sanctioned places to
      read or extend these pairs; both sides of the wire route through
      them so the alignment invariant has one definition.
    * ``seq.frontend_session`` rides inside each pickled
      :class:`~gllm.runtime.sequence.GenerationSequence` and is
      echoed back by the worker (``client_seq_id`` / translate) -- it
      is not a package field, which is why it gets no package-level
      helper here.

    On the monolith path every stamp field stays ``None`` and the
    helpers below reduce to no-ops / legacy answers, so the protocol
    costs nothing when unused.
    """

    def __init__(self, schedule_lists: List[GenerationSequence]):
        # front-end => worker
        self.log = True
        self.schedule_lists = schedule_lists
        self.abort_ids = []  # seq_ids (CLIENT ids) to abort
        # Frontend session stamps aligned with ``abort_ids``: which session's
        # request an abort targets. The standalone worker remaps client ids
        # to internal ids per session, so a bare id names TWO live requests
        # once a frontend restarts (both pools start at 0); the worker
        # resolves each abort through (stamp, id) and ignores stamps it does
        # not know. Monolith seqs are unstamped (None entries) and resolve
        # unconditionally -- see Worker._route_frontend_aborts.
        self.abort_sessions = None
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
        self.request_errors = {}  # terminal per-request capacity errors
        self.act_schedule_ids = []
        self.next_tokens = []
        # Per-token logprobs aligned with ``next_tokens`` (one entry per acted
        # seq). Each entry is ``(sampled_logprob, top_ids, top_vals)`` or
        # ``None`` when the seq did not request logprobs. Empty on the common
        # (no-logprobs) path so it adds nothing to the pickled payload.
        self.logprobs = []
        # Per-REQUEST frontend session epochs, aligned with
        # ``act_schedule_ids`` (LLM.frontend_epoch echoed back by the worker
        # from each seq's ``frontend_session``). Session identity must travel
        # with the REQUEST, not with "last package seen": a tick with no new
        # input must not reset it, and a late output of a dead session's
        # request must not inherit the new session's stamp. ``None``/empty
        # means legacy (unstamped) workers.
        self.sessions = None
        # Same idea for ``free_ids`` (aligned with it): the ids being freed
        # this tick. Free-only rows -- aborts, finished-before-acted, capacity
        # errors -- frequently lack a live seq at stamp time, so their entry
        # can be ``None``; the standalone frontend fails closed on those.
        self.free_sessions = None
        # Prompt-token logprobs, keyed by seq_id, sent once when a seq finishes
        # prefill. Each value is the seq's ``prompt_logprobs_data`` list
        # (per prompt position: ``None`` or ``(token_id, logprob, ids, vals)``).
        self.prompt_logprobs = {}

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

    # ------------------------------------------------------------------
    # Frontend session protocol: alignment helpers (see class docstring)
    # ------------------------------------------------------------------

    @staticmethod
    def _backfilled(stamps, n_ids):
        """Stamp list of exactly ``n_ids`` entries, legacy None-filled.

        ``None`` input becomes an all-None list (legacy); a list is
        trusted to be well-formed (the sender's own helper built it).
        """
        if stamps is None:
            return [None] * n_ids
        return list(stamps)

    def merge_aligned(self, other: "IPCPackage") -> None:
        """Fold *other*'s request-dir content into self, PRESERVING the
        ``abort_ids`` / ``abort_sessions`` positional alignment.

        This is the ONLY sanctioned way to merge drained request
        packages: merging the bare id lists would silently drop the
        session stamps and downgrade the aggregate to legacy routing
        (aborting EVERY same-id request across sessions). Stamps from
        unstamped packages are backfilled with None so the merged list
        is always exactly as long as the merged id list.
        """
        self.schedule_lists.extend(other.schedule_lists)
        n_in = len(other.abort_ids)
        in_stamps = self._backfilled(other.abort_sessions, n_in)
        if self.abort_sessions is None:
            base = [None] * len(self.abort_ids)
        else:
            base = list(self.abort_sessions)
        self.abort_ids = self.abort_ids + other.abort_ids
        self.abort_sessions = base + in_stamps

    def abort_stamps_valid(self) -> bool:
        """True iff ``abort_sessions`` is absent (legacy/monolith) or
        exactly as long as ``abort_ids``. Callers (frontend drain) use
        this to fail closed on malformed packets instead of guessing
        which session an abort belongs to."""
        if self.abort_sessions is None:
            return True
        return len(self.abort_sessions) == len(self.abort_ids)

    def output_stamps_valid(self) -> bool:
        """True iff the output stamp lists are absent (legacy/monolith)
        or exactly as long as their id lists (``sessions`` vs
        ``act_schedule_ids``, ``free_sessions`` vs ``free_ids``). The
        standalone frontend applies this before trusting any row."""
        for stamps, ids in (
            (self.sessions, self.act_schedule_ids),
            (self.free_sessions, self.free_ids),
        ):
            if stamps is None:
                continue
            if len(stamps) != len(ids):
                return False
        return True

    def act_session_at(
        self, idx: int, our_epoch, valid: bool = True, legacy_ok: bool = True
    ) -> bool:
        """Whether row ``idx`` of ``act_schedule_ids`` belongs to THIS
        frontend incarnation.

        This is the single sanctioned frontend-side row gate (used by
        :meth:`LLM._apply_ipc_package`). Semantics:

        * ``valid`` False -> every row is foreign (the caller checked
          :meth:`output_stamps_valid` once per package, O(1), and a
          malformed packet fails closed in one shot);
        * no ``sessions`` list at all -> LEGACY packet: ours only when
          ``legacy_ok``. The MONOLITH frontend passes ``legacy_ok=True``
          (its worker never remaps ids -- nothing to filter); the
          STANDALONE frontend passes ``legacy_ok=False`` because a
          stamp-less row from its fleet names no session and MUST NOT
          touch a request of the same bare id (fail closed);
        * otherwise the row is ours iff ``sessions[idx] == our_epoch``;
          an index past the end of the list is unconditionally foreign.

        A batch may carry the same client id twice (once per session),
        so the check is strictly positional -- never dict-by-id.
        """
        if not valid:
            return False
        stamps = self.sessions
        if stamps is None:
            return legacy_ok
        if idx >= len(stamps):
            return False
        return stamps[idx] == our_epoch

    def free_session_at(
        self, idx: int, our_epoch, valid: bool = True, legacy_ok: bool = True
    ) -> bool:
        """Same as :meth:`act_session_at` for ``free_ids`` rows
        (``free_sessions``). Fail-closed on a malformed list (either
        via ``valid`` or a per-row length miss); a stamp-less list is
        legacy -- honored only when ``legacy_ok`` (monolith)."""
        if not valid:
            return False
        stamps = self.free_sessions
        if stamps is None:
            return legacy_ok
        if idx >= len(stamps):
            return False
        return stamps[idx] == our_epoch


class zmqComm:
    def __init__(
        self,
        host_addr,
        launch_mode,
        master_addr,
        schedule_path,
        output_path,
        token_path,
        frontend=False,
        dp_rank=0,
        dp_size=1,
        standalone_remote=False,
        standalone_worker=False,
    ):
        self.host_addr = host_addr
        self.master_addr = master_addr
        self.launch_mode = launch_mode
        self.schedule_path = schedule_path
        self.output_path = output_path
        self.token_path = token_path
        self.frontend = frontend
        # Data-parallel (DP) attention. When ``dp_size > 1`` the engine runs
        # ``dp_size`` independent full-model replicas. Each replica binds its
        # *own* request PULL socket (``schedule_path_dp{r}``) and the frontend
        # keeps one PUSH socket per replica so it can round-robin new requests
        # and broadcast aborts/control. Outputs still fan-in to the single
        # frontend output PULL. ``dp_rank`` selects this replica's request path.
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        # Decoupled deployment: a standalone worker fleet binds its
        # frontend-facing PULL sockets on remote-reachable paths (tcp:// when
        # --worker-transport-base-port is set, ipc:// otherwise). The ipc
        # binder is the worker itself (launch_mode 'normal' worker branch
        # below); the frontend only ever *connects*. ``standalone_remote`` is
        # set by the frontend so it knows the schedule path is remote and
        # must not be treated as a local ipc path it binds.
        self.standalone_remote = standalone_remote
        self.standalone_worker = standalone_worker
        # Worker-side output hook installed by the Worker (None on
        # frontends): translates internal seq ids back to the dispatching
        # frontend's client ids and attaches the per-row session stamps
        # before the package hits the wire (see send_output).
        self._output_committer = None
        # Worker-side (standalone) per-seq identity bookkeeping, owned by
        # the WORKER's comm (never the frontend's): registered at
        # ADMISSION (Worker._remap_client_ids), before the scheduler can
        # free the seq (first token == EOS / max_tokens=1 requests never
        # reach a live-queue lookup); reclaimed one output tick after
        # their terminal row is translated (see
        # Worker.translate_output_for_frontend). Keyed by INTERNAL id.
        self._session_identity = {}
        self._identity_reclaim = set()

    def init(self):
        self.ctx = zmq.Context()
        # Persistent zmq-sender threads keyed by socket. See ``_get_sender``
        # for why we avoid the prior fresh-thread-per-send pattern.
        self._senders: Dict["zmq.Socket", "queue.SimpleQueue"] = {}
        self._sender_threads: Dict["zmq.Socket", "threading.Thread"] = {}

        if self.frontend and self.standalone_remote:
            # Decoupled deployment: the standalone worker fleet already bound
            # these PULL endpoints (see gllm.engine.llm.LLM._publish_worker_endpoint);
            # the frontend only connects. Same socket roles as the monolith
            # frontend (PUSH schedule, PULL output); the token path is unused
            # by the frontend (worker-internal) but is created for symmetry
            # and future control channels.
            # schedule: the worker BINDs its request PULL on this
            # (remote-reachable) path; the frontend PUSH connects to it.
            self.request_socket = make_socket(self.ctx, self.schedule_path, zmq.PUSH)
            # output: the worker BINDs its output PUSH on the advertised path
            # (PUSH connects by default -- wrong direction for a remote
            # frontend), so THIS side must PULL-connect. A tcp:// PULL cannot
            # go through make_socket (asserts by design, routing binders to
            # make_pull_bind); ipc:// binds locally as before.
            if self.output_path.startswith("tcp://"):
                self.output_socket = self.ctx.socket(zmq.PULL)
                self.output_socket.connect(self.output_path)
                self.output_socket.setsockopt(zmq.RCVHWM, 0)
                self.output_socket.setsockopt(
                    zmq.RCVBUF, int(0.5 * 1024**3)
                )
                _apply_tcp_keepalive(self.output_socket)
            else:
                self.output_socket = make_socket(self.ctx, self.output_path, zmq.PULL)
            self.token_socket = None
            return

        if self.frontend:  # front-end process
            if self.dp_size > 1:
                # One PUSH per DP replica (connects to that replica's request
                # PULL bind). Enables explicit round-robin of new requests and
                # broadcast of aborts/control across replicas. ``request_socket``
                # aliases replica 0 for any legacy single-socket call site.
                self.request_sockets = [
                    make_socket(self.ctx, f"{self.schedule_path}_dp{r}", zmq.PUSH)
                    for r in range(self.dp_size)
                ]
                self.request_socket = self.request_sockets[0]
            else:
                self.request_socket = make_socket(
                    self.ctx, self.schedule_path, zmq.PUSH
                )
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

        # The frontend driver is PP=0 TP=0. Non-DP: that's global rank 0 (the
        # sole poller, unchanged). DP+EP: each DP group's tp_rank==0 is a
        # driver, so there are ``dp_size`` drivers, each binding a *distinct*
        # request path (``schedule_path_dp{dp_rank}``); the frontend keeps one
        # PUSH per DP group pointed at these paths (see the frontend branch).
        if pp_rank == 0 and tp_rank == 0:
            req_path = self.schedule_path
            if self.dp_size > 1:
                req_path = f"{self.schedule_path}_dp{self.dp_rank}"
            # tcp:// request endpoints are bound (standalone worker exposing a
            # fixed, remotely reachable port); ipc:// goes through make_socket
            # (bind + buffer tuning).
            if req_path.startswith("tcp://"):
                self.request_socket = make_pull_bind(self.ctx, req_path)
            else:
                self.request_socket = make_socket(self.ctx, req_path, zmq.PULL)
            # output: this side must BIND (the remote frontend PULL-connects
            # to the advertised address); make_socket's PUSH CONNECTS, which
            # is only correct for ipc:// where the local frontend binds.
            if self.output_path.startswith("tcp://"):
                push = self.ctx.socket(zmq.PUSH)
                push.bind(self.output_path)
                push.setsockopt(zmq.SNDHWM, 0)
                push.setsockopt(zmq.SNDBUF, int(0.5 * 1024**3))
                _apply_tcp_keepalive(push)
                self.output_socket = push
            else:
                self.output_socket = make_socket(self.ctx, self.output_path, zmq.PUSH)
            if pp_size > 1:
                # last-stage output_rank => this column's PP=0 driver : next
                # tokens (single PULL, broadcast inside the stage-0 TP group
                # below). DP+PP: each DP group is an independent replica, so the
                # token leg is keyed per DP group (``token_path_dp{d}``) -- group
                # ``d``'s last stage pushes to group ``d``'s PP=0 driver.
                tok_path = self.token_path
                if self.dp_size > 1:
                    tok_path = f"{self.token_path}_dp{self.dp_rank}"
                if self.launch_mode == "normal":
                    self.token_socket = make_socket(
                        self.ctx, tok_path, zmq.PULL
                    )
                else:
                    self.token_socket, port_token = make_pull_random(
                        self.ctx, self.master_addr
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
            # Ranks per pipeline stage. DP+PP: a stage is a ``dp x tp`` grid, so
            # this column's PP-other rank at stage ``pp`` sits ``pp * stage_size``
            # above this driver's global rank (same ``(dp, tp)`` position).
            stage_size = self.dp_size * tp_size
            if pp_size > 1:
                if self.launch_mode == "normal":
                    for pp in range(1, pp_size):
                        target_rank = pp * stage_size + rank
                        socket = make_socket(
                            self.ctx,
                            f"{self.schedule_path}_{target_rank}",
                            zmq.PUSH,
                        )
                        self.schedule_other_sockets.append(socket)
                else:
                    for pp in range(1, pp_size):
                        target_rank = pp * stage_size + rank
                        # The PULL binder (target) picks a free port and sends
                        # back its (addr, port); we just connect to it.
                        info = [None, None]
                        recv_obj_list(info, target_rank)
                        addr_each, port_each = info
                        socket = make_socket(
                            self.ctx,
                            f"tcp://{addr_each}:{port_each}",
                            zmq.PUSH,
                        )
                        self.schedule_other_sockets.append(socket)
            # PP-0 TP fan-out (replaces the NCCL flag broadcast in
            # :meth:`broadcast_input_to_tp`). For tp_size==1 the
            # broadcast short-circuits and we never touch any of
            # these sockets, so skip the setup entirely.
            if tp_size > 1:
                if tp_rank == 0:
                    # Each DP group's tp_rank==0 fans out to *its own* TP peers
                    # (global ranks ``rank+1 .. rank+tp_size-1``). For the
                    # non-DP case ``rank == 0`` so this is the original
                    # ``peer_rank = peer_tp``; for DP+TP each group's driver
                    # targets its own peers (global ranks are unique, so the
                    # ``_tpinput_{peer_rank}`` paths never collide across groups).
                    for peer_tp in range(1, tp_size):
                        peer_rank = rank + peer_tp
                        if self.launch_mode == "normal":
                            socket = make_socket(
                                self.ctx,
                                f"{self.schedule_path}_tpinput_{peer_rank}",
                                zmq.PUSH,
                            )
                        else:
                            # The PULL peer picks a free port and sends back its
                            # (addr, port); we just connect to it.
                            info = [None, None]
                            recv_obj_list(info, peer_rank)
                            addr_each, port_each = info
                            socket = make_socket(
                                self.ctx,
                                f"tcp://{addr_each}:{port_each}",
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
                        # Bind a free port and hand (addr, port) to rank 0.
                        self._input_tp_recv_socket, port_input = make_pull_random(
                            self.ctx, self.host_addr
                        )
                        send_obj_list([self.host_addr, port_input], 0)
        else:
            # PP-other rank: pull from its own column driver
            # (rank ``tp_rank`` on PP=0).
            if self.launch_mode == "normal":
                self.schedule_socket = make_socket(
                    self.ctx, f"{self.schedule_path}_{rank}", zmq.PULL
                )
            else:
                # Bind a free port and hand (addr, port) to the column driver.
                self.schedule_socket, port_schedule = make_pull_random(
                    self.ctx, self.host_addr
                )
                send_obj_list([self.host_addr, port_schedule], tp_rank)

        if is_output_rank() and pp_size != 1:
            # output_rank (last-PP TP=0) => this column's PP=0 driver : tokens.
            # DP+PP keys the leg per DP group (see the PP=0 PULL bind above).
            tok_path = self.token_path
            if self.dp_size > 1:
                tok_path = f"{self.token_path}_dp{self.dp_rank}"
            if self.launch_mode == "normal":
                self.token_socket = make_socket(self.ctx, tok_path, zmq.PUSH)
            else:
                port_token = [None]
                recv_obj_list(port_token, 0)
                self.token_socket = make_socket(
                    self.ctx,
                    f"tcp://{self.master_addr}:{port_token[0]}",
                    zmq.PUSH,
                )

    def send_tokens(self, tokens):
        # ``tokens`` is either a plain ``list`` of token ids or a
        # ``(tokens, gen_logprobs, prompt_logprobs)`` tuple (PP>1 logprobs ride
        # the same socket back to rank 0). ``recv_tokens`` returns it verbatim
        # for the caller to unpack.
        assert type(tokens) in (list, tuple)
        self.token_socket.send_pyobj(tokens)

    def recv_tokens(self, block: bool = False):
        if block or self.token_socket.poll(timeout=0) != 0:
            next_tokens = self.token_socket.recv_pyobj()
            return next_tokens
        else:
            return None

    def send_output(self, output):
        # Session stamps ride the OUTPUT package (see Worker
        # translate_output_for_frontend / stamp_output_sessions): the worker
        # translates its internal ids back to client ids and attaches each
        # row's frontend session, so the frontend filters by stamp. No-op
        # on the monolith (no ``_output_committer`` installed).
        commit = getattr(self, "_output_committer", None)
        if commit is not None:
            commit(output)
        self.output_socket.send_pyobj(output)

    def recv_output(self):
        if self.output_socket.poll(timeout=0) != 0:
            output = self.output_socket.recv_pyobj()
            return output
        else:
            return None

    def frontend_gone(self) -> bool:
        """True iff the FRONTEND side of this transport has gone away.

        The worker's output leg is a PUSH into the frontend's PULL. A
        CRASHED frontend destroys its PULL, so the kernel tears the PUSH
        leg down and the socket starts failing immediately; a CLEAN
        frontend death (SIGKILL) leaves the leg half-open until keepalive
        (see :func:`_apply_tcp_keepalive`) tears it down in ~90s. Either
        way the leg's send fileno eventually drops below 0, which this
        detects. Only meaningful for a standalone *worker* comm (its output
        socket is the frontend-facing PUSH); False everywhere else.
        """
        if not getattr(self, "standalone_worker", False):
            return False  # frontends / monolith: no frontend-facing PUSH leg
        sock = getattr(self, "output_socket", None)
        if sock is None:
            return True  # torn down entirely
        try:
            return sock.getsockopt(zmq.SNDFILENO) < 0
        except Exception:
            return False

    def close(self):
        """Tear down every sender thread and socket, then terminate the ctx.

        Idempotent; intended for the decoupled-deployment paths where a
        frontend may reconnect in-process (or a test process must exit
        cleanly) and a lingering zmq I/O thread holding open ipc://
        connections would otherwise block subsequent CUDA init or process
        shutdown.
        """
        for sock, q in list(getattr(self, "_senders", {}).items()):
            try:
                q.put(_SHUTDOWN)
            except Exception:
                pass
        # Join the sender threads before the primary sockets close. A
        # sender parked on a LIVE peer drains on _SHUTDOWN and exits
        # quickly (bounded join); a sender parked in send_pyobj on a DEAD
        # peer is unstuck only by the socket close below, so this first
        # pass yields fast (<=~1.0s) and a second short pass reaps it
        # after the sockets go. Bounding the first join avoids paying
        # 2x the full timeout for a single dead-peer sender. Sender
        # threads are daemon, so none can strand the process; this just
        # ensures a live sender never outlives the ctx.term() it would
        # block on.
        sender_threads = getattr(self, "_sender_threads", {})
        for sock, t in list(sender_threads.items()):
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
        self._senders.clear()
        self._sender_threads.clear()
        for attr in ("request_socket", "output_socket", "token_socket"):
            sock = getattr(self, attr, None)
            if sock is not None:
                try:
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.close(linger=0)
                except Exception:
                    pass
                setattr(self, attr, None)
        # ``request_sockets`` is only present on multi-DP frontends; the
        # attribute may never have been created.
        for sock in list(getattr(self, "request_sockets", None) or []):
            try:
                sock.setsockopt(zmq.LINGER, 0)
                sock.close(linger=0)
            except Exception:
                pass
        if hasattr(self, "request_sockets"):
            self.request_sockets = []
        # Second (short) rejoin: the socket closes above unstick any sender
        # still parked in send_pyobj; reap it before ctx.term() so it cannot
        # hold a socket the term would block on.
        for sock, t in list(sender_threads.items()):
            if t.is_alive():
                try:
                    t.join(timeout=1.0)
                except Exception:
                    pass
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            try:
                ctx.term()
            except Exception:
                pass
            self.ctx = None

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
        self._sender_threads[socket] = t
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
            # Source rank inside the TP group: tp_rank == 0 of this TP
            # subgroup. ``get_rank() - get_tp_rank()`` is that group's tp0
            # global rank -- equal to ``pp_rank * tp_size`` for the non-DP case
            # and to ``dp_rank * tp_size`` for DP+TP (each DP group broadcasts
            # its own sampled tokens within its own TP subgroup).
            self._tp_bcast_src_rank = get_rank() - get_tp_rank()

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

        if get_tp_rank() == 0:
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

        For the overlap path this method is unnecessary: sampled tokens
        already travel GPU-side (within TP and, for PP>1, through the FutureMap
        feedback path), and every PP-0 TP rank D2H-copies its result locally.

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

    def send_ipc_package_nonblocking(self, ipc_package, timeout=1.0) -> bool:
        """Send without the ability to wedge the caller's thread.

        A blocking ``send_pyobj`` on a PUSH socket with a large SNDBUF parks
        the whole payload in a zero-copy buffer and only returns once it is
        fully staged -- with no connected receiver (worker fleet down) that
        is effectively forever. The standalone-frontend schedule loop runs on
        the *single* engine-IO thread, so a parked send also blocks the
        liveness check, reconnect and /health probe behind it.

        Sends with a wall-clock bound instead: True when the payload was
        accepted, False on timeout (caller decides whether to retry/queue).
        """
        data = pickle.dumps(ipc_package)
        deadline = time.monotonic() + timeout
        while True:
            # PUSH readiness is about the SEND buffer: poll POLLOUT explicitly
            # (the default poll mask is POLLIN, which a PUSH socket never
            # raises -- even a healthy, writable connection would time out).
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                return False
            if self.request_socket.poll(timeout=remaining_ms, flags=zmq.POLLOUT):
                try:
                    self.request_socket.send(data, zmq.NOBLOCK)
                    return True
                except zmq.ZMQError:
                    pass  # buffer filled mid-flush; re-poll
            else:
                if time.monotonic() >= deadline:
                    return False

    def drain_request_buffer(self):
        """Best-effort drop of anything left staged on the request PUSH.

        Called after the worker transport is torn down (fleet restart): the
        old socket's buffers may hold payloads addressed to a fleet that is
        gone, and a brand-new socket on the same endpoint must start clean.
        ZeroMQ buffers are per-socket, so this is belt-and-braces for the
        local case where an OS-level endpoint could replay them. Note the
        request leg is a PUSH: PUSH sockets cannot recv (it would raise
        EFSM / ZMQERRNO), so in practice the loop exits on the first
        iteration and this is a defensive no-op kept for future socket-type
        changes.
        """
        sock = getattr(self, "request_socket", None)
        if sock is None:
            return
        while True:
            try:
                sock.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
    def send_ipc_package_to_dp(self, ipc_package, dp_index):
        """Send a package to one DP replica (frontend, dp_size > 1 only)."""
        self.request_sockets[dp_index].send_pyobj(ipc_package)

    def broadcast_ipc_package_to_dp(self, ipc_package):
        """Send a copy of the package to every DP replica (aborts/control)."""
        for sock in self.request_sockets:
            sock.send_pyobj(ipc_package)

    def recv_ipc_package(self):
        if self.request_socket.poll(timeout=0) != 0:
            ipc_package = self.request_socket.recv_pyobj()
            return ipc_package
        else:
            return None
