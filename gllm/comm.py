import queue
import threading
from typing import Dict, List, Optional, Tuple

import torch
import zmq

from gllm.dist_utils import (
    get_output_rank,
    get_pp_size,
    get_rank,
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
        # worker => front-end
        self.free_ids = []  # seq_ids to free
        self.act_schedule_ids = []
        self.next_tokens = []


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
        else:  # worker process
            if get_rank() == 0:
                # front-end => rank 0
                self.request_socket = make_socket(
                    self.ctx, self.schedule_path, zmq.PULL
                )
                # rank 0 => front-end
                self.output_socket = make_socket(self.ctx, self.output_path, zmq.PUSH)

                if get_world_size() != 1:
                    self.schedule_first_pp_sockets: List[zmq.Socket] = []
                    self.schedule_other_sockets: List[zmq.Socket] = []
                    if self.launch_mode == "normal":
                        # rank 0 => other ranks : batched seqs
                        for rank in range(1, get_world_size()):
                            socket = make_socket(
                                self.ctx, f"{self.schedule_path}_{rank}", zmq.PUSH
                            )
                            if rank < get_tp_size():
                                self.schedule_first_pp_sockets.append(socket)
                            else:
                                self.schedule_other_sockets.append(socket)
                        # last rank => rank 0 : next tokens
                        self.token_socket = make_socket(
                            self.ctx, self.token_path, zmq.PULL
                        )
                    else:
                        # rank 0 => other ranks : batched seqs
                        self.schedule_sockets = []
                        for rank in range(1, get_world_size()):
                            port_each = self.port_base + rank
                            send_obj_list([port_each], rank)
                            addr_each = [None]
                            recv_obj_list(addr_each, rank)
                            socket = make_socket(
                                self.ctx, f"tcp://{addr_each[0]}:{port_each}", zmq.PUSH
                            )
                            if rank < get_tp_size():
                                self.schedule_first_pp_sockets.append(socket)
                            else:
                                self.schedule_other_sockets.append(socket)
                        # output rank => rank 0 : next tokens
                        port_token = self.port_base + get_world_size()
                        self.token_socket = make_socket(
                            self.ctx, f"tcp://{self.master_addr}:{port_token}", zmq.PULL
                        )
                        send_obj_list([port_token], get_output_rank())
            else:
                # rank 0 => other ranks : batched seqs
                if self.launch_mode == "normal":
                    self.schedule_socket = make_socket(
                        self.ctx, f"{self.schedule_path}_{get_rank()}", zmq.PULL
                    )
                else:
                    port_schedule = [None]
                    recv_obj_list(port_schedule, 0)
                    send_obj_list([self.host_addr], 0)
                    self.schedule_socket = make_socket(
                        self.ctx, f"tcp://{self.host_addr}:{port_schedule[0]}", zmq.PULL
                    )

            if is_output_rank() and get_pp_size() != 1:
                # output rank => rank 0 : next tokens
                if self.launch_mode == "normal":
                    self.token_socket = make_socket(self.ctx, self.token_path, zmq.PUSH)
                else:
                    port_token = [None]
                    recv_obj_list(port_token, 0)
                    self.token_socket = make_socket(
                        self.ctx, f"tcp://{self.master_addr}:{port_token[0]}", zmq.PUSH
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

    def send_schedule_seqs(
        self,
        seqs: List[Sequence],
        pos: Optional[torch.Tensor],
        is_first_pp: bool,
        control_cmd_code: int = 0,
    ):
        if is_first_pp:
            schedule_sockets = self.schedule_first_pp_sockets
        else:
            schedule_sockets = self.schedule_other_sockets
        if not schedule_sockets:
            return
        data = (seqs, pos, control_cmd_code)
        for socket in schedule_sockets:
            self._get_sender(socket).put(data)

    def broadcast_control_cmd(
        self, control_cmd_code: int, profile_session_dir: Optional[str] = None
    ):
        data = ([], None, control_cmd_code, profile_session_dir)
        for socket in self.schedule_first_pp_sockets + self.schedule_other_sockets:
            self._get_sender(socket).put(data)

    def recv_schedule_seqs(self):
        if self.schedule_socket.poll(timeout=0) != 0:
            return self.schedule_socket.recv_pyobj()
        else:
            return None

    def send_ipc_package(self, ipc_package):
        self.request_socket.send_pyobj(ipc_package)

    def recv_ipc_package(self):
        if self.request_socket.poll(timeout=0) != 0:
            ipc_package = self.request_socket.recv_pyobj()
            return ipc_package
        else:
            return None
