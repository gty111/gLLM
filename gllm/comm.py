import threading
from typing import List, Optional

import torch
import zmq

from gllm.dist_utils import (
    get_rank,
    get_tp_size,
    get_tp_rank,
    get_pp_size,
    is_first_pp_rank,
    is_last_pp_rank,
)
from gllm.sequence import Sequence
from gllm.utils import make_socket


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
        tp_size=1,
    ):
        self.host_addr = host_addr
        self.port_base = port_base
        self.master_addr = master_addr
        self.launch_mode = launch_mode
        self.schedule_path = schedule_path
        self.output_path = output_path
        self.token_path = token_path
        # Frontend
        self.frontend = frontend
        self.tp_size = tp_size # only used for frontend
        
    def get_schedule_path(self, rank):
        return f"{self.schedule_path}_rank_{rank}"
    
    def get_token_path(self, rank):
        return f"{self.token_path}_rank_{rank}"
    
    def init_request_socket(self):
        # front-end <=> PP rank 0
        if self.frontend:
            self.request_sockets = []
            for tp_rank in range(self.tp_size):
                self.request_sockets.append(
                    make_socket(self.ctx, 
                                self.get_schedule_path(tp_rank), 
                                zmq.PUSH))
        elif is_first_pp_rank():
            self.request_socket = make_socket(
                self.ctx,
                self.get_schedule_path(get_tp_rank()),
                zmq.PULL,
            )
    
    def init_output_socket(self):
        # frontend <=> rank 0
        if self.frontend:
            self.output_socket = make_socket(self.ctx, self.output_path, zmq.PULL)
        elif get_rank() == 0:
            self.output_socket = make_socket(self.ctx, self.output_path, zmq.PUSH)
            
    def init_pp_schedule_socket(self):
        # prior PP rank <=> next PP rank
        if self.frontend:
            return
        if not is_first_pp_rank():
            self.schedule_socket = make_socket(
                self.ctx,
                self.get_schedule_path(get_rank()),
                zmq.PULL,
            )
        if not is_last_pp_rank():
            self.schedule_sockets = []
            for idx in range(1, get_pp_size()):
                self.schedule_sockets.append(
                    make_socket(self.ctx, 
                                self.get_schedule_path(idx*get_tp_size() + get_rank()), 
                                zmq.PUSH))
    
    def init_token_socket(self):
        # last PP rank <=> first PP rank
        if self.frontend:
            return
        if is_last_pp_rank():
            self.token_socket = make_socket(
                self.ctx,
                self.get_token_path(get_tp_rank()),
                zmq.PUSH,
            )
        if is_first_pp_rank():
            self.token_socket = make_socket(
                self.ctx,
                self.get_token_path(get_tp_rank()),
                zmq.PULL,
            )
        

    def init(self):
        self.ctx = zmq.Context()
        
        assert self.launch_mode == "normal"
        
        self.init_request_socket()
        
        self.init_output_socket()
        
        self.init_pp_schedule_socket()
        
        self.init_token_socket()

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
        if get_rank() == 0:
            self.output_socket.send_pyobj(output)

    def recv_output(self):
        if self.output_socket.poll(timeout=0) != 0:
            output = self.output_socket.recv_pyobj()
            return output
        else:
            return None

    def send_schedule_seqs(
        self,
        seqs: List[Sequence],
        pos: Optional[torch.Tensor],
        control_cmd_code: int = 0,
    ):
        data = (seqs, pos, control_cmd_code)
        for schedule_socket in self.schedule_sockets: 
            threading.Thread(target=schedule_socket.send_pyobj, args=(data,)).start()

    def send_control_cmd(
        self, control_cmd_code: int, profile_session_dir: Optional[str] = None
    ):
        data = ([], None, control_cmd_code, profile_session_dir)
        for schedule_socket in self.schedule_sockets:
            threading.Thread(target=schedule_socket.send_pyobj, args=(data,)).start()

    def recv_schedule_seqs(self):
        if self.schedule_socket.poll(timeout=0) != 0:
            return self.schedule_socket.recv_pyobj()
        else:
            return None

    def send_ipc_package(self, ipc_package):
        for request_socket in self.request_sockets:
            threading.Thread(target=request_socket.send_pyobj, args=(ipc_package,)).start()

    def recv_ipc_package(self):
        if self.request_socket.poll(timeout=0) != 0:
            ipc_package = self.request_socket.recv_pyobj()
            return ipc_package
        else:
            return None
