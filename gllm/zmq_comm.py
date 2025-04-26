import zmq
import pickle

from gllm.utils import make_socket
from gllm.dist_utils import send_obj_list, recv_obj_list

class zmqComm:
    def __init__(self, port_base, launch_mode, master_addr, pp_rank, pp_size, schedule_path, output_path, token_path):
        self.port_base = port_base
        self.master_addr = master_addr
        self.launch_mode = launch_mode
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.schedule_path = schedule_path
        self.output_path = output_path
        self.token_path = token_path

    def init(self):
        self.ctx = zmq.Context()
        
        if self.pp_size == 0:  # front-end process
            self.request_socket = make_socket(
                self.ctx, self.schedule_path, zmq.PUSH)
            self.output_socket = make_socket(
                self.ctx, self.output_path, zmq.PULL)
        else: # worker process
            if self.pp_rank == 0:
                # front-end => rank 0
                self.request_socket = make_socket(
                    self.ctx, self.schedule_path, zmq.PULL)
                # rank 0 => front-end
                self.output_socket = make_socket(
                    self.ctx, self.output_path, zmq.PUSH)
                
                if self.pp_size != 1:
                    if self.launch_mode == 'normal':
                        # rank 0 => other ranks : batched seqs
                        self.schedule_sockets = []
                        for i in range(1, self.pp_size):
                            self.schedule_sockets.append(make_socket(
                                self.ctx, f'{self.schedule_path}_{i}', zmq.PUSH))
                        # last rank => rank 0 : next tokens
                        self.token_socket = make_socket(
                            self.ctx, self.token_path, zmq.PULL)
                    else:
                        # rank 0 => other ranks : batched seqs
                        self.schedule_sockets = []
                        for i in range(1, self.pp_size):
                            port_each = self.port_base+i
                            self.schedule_sockets.append(make_socket(
                                self.ctx, f'tcp://{self.master_addr}:{port_each}', zmq.PUSH))
                            send_obj_list([port_each],i)
                        # last rank => rank 0 : next tokens
                        port_token = self.port_base+self.pp_size
                        self.token_socket = make_socket(
                            self.ctx, f'tcp://{self.master_addr}:{port_token}', zmq.PULL)
                        send_obj_list([port_token], self.pp_size-1)
            else:
                # rank 0 => other ranks : batched seqs
                if self.launch_mode == 'normal':
                    self.schedule_socket = make_socket(
                        self.ctx, f'{self.schedule_path}_{self.pp_rank}', zmq.PULL)
                else:
                    port_schedule = [None]
                    recv_obj_list(port_schedule, 0)
                    self.schedule_socket = make_socket(
                        self.ctx, f'tcp://{self.master_addr}:{port_schedule[0]}', zmq.PULL)

            if self.pp_rank == self.pp_size - 1 and self.pp_size != 1:
                # last rank => rank 0 : next tokens
                if self.launch_mode == 'normal':
                    self.token_socket = make_socket(
                        self.ctx, self.token_path, zmq.PUSH)
                else:
                    port_token = [None]
                    recv_obj_list(port_token, 0)
                    self.token_socket = make_socket(
                        self.ctx, f'tcp://{self.master_addr}:{port_token[0]}', zmq.PUSH)

        

    def send_tokens(self, tokens):
        assert type(tokens) == list
        tokens_bytes = pickle.dumps(tokens)
        self.token_socket.send(tokens_bytes, copy=False)

    def recv_tokens(self):
        if self.token_socket.poll(timeout=0) != 0:
            recv_bytes = self.token_socket.recv(copy=False)
            next_tokens = pickle.loads(recv_bytes)
            return next_tokens
        else:
            return None

    def send_output(self, output):
        output_bytes = pickle.dumps(output)
        self.output_socket.send(output_bytes, copy=False)

    def recv_output(self):
        if self.output_socket.poll(timeout=0) != 0:
            recv_bytes = self.output_socket.recv(copy=False)
            output = pickle.loads(recv_bytes)
            return output
        else:
            return None

    def send_schedule(self, seqs):
        seqs_bytes = pickle.dumps(seqs)
        for socket in self.schedule_sockets:
            socket.send(seqs_bytes, copy=False)

    def recv_schedule(self):
        if self.schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.schedule_socket.recv(copy=False)
            seqs = pickle.loads(recv_bytes)
            return seqs
        else:
            return None

    def send_requests(self, ipc_package):
        ipc_bytes = pickle.dumps(ipc_package)
        self.request_socket.send(ipc_bytes, copy=False)

    def recv_requests(self):
        if self.request_socket.poll(timeout=0) != 0:
            recv_bytes = self.request_socket.recv(copy=False)
            ipc_package = pickle.loads(recv_bytes)
            return ipc_package
        else:
            return None
