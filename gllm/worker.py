import torch
import torch.distributed as dist
import traceback
import logging

from attr import dataclass
from collections import deque
from logger import logger
from transformers.image_utils import load_images
from gllm.layers.rotary_embedding import MRotaryEmbedding
from typing import Dict

from gllm.sequence import Sequence
from gllm.input_data import InputData
from gllm.model_runner import ModelRunner
from gllm.comm import zmqComm, IPCPackage
from gllm.worker_scheduler import WorkerScheduler
from gllm.dist_utils import (get_local_rank, init_dist, send_pp_data, recv_pp_data, 
                             get_rank, get_world_size, is_last_pp_rank,
                             get_pp_size, get_next_pp_rank, get_last_pp_rank,
                             is_output_rank)

@dataclass
class EmbeddingInfo:
    embedding: torch.Tensor = None
    positions: torch.Tensor = None
    mrope_position_delta:int = None
    stale: bool = False

# Used with PipeAsyncLLM
class Worker:

    def __init__(self, model_runner: ModelRunner, local_rank, pp_rank, tp_rank, 
                 pp_size, tp_size, use_ep, master_addr, master_port, comm: zmqComm, mp_alive,
                 mp_load_progress, assigned_layers, use_cp_schedule):
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
        self.use_cp_schedule = use_cp_schedule
        self.use_mla = model_runner.model_loader.use_mla

    def init_logger(self):
        tp_ep_log = 'TP' if not self.use_ep or self.tp_size == 1 else 'TP/EP'
        formater = logging.Formatter(
            f'[%(asctime)s %(filename)s:%(lineno)d Worker{self.pp_rank*self.tp_size+self.tp_rank} '
            f'PP{self.pp_rank} {tp_ep_log}{self.tp_rank}] %(levelname)s - %(message)s',
            datefmt="%H:%M:%S")
        for handler in logger.handlers:
            handler.setFormatter(formater)

    def init(self):
        self.init_logger()
        if self.pp_size > 1 or self.tp_size > 1:
            init_dist(self.pp_size, self.tp_size, self.use_ep, self.local_rank, self.pp_rank, self.tp_rank, self.master_addr, 
                    self.master_port, self.assigned_layers)
        self.rank = get_rank()
        torch.cuda.set_device(f'cuda:{self.local_rank}')
        
        
        self.comm.init()
        
        self.model_runner.init(self.mp_load_progress)
        self.hidden_size = self.model_runner.model_loader.hidden_size
        self.ret_residual = self.model_runner.model.ret_residual
        self.use_mm = self.model_runner.use_mm
        self.model = self.model_runner.model
        if self.model_runner.use_mm:
            self.image_processor = self.model_runner.image_processor
        
        if self.rank == 0:
            self.worker_scheduler = WorkerScheduler(
                self.pp_size,
                self.model_runner, 
                self.use_cp_schedule,
            )
            # embedding cache: seq_id => embedding
            self.embedding_cache:Dict[int,EmbeddingInfo] = {}
        elif self.pp_rank == 0:
            # Input data for each rank except 0
            self.schedule_queue = deque()
        else:
            # Input data for each rank except 0
            self.schedule_queue = deque()
            # Input data and intermediate data for rank except 0
            self.run_queue = deque()

        self.mp_alive[self.local_rank] = 1

        logger.info(f'Initialization complete')

    # driver worker => other workers
    def recv_schedule_seqs(self):
        seqs_data = self.comm.recv_schedule_seqs()
        if seqs_data is not None:
            seqs, positions = seqs_data
            positions = positions.to(f'cuda:{get_local_rank()}')
            self.schedule_queue.append(
                InputData(seqs, self.model_runner.memory_manager,
                          use_mla=self.use_mla, positions=positions))

    # pp last rank => pp next rank
    def recv_intermediate_data(self):
        if len(self.schedule_queue) != 0:
            input_data = self.schedule_queue.popleft()
            intermediate_data = recv_pp_data(
                get_last_pp_rank(),
                [input_data.tokens.shape[0], self.hidden_size], self.ret_residual)
            self.run_queue.append((input_data, intermediate_data))

    def forward_pp(self):
        if len(self.run_queue) != 0:
            hidden_states = None
            residual = None
            if self.ret_residual:
                input_data, (hidden_states, residual) = self.run_queue.popleft()
            else:
                input_data, hidden_states = self.run_queue.popleft()
            
            output = self.model_runner.step_once(
                input_data, hidden_states, residual)
            if is_output_rank():
                self.comm.send_tokens(output)
            elif not is_last_pp_rank():
                send_pp_data(output, get_next_pp_rank())

    def recv_ipc_package(self):
        # To avoid request accumulation, we fetch all packages in comm
        cum_ipc_package = IPCPackage([])
        while True:
            ipc_package:IPCPackage = self.comm.recv_ipc_package()
            if ipc_package is not None:
                cum_ipc_package.schedule_lists.extend(ipc_package.schedule_lists)
                cum_ipc_package.abort_ids.extend(ipc_package.abort_ids)
                cum_ipc_package.log &= ipc_package.log
            else:
                break
        if len(cum_ipc_package.schedule_lists) != 0 or len(cum_ipc_package.abort_ids) != 0:
            self.worker_scheduler.add_new_requests(cum_ipc_package.schedule_lists)
            self.worker_scheduler.add_abort_ids(cum_ipc_package.abort_ids)
            self.worker_scheduler.set_log(cum_ipc_package.log)

    def recv_next_tokens(self):
        if self.pp_size != 1:  # recv tokens from last rank
            next_tokens = self.comm.recv_tokens()
            if next_tokens is not None:
                self.worker_scheduler.add_next_tokens(next_tokens)
                
    def check_abort_seqs(self):
        ipc_package = self.worker_scheduler.check_abort_seqs()
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def process_output(self):
        ipc_package = self.worker_scheduler.process_output()
        if ipc_package is not None:
            self.comm.send_output(ipc_package)

    def forward_tp(self):
        if len(self.schedule_queue) != 0:
            input_data = self.schedule_queue.popleft()
            output = self.model_runner.step_once(input_data)
            if get_pp_size() != 1:
                send_pp_data(output, get_next_pp_rank())
                
    
    @torch.inference_mode()
    def mm_prepare_inputs(self, seqs: list[Sequence]):
        # Calculate the embedding and positions of pic
        batch_embeddings = []
        batch_positions = []
        for seq in seqs:
            embedding = None
            position = None
            if seq.computed_prompt:
                embedding_info = self.embedding_cache[seq.seq_id]
                assert embedding_info.stale
                position = MRotaryEmbedding.get_next_input_positions(
                    embedding_info.mrope_position_delta,
                    seq.computed_token_num,
                    seq.seq_len
                )
                position = torch.tensor(position)
                embedding = self.model.get_input_embeddings(
                    torch.tensor(seq.token_ids[seq.computed_token_num:seq.seq_len]))
            else:
                embedding_info = None
                if seq.seq_id not in self.embedding_cache or self.embedding_cache[seq.seq_id].stale:
                    mm_embeddings = None
                    image_grid_thw = None
                    if seq.mm_contents is not None:
                        images = load_images(seq.mm_contents)
                        images_input = self.image_processor(images=images)
                        image_grid_thw = images_input['image_grid_thw']
                        mm_embeddings = self.model.get_multimodal_embeddings(**images_input)
                    prompt_embeddings = self.model.get_input_embeddings(
                        torch.tensor(seq.token_ids), mm_embeddings)
                    prompt_positions, mrope_position_delta = MRotaryEmbedding.get_input_positions(
                        input_tokens=seq.token_ids,
                        hf_config=self.model.config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=None,
                        second_per_grid_ts=None,
                    )
                    embedding_info = EmbeddingInfo(
                        prompt_embeddings, prompt_positions, mrope_position_delta)
                    self.embedding_cache[seq.seq_id] = embedding_info
                    position = prompt_positions[
                        :, seq.computed_token_num:seq.seq_len]
                    embedding = prompt_embeddings[
                        seq.computed_token_num:seq.seq_len, :]
                else:
                    embedding_info = self.embedding_cache[seq.seq_id]
                    position = embedding_info.positions[
                        :, seq.computed_token_num:seq.seq_len]
                    embedding = embedding_info.embedding[
                        seq.computed_token_num:seq.seq_len, :]
                if seq.seq_len == seq.prompt_len:
                    # invalidate embedding_cache
                    embedding_info.stale = True
                    embedding_info.embedding = None
                    embedding_info.positions = None
            batch_embeddings.append(embedding)
            batch_positions.append(position)
        input_embeddings = torch.concat(batch_embeddings)
        positions = torch.concat(batch_positions, dim=1)
        return input_embeddings, positions
    
    def schedule_forward(self):
        schedule_seqs = self.worker_scheduler.schedule_once()
        if len(schedule_seqs) != 0:
            if self.use_mm:
                input_embeddings, positions = self.mm_prepare_inputs(schedule_seqs)
            input_data = InputData(
                schedule_seqs, self.model_runner.memory_manager,
                use_mla=self.use_mla, positions=positions)
            if get_world_size() > 1:
                self.comm.send_schedule_seqs((schedule_seqs, positions))
            output = self.model_runner.step_once(input_data, input_embeddings=input_embeddings)

            if not is_output_rank():
                send_pp_data(output, get_next_pp_rank())
            else:
                self.worker_scheduler.add_next_tokens(output)

    def run_driver(self):
        self.check_abort_seqs()
        self.recv_ipc_package()
        self.recv_next_tokens()
        self.schedule_forward()
        self.process_output()
        
    def run_first_tp(self):
        self.recv_schedule_seqs()
        self.forward_tp()

    def run_other(self):
        self.recv_schedule_seqs()
        self.recv_intermediate_data()
        self.forward_pp()

    def handle_keyboardInterrupt(self):
        self.mp_alive[self.local_rank] = -1
        logger.info(f'Exit')
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
            if worker.rank == 0:
                worker.run_driver()
            elif worker.pp_rank == 0:
                worker.run_first_tp()
            else:
                worker.run_other()
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
