import asyncio
import torch.multiprocessing as mp
import sys

from logger import logger
from typing import List, Dict
from fastapi import Request

from gllm.utils import (make_async, wait_worker,
                        check_worker_alive, random_uuid, get_model_load_pbar)
from gllm.llm_engine import LLM
from gllm.async_worker import AsyncWorker, run_worker_async
from gllm.worker import Worker, run_worker
from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.frontend_scheduler import IPCPackage
from gllm.zmq_comm import zmqComm


class AsyncStream:

    def __init__(self, raw_request: Request):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
        self._raw_request = raw_request

    def put(self, item: str):
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self):
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    
    async def is_disconnected(self):
        return await self._raw_request.is_disconnected()


def _log_task_completion(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        logger.error("Engine background task failed", exc_info=e)


class AsyncLLM(LLM):

    def __init__(self, *args, **kwargs):
        if kwargs['pp_size'] != 1 or kwargs['tp_size'] != 1:
            raise Exception('TP and PP are not support by AsyncLLM, please use PipeAsyncLLM!')
        super().__init__(*args, **kwargs)
        super().init()

        logger.info('Using AsyncLLM backend')

        self.async_streams: Dict[int, AsyncStream] = {}
        self.background_engine = None

    async def add_requests_async(self, raw_request: Request, token_ids: List[int], output_len: int, ignore_eos: bool,
                                 temperature: float, top_p: float, top_k: float, repetition_penalty: float):
        seq = self.allocate_seq(token_ids, output_len, ignore_eos,
                                temperature, top_p, top_k, repetition_penalty)
        stream = AsyncStream(raw_request)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.background_engine is None:
            self.start_background_engine()
        return stream

    async def step_async(self):
        ipc_package = self.scheduler.schedule(
            self.model_runner.memory_manager, log=True)
        next_tokens = await make_async(self.model_runner.step_once)(
            InputData(ipc_package.schedule_lists, self.model_runner.memory_manager))
        self.scheduler.update_seqs(
            ipc_package, next_tokens, self.model_runner.memory_manager)
        for seq in ipc_package.schedule_lists:
            self.async_streams[seq.seq_id].put(
                seq.detokenize_inc(self.model_runner.tokenizer))
        for seq_id in self.scheduler.finish_ids:
            self.async_streams[seq_id].finish()
            del self.async_streams[seq_id]
        self.free_finish_ids(self.scheduler.get_finish_ids())

    async def run_engine(self):
        while self.scheduler.has_seqs():
            await self.step_async()
        self.background_engine = None

    def start_background_engine(self):
        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine())
        self._background_loop_unshielded.add_done_callback(
            _log_task_completion)
        self.background_engine = asyncio.shield(
            self._background_loop_unshielded)


class PipeAsyncLLM(LLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.info('Using PipeAsyncLLM backend')

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None

        self.wait_lists: List[Sequence] = []
        self.abort_ids: List[int] = []
        self.running_maps: Dict[int, Sequence] = dict() # seq_id => Sequence
        
        if self.launch_mode != 'normal':
            if self.worker_ranks is None:
                logger.error('Please specify arg --ranks when the launching mode is master/slave')
                sys.exit(1)
            self.act_worker_ranks = [int(i) for i in self.worker_ranks.split(',')]
            assert len(self.act_worker_ranks) != 0
        else:
            self.act_worker_ranks = list(range(self.pp_size*self.tp_size))
        self.num_workers = len(self.act_worker_ranks)

        self.ctx = mp.get_context('spawn')
        self.mp_alive = self.ctx.Array('i', [0 for i in range(self.num_workers)])
        self.mp_load_progress = self.ctx.Array(
            'i', [0 for i in range(self.num_workers*2)])

        ipc_path_prefix = random_uuid()
        self.schedule_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_schedule'
        self.output_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_output'
        self.token_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_token'

        self.comm = zmqComm(self.host, self.zmq_port_base, self.launch_mode, self.master_addr, 
                            self.schedule_path, self.output_path, self.token_path, frontend=True)
        self.comm.init()

        logger.info(f'Launching worker {self.act_worker_ranks} ...')
        if self.use_async_worker:
            logger.warning(f'AsyncWorker is an experimental feature')
        if self.launch_mode != 'normal':
            logger.warning(f'Multi-node support is an experimental feature')
            
        self.process_list = []
        for local_rank, rank in enumerate(self.act_worker_ranks):
            pp_rank = rank // self.tp_size
            tp_rank = rank % self.tp_size
            self.start_worker(local_rank, pp_rank, tp_rank)

        if kwargs['load_format'] == 'auto':
            self.load_progress()

        # wait worker start
        wait_worker(self.mp_alive, self.num_workers)

    def load_progress(self):
        total_weights = 0
        while True:
            check_worker_alive(self.mp_alive)
            ready = True
            total_weights = 0
            for i in range(self.num_workers):
                if self.mp_load_progress[i*2] == 0:
                    ready = False
                    continue
                total_weights += self.mp_load_progress[i*2]
            if ready:
                break
        pbar = get_model_load_pbar(total_weights)
        last_total_weights = 0
        while True:
            check_worker_alive(self.mp_alive)
            cur_total_weights = 0
            for i in range(self.num_workers):
                cur_total_weights += self.mp_load_progress[i*2+1]
            pbar.update(cur_total_weights-last_total_weights)
            last_total_weights = cur_total_weights
            if cur_total_weights == total_weights:
                break

    def add_requests(self, requests: List[Sequence]):
        self.wait_lists.extend(requests)

    async def add_requests_async(self, raw_request: Request, token_ids: List[int], output_len: int, ignore_eos: bool,
                                 temperature: float, top_p: float, top_k: float, repetition_penalty: float):
        seq = self.allocate_seq(token_ids, output_len, ignore_eos,
                                temperature, top_p, top_k, repetition_penalty)
        stream = AsyncStream(raw_request)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream
    
    async def check_abort_seqs(self):
        for id, seq in self.running_maps.items():
            if await self.async_streams[id].is_disconnected() and not seq.is_abort:
                self.abort_ids.append(id)
                seq.is_abort = True
                
    def recv_ipc_package(self):
        ipc_package:IPCPackage = self.comm.recv_output()
        if ipc_package is not None:
            for idx, id in enumerate(ipc_package.act_schedule_ids):
                if len(ipc_package.next_tokens) != 0:
                    seq: Sequence = self.running_maps[id]
                    seq.append(ipc_package.next_tokens[idx])
                    self.async_streams[id].put(
                        seq.detokenize_inc(self.model_runner.tokenizer))
                if id in ipc_package.free_ids:
                    self.running_maps.pop(id)
                    self.async_streams[id].finish()
                    del self.async_streams[id]
            self.free_finish_ids(ipc_package.free_ids)

    def send_ipc_package(self):
        if len(self.wait_lists) != 0 or len(self.abort_ids) != 0:
            for seq in self.wait_lists:
                self.running_maps[seq.seq_id] = seq
            ipc_package = IPCPackage(self.wait_lists)
            if len(self.abort_ids) != 0:
                logger.warning(f'Abort {len(self.abort_ids)} request(s) due to loss of network connection')
            ipc_package.abort_ids = self.abort_ids
            self.wait_lists = []
            self.abort_ids = []
            self.comm.send_ipc_package(ipc_package)

    async def run_schedule_engine(self):
        while True:
            check_worker_alive(self.mp_alive)
            self.recv_ipc_package()
            await self.check_abort_seqs()
            self.send_ipc_package()
            await asyncio.sleep(0)

    def start_worker(self, local_rank, pp_rank, tp_rank):
        worker_cls = Worker if not self.use_async_worker else AsyncWorker
        comm = zmqComm(self.host,
                       self.zmq_port_base,
                       self.launch_mode,
                       self.master_addr,
                       self.schedule_path,
                       self.output_path,
                       self.token_path)
        worker = worker_cls(self.model_runner,
                            local_rank,
                            pp_rank,
                            tp_rank,
                            self.pp_size,
                            self.tp_size,
                            self.use_ep,
                            self.master_addr,
                            self.master_port,
                            comm,
                            self.mp_alive,
                            self.mp_load_progress,
                            self.assigned_layers,
                            self.use_naive_schedule)
        process = self.ctx.Process(
                    target=run_worker if not self.use_async_worker else run_worker_async,
                    args=(worker,),
                    daemon=True)
        self.process_list.append(process)
        process.start()

    def start_schedule_engine(self):
        # launch schedule engine
        self._schedule_task = asyncio.get_event_loop(
        ).create_task(self.run_schedule_engine())
        self._schedule_task.add_done_callback(
            _log_task_completion)
        self.schedule_engine = asyncio.shield(
            self._schedule_task)
