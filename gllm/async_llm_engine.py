import asyncio
import torch.multiprocessing as mp
import zmq
import pickle

from logger import logger
from typing import List, Dict
from fastapi import Request
from tqdm import tqdm

from gllm.utils import make_async, make_socket, wait_worker, check_worker_alive, random_uuid
from gllm.llm_engine import LLM
from gllm.async_worker import AsyncWorker, run_worker
from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.scheduler import SchedulerOutput


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
        kwargs.pop('assigned_layers')
        kwargs.pop('use_naive_schedule')
        assert kwargs['pp_size'] == 1 and "AsyncLLM doesn't support degree of PP > 1"
        logger.info('Using AsyncLLM backend')
        super().__init__(*args, **kwargs)
        super().init()

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
        schedulerOutput = self.scheduler.schedule(self.model_runner.memory_manager, log=True)
        next_tokens = await make_async(self.model_runner.step_once)(
            InputData(schedulerOutput.schedule_lists, self.model_runner.memory_manager))
        self.scheduler.update_seqs(schedulerOutput, next_tokens, self.model_runner.memory_manager)
        for seq in schedulerOutput.schedule_lists:
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
        logger.info('Using PipeAsyncLLM backend')
        
        self.assigned_layers = kwargs.pop('assigned_layers')
        self.use_naive_schedule = kwargs.pop('use_naive_schedule')
        
        super().__init__(*args, **kwargs)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None
        self.process_output_engine = None
        
        self.wait_lists: List[Sequence] = []
        self.running_maps: Dict[int,Sequence] = dict()

        self.ctx = mp.get_context('spawn')
        self.mp_alive = self.ctx.Array('i',[0 for i in range(self.pp_size)])
        self.mp_load_progress = self.ctx.Array('i',[0 for i in range(self.pp_size*2)])

        ipc_path_prefix = random_uuid()
        self.schedule_ipc_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_schedule'
        self.output_ipc_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_output'
        self.token_ipc_path = f'ipc:///tmp/{ipc_path_prefix}_gllm_token'

        logger.info(f"Launching {self.pp_size} worker(s) ...")
        for pp_rank in range(self.pp_size):
            self.start_worker(pp_rank)

        if kwargs['load_format'] == 'auto':
            self.load_progress()
        
        # wait worker start
        wait_worker(self.mp_alive, self.pp_size)
    
    def load_progress(self):
        total_weights = 0
        while True:
            ready = True
            total_weights = 0
            for i in range(self.pp_size):
                if self.mp_load_progress[i*2] == 0:
                    ready = False
                    continue
                total_weights += self.mp_load_progress[i*2]
            if ready:
                break
        pbar = tqdm(total=total_weights, bar_format='Loading model weights ... {l_bar}{bar}{r_bar}', ncols=100)
        last_total_weights = 0
        while True:
            cur_total_weights = 0
            for i in range(self.pp_size):
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

    async def run_schedule_engine(self):
        zmq_ctx = zmq.Context()
        schedule_socket = make_socket(
            zmq_ctx, self.schedule_ipc_path, zmq.PUSH)
        output_socket = make_socket(zmq_ctx, self.output_ipc_path, zmq.PULL)
        while True:
            check_worker_alive(self.mp_alive)
            if output_socket.poll(timeout=0) != 0:
                recv_bytes = output_socket.recv(copy=False)
                recv_data = pickle.loads(recv_bytes)
                schedulerOutput, next_tokens = recv_data
                for idx, id in enumerate(schedulerOutput.act_schedule_ids):
                    seq: Sequence = self.running_maps[id]
                    seq.token_ids.append(next_tokens[idx])
                    self.async_streams[id].put(seq.detokenize_inc(self.model_runner.tokenizer))
                    if id in schedulerOutput.free_ids:
                        self.running_maps.pop(id)
                        self.async_streams[id].finish()
                        del self.async_streams[id]
                self.free_finish_ids(schedulerOutput.free_ids)
            if len(self.wait_lists) != 0:
                for seq in self.wait_lists:
                    self.running_maps[seq.seq_id] = seq
                schedulerOutput = SchedulerOutput(self.wait_lists)
                self.wait_lists = []
                schedule_bytes = pickle.dumps(schedulerOutput)
                schedule_socket.send(schedule_bytes, copy=False)
            await asyncio.sleep(0)

    def start_worker(self, pp_rank):
        worker = AsyncWorker(self.model_runner,
                        pp_rank,
                        self.pp_size,
                        self.master_addr,
                        self.master_port,
                        self.schedule_ipc_path,
                        self.output_ipc_path,
                        self.token_ipc_path,
                        self.mp_alive,
                        self.mp_load_progress,
                        self.assigned_layers,
                        self.use_naive_schedule)
        self.ctx.Process(
            target=run_worker,
            args=(worker,),
            daemon=True).start()

    def start_schedule_engine(self):
        # launch schedule engine
        self._schedule_task = asyncio.get_event_loop(
        ).create_task(self.run_schedule_engine())
        self._schedule_task.add_done_callback(
            _log_task_completion)
        self.schedule_engine = asyncio.shield(
            self._schedule_task)
