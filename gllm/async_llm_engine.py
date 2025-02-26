import asyncio
import multiprocessing as mp
import time
import zmq
import pickle

from logger import logger
from typing import List, Dict
from fastapi import Request

from gllm.utils import make_async, make_socket
from gllm.llm_engine import LLM
from gllm.model_runner import ModelRunner
from gllm.scheduler import SchedulerOutput, DeltaSchedulerOutput


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


class AsyncEngineDeadError(RuntimeError):
    pass


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
        super().__init__(*args, **kwargs)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.background_engine = None

    async def add_requests_async(self, raw_request: Request, token_ids: List[int], output_len: int, ignore_eos: bool,
                                 temperature: float, top_p: float, top_k: float):
        seq = self.allocate_seq(token_ids, output_len,
                                ignore_eos, temperature, top_p, top_k)
        stream = AsyncStream(raw_request)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.background_engine is None:
            self.start_background_engine()
        return stream

    async def check_abort_request(self, schedulerOutput: SchedulerOutput):
        # we only check prefill requests
        if not schedulerOutput.computed_prompt:
            abort_seqs = []
            for seq in schedulerOutput.schedule_lists:
                if await self.async_streams[seq.seq_id]._raw_request.is_disconnected():
                    abort_seqs.append(seq)
            for seq in abort_seqs:
                schedulerOutput.schedule_lists.remove(seq)
                self.scheduler.finish_lists.append(seq)

    async def step_async(self):
        schedulerOutput = self.scheduler.schedule(
            self.model_runner.memory_manager.get_num_free_pages(), log=True, delta=False)
        await self.check_abort_request(schedulerOutput)
        if len(schedulerOutput.schedule_lists) == 0:
            self.scheduler.num_schedule_prefill -= 1
            return
        await make_async(self.model_runner.step_once)(schedulerOutput)
        self.scheduler.update_seqs(schedulerOutput)
        for seq in schedulerOutput.schedule_lists:
            self.async_streams[seq.seq_id].put(
                seq.detokenize_inc(self.model_runner.tokenizer))
        for seq in self.scheduler.finish_lists:
            self.async_streams[seq.seq_id].finish()
            del self.async_streams[seq.seq_id]
        self.free_finish_requests()

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
        logger.info('Enable pipeline schedule')
        super().__init__(*args, **kwargs)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None
        self.process_output_engine = None
        self.gpu_engine = None

        self.ctx = mp.get_context('spawn')
        self.num_free_pages = self.ctx.Value('i', 0)
        
        self.schedule_ipc_path = 'ipc:///tmp/gllm_schedule'
        self.output_ipc_path = 'ipc:///tmp/gllm_output'

        self.start_gpu_engine()

    async def add_requests_async(self, raw_request: Request, token_ids: List[int], output_len: int, ignore_eos: bool,
                                 temperature: float, top_p: float, top_k: float):
        seq = self.allocate_seq(token_ids, output_len,
                                ignore_eos, temperature, top_p, top_k)
        stream = AsyncStream(raw_request)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream

    async def check_abort_request(self, schedulerOutput):
        # we only check prefill requests
        if isinstance(schedulerOutput, SchedulerOutput):
            abort_seqs = []
            for seq in schedulerOutput.schedule_lists:
                if await self.async_streams[seq.seq_id]._raw_request.is_disconnected():
                    abort_seqs.append(seq)
            for seq in abort_seqs:
                schedulerOutput.schedule_lists.remove(seq)
                self.scheduler.finish_lists.append(seq)

    async def run_schedule(self, schedule_socket):
        
        if not self.scheduler.has_seqs():
            self.schedule_engine = None
            return True
        
        if not self.scheduler.has_scheduled_seqs():
            await asyncio.sleep(0)
            return False
        
        schedulerOutput = self.scheduler.schedule(
            self.num_free_pages.value, log=True, delta=True)

        # check abort requests
        await self.check_abort_request(schedulerOutput)
        if isinstance(schedulerOutput, SchedulerOutput) and len(schedulerOutput.schedule_lists) == 0:
            self.scheduler.num_schedule_prefill -= 1
            await asyncio.sleep(0)
            return False
        
        if isinstance(schedulerOutput, DeltaSchedulerOutput) and len(schedulerOutput.delta_schedule_list) == 0:
            await asyncio.sleep(0)
            return False
        
        schedule_bytes = pickle.dumps(schedulerOutput)
        schedule_socket.send(schedule_bytes, copy=False)

        return False

    async def run_schedule_engine(self):
        zmq_ctx = zmq.Context()
        schedule_socket = make_socket(zmq_ctx, self.schedule_ipc_path, zmq.PUSH)
        output_socket = make_socket(zmq_ctx, self.output_ipc_path, zmq.PULL)
        while True:
            if output_socket.poll(timeout=0) != 0:

                recv_bytes = output_socket.recv(copy=False)
                schedulerOutput, next_tokens = pickle.loads(recv_bytes)

                self.scheduler.update_seqs(
                    schedulerOutput, next_tokens, delta=True)
                # overlap gpu execution and output process
                exit = await self.run_schedule(schedule_socket)
                act_schedule_list = None
                if isinstance(schedulerOutput, SchedulerOutput):
                    act_schedule_list = schedulerOutput.schedule_lists
                elif isinstance(schedulerOutput, DeltaSchedulerOutput):
                    act_schedule_list = self.scheduler.decode_batch.schedule_lists
                else:
                    assert 0
                for seq in act_schedule_list:
                    self.async_streams[seq.seq_id].put(
                        seq.detokenize_inc(self.model_runner.tokenizer))
                for seq in self.scheduler.finish_lists:
                    self.async_streams[seq.seq_id].finish()
                    del self.async_streams[seq.seq_id]
                self.free_finish_requests()

                if exit:
                    return
            if await self.run_schedule(schedule_socket):
                return
            await asyncio.sleep(0)

    def run_gpu_engine(model_runner: ModelRunner, num_free_pages, schedule_ipc_path, output_ipc_path):
        zmq_ctx = zmq.Context()
        schedule_socket = make_socket(zmq_ctx, schedule_ipc_path, zmq.PULL)
        output_socket = make_socket(zmq_ctx,output_ipc_path, zmq.PUSH)
        decode_batch = SchedulerOutput([])
        while True:
            num_free_pages.value = model_runner.memory_manager.get_num_free_pages()

            schedulerOutput: SchedulerOutput = None
            act_schedule_list = None
            next_tokens = None
            if schedule_socket.poll(timeout=0) != 0:
                recv_bytes = schedule_socket.recv(copy=False)
                schedulerOutput = pickle.loads(recv_bytes)
                
                if isinstance(schedulerOutput, DeltaSchedulerOutput):
                    decode_batch.schedule_lists.extend(
                        schedulerOutput.delta_schedule_list)
                    next_tokens = model_runner.step_once(decode_batch)
                    act_schedule_list = decode_batch.schedule_lists
                elif isinstance(schedulerOutput, SchedulerOutput):
                    next_tokens = model_runner.step_once(schedulerOutput)
                    act_schedule_list = schedulerOutput.schedule_lists
                else:
                    assert 0
            elif len(decode_batch.schedule_lists) != 0:
                schedulerOutput = DeltaSchedulerOutput([],[])
                next_tokens = model_runner.step_once(decode_batch)
                act_schedule_list = decode_batch.schedule_lists
            else:
                continue
            
            keep_indices = []
            free_indices = []
            for idx, seq in enumerate(act_schedule_list):
                if seq.is_finish():
                    free_indices.append(idx)
                else:
                    keep_indices.append(idx)
            schedulerOutput.free_indices = free_indices
            if isinstance(schedulerOutput, DeltaSchedulerOutput):
                decode_batch.schedule_lists = [
                    decode_batch.schedule_lists[i] for i in keep_indices]

            output_bytes = pickle.dumps((schedulerOutput, next_tokens))
            output_socket.send(output_bytes, copy=False)


    def start_gpu_engine(self):
        self.gpu_engine = self.ctx.Process(
            target=PipeAsyncLLM.run_gpu_engine,
            args=(self.model_runner,
                  self.num_free_pages,
                  self.schedule_ipc_path,
                  self.output_ipc_path))
        self.gpu_engine.start()

    def start_schedule_engine(self):
        # launch schedule engine
        self._schedule_task = asyncio.get_event_loop(
        ).create_task(self.run_schedule_engine())
        self._schedule_task.add_done_callback(
            _log_task_completion)
        self.schedule_engine = asyncio.shield(
            self._schedule_task)
