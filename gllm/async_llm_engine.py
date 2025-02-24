import asyncio
import multiprocessing as mp
import time
from logger import logger
from typing import List, Dict
from multiprocessing import Queue
from fastapi import Request

from gllm.utils import make_async
from gllm.sequence import Sequence
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
        self.schedule_outputs: Queue = self.ctx.Queue()
        self.run_outputs: Queue = self.ctx.Queue()
        self.num_free_pages = self.ctx.Value('i', 0)

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

    async def run_schedule_engine(self):
        while True:
            if not self.run_outputs.empty():
                schedulerOutput, next_tokens = self.run_outputs.get()
                # print(
                #     f"GPU=>output {(time.time()-schedulerOutput.gpu_time)*1000}", flush=True)
                # print(f"OUTPUT START {time.time()%1000}", flush=True)
                self.scheduler.update_seqs(
                    schedulerOutput, next_tokens, delta=True)
                act_schedule_list = None
                if isinstance(schedulerOutput, SchedulerOutput):
                    act_schedule_list = schedulerOutput.schedule_lists
                elif isinstance(schedulerOutput, DeltaSchedulerOutput):
                    act_schedule_list = self.scheduler.decode_batch.schedule_lists
                for seq in act_schedule_list:
                    self.async_streams[seq.seq_id].put(
                        seq.detokenize_inc(self.model_runner.tokenizer))
                for seq in self.scheduler.finish_lists:
                    self.async_streams[seq.seq_id].finish()
                    del self.async_streams[seq.seq_id]
                self.free_finish_requests()
            # P=0 D=0 or P=1 D=0 or D=1 schedule P
            if self.scheduler.num_schedule_decode + self.scheduler.num_schedule_prefill == 0 or (
                self.scheduler.num_schedule_prefill == 1 and self.scheduler.num_schedule_decode == 0) or (
                    self.scheduler.num_schedule_decode == 1 and self.scheduler.can_schedule_prefill()):
                if not self.scheduler.has_seqs():
                    self.schedule_engine = None
                    return
                if not self.scheduler.has_scheduled_seqs():
                    await asyncio.sleep(0)
                    continue
                schedulerOutput = self.scheduler.schedule(
                    self.num_free_pages.value, log=True, delta=True)
                await self.check_abort_request(schedulerOutput)
                if isinstance(schedulerOutput, SchedulerOutput) and len(schedulerOutput.schedule_lists) == 0:
                    self.scheduler.num_schedule_prefill -= 1
                    await asyncio.sleep(0)
                    continue
                # schedulerOutput.schedule_time = time.time()
                self.schedule_outputs.put_nowait(schedulerOutput)
                # print(f"SCHEDULE {time.time()%1000}", flush=True)
            await asyncio.sleep(0)

    def run_gpu_engine(schedule_outputs: Queue, run_outputs: Queue, model_runner: ModelRunner, num_free_pages):
        decode_batch = SchedulerOutput([])
        while True:
            num_free_pages.value = model_runner.memory_manager.get_num_free_pages()

            schedulerOutput: SchedulerOutput = schedule_outputs.get()
            # print(
            #     f"Schedule=>GPU {(time.time()-schedulerOutput.schedule_time)*1000}", flush=True)
            # print(f"GPU START {time.time()%1000}", flush=True)

            if isinstance(schedulerOutput, DeltaSchedulerOutput):
                decode_batch.schedule_lists.extend(
                    schedulerOutput.delta_schedule_list)
                next_tokens = model_runner.step_once(decode_batch)
                act_schedule_list = decode_batch.schedule_lists
            elif isinstance(schedulerOutput, SchedulerOutput):
                next_tokens = model_runner.step_once(schedulerOutput)
                act_schedule_list = schedulerOutput.schedule_lists
            keep_indices = []
            free_indices = []
            for idx, seq in enumerate(act_schedule_list):
                if seq.is_finish():
                    free_indices.append(idx)
                else:
                    keep_indices.append(idx)
            schedulerOutput.keep_indices = keep_indices
            schedulerOutput.free_indices = free_indices
            if isinstance(schedulerOutput, DeltaSchedulerOutput):
                decode_batch.schedule_lists = [
                    decode_batch.schedule_lists[i] for i in keep_indices]
            # schedulerOutput.gpu_time = time.time()
            run_outputs.put_nowait((schedulerOutput, next_tokens))
            # print(f"GPU END {time.time()%1000}", flush=True)

    def start_gpu_engine(self):
        self.gpu_engine = self.ctx.Process(
            target=PipeAsyncLLM.run_gpu_engine,
            args=(self.schedule_outputs,
                  self.run_outputs,
                  self.model_runner,
                  self.num_free_pages))
        self.gpu_engine.start()

    def start_schedule_engine(self):
        # launch schedule engine
        self._schedule_task = asyncio.get_event_loop(
        ).create_task(self.run_schedule_engine())
        self._schedule_task.add_done_callback(
            _log_task_completion)
        self.schedule_engine = asyncio.shield(
            self._schedule_task)
