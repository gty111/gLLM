import asyncio
import multiprocessing as mp
from logger import logger
from typing import List, Dict
from multiprocessing import Queue
from fastapi import Request

from gllm.utils import make_async
from gllm.sequence import Sequence
from gllm.llm_engine import LLM
from gllm.model_runner import ModelRunner

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
        seq = self.allocate_seq(token_ids, output_len, ignore_eos, temperature, top_p, top_k)
        stream = AsyncStream(raw_request)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.background_engine is None:
            self.start_background_engine()
        return stream
    
    async def check_abort_request(self, scheduled_seqs: List[Sequence]):
        if not scheduled_seqs[0].computed_prompt:
            abort_seqs = []
            for seq in scheduled_seqs:
                if await self.async_streams[seq.seq_id]._raw_request.is_disconnected():
                    abort_seqs.append(seq)
            for seq in abort_seqs:
                scheduled_seqs.remove(seq)
                self.scheduler.finish_lists.append(seq)  

    async def step_async(self):
        scheduled_seqs = self.scheduler.schedule(self.model_runner.memory_manager.get_num_free_pages(), True)
        await self.check_abort_request(scheduled_seqs)
        if len(scheduled_seqs) == 0:
            self.scheduler.num_schedule_running -= 1
            return
        await make_async(self.model_runner.step_once)(seqs=scheduled_seqs)
        self.scheduler.update_seqs(scheduled_seqs)
        for seq in scheduled_seqs:
            self.async_streams[seq.seq_id].put(seq.detokenize_inc(self.model_runner.tokenizer))
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
        super().__init__(*args, **kwargs)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None
        self.process_output_engine = None
        self.gpu_engine = None

        self.ctx = mp.get_context('spawn')
        self.schedule_outputs: Queue = self.ctx.Queue()
        self.run_outputs: Queue = self.ctx.Queue()
        self.control_schedule : asyncio.Queue = asyncio.Queue()
        self.num_free_pages = self.ctx.Value('i', 0)

        self.start_gpu_engine()
        
        logger.info('Enable pipeline schedule')
        

    async def add_requests_async(self, raw_request:Request, token_ids: List[int], output_len: int, ignore_eos:bool,
                                 temperature: float, top_p: float, top_k: float):
        seq = self.allocate_seq(token_ids, output_len, ignore_eos, temperature, top_p, top_k)
        stream = AsyncStream(raw_request)
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream      
    
    async def check_abort_request(self, scheduled_seqs: List[Sequence]):
        if not scheduled_seqs[0].computed_prompt:
            abort_seqs = []
            for seq in scheduled_seqs:
                if await self.async_streams[seq.seq_id]._raw_request.is_disconnected():
                    abort_seqs.append(seq)
            for seq in abort_seqs:
                scheduled_seqs.remove(seq)
                self.scheduler.finish_lists.append(seq)    
        
    async def run_schedule_engine(self):
        while True:
            await self.control_schedule.get()
            # print("SCHEDULE: start",time.time())
            if not self.scheduler.has_seqs():
                self.schedule_engine = None
                self.control_schedule.put_nowait(0)
                return
            if not self.scheduler.has_scheduled_seqs() or self.scheduler.delay_schedule():
                self.control_schedule.put_nowait(0)
                await asyncio.sleep(0)
                continue
            scheduled_seqs = self.scheduler.schedule(self.num_free_pages.value, True)
            await self.check_abort_request(scheduled_seqs)
            if len(scheduled_seqs) == 0:
                self.scheduler.num_schedule_running -= 1
                self.control_schedule.put_nowait(0)
                await asyncio.sleep(0)
                continue
            # print("SCHEDULE: end",time.time())
            self.schedule_outputs.put_nowait(scheduled_seqs)
        
    def run_gpu_engine(schedule_outputs:Queue, run_outputs:Queue, model_runner:ModelRunner, num_free_pages):
        while True:
            num_free_pages.value = model_runner.memory_manager.get_num_free_pages()
            # print("GPU: wait",time.time())
            scheduled_seqs = schedule_outputs.get()
            # print("GPU: start",time.time())
            model_runner.step_once(seqs=scheduled_seqs)
            # print("GPU: end",time.time())
            run_outputs.put_nowait(scheduled_seqs)
            
    async def run_process_output_engine(self):
        self.control_schedule.put_nowait(0)
        while True:
            self.control_schedule.put_nowait(0)
            # print("OUTPUT: wait",time.time())
            while self.run_outputs.empty():
                await asyncio.sleep(0)
            scheduled_seqs:List[Sequence] = self.run_outputs.get()
            # print("OUTPUT: start",time.time())
            self.scheduler.update_seqs(scheduled_seqs)
            for seq in scheduled_seqs:
                self.async_streams[seq.seq_id].put(seq.detokenize_inc(self.model_runner.tokenizer))
            for seq in self.scheduler.finish_lists:
                self.async_streams[seq.seq_id].finish()
                del self.async_streams[seq.seq_id]
            self.free_finish_requests()
            # print("OUTPUT: end",time.time())


    def start_gpu_engine(self):
        self.gpu_engine = self.ctx.Process(
            target=PipeAsyncLLM.run_gpu_engine,args=(self.schedule_outputs, 
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
        
        if self.process_output_engine is None:
            self.start_process_output_engine()

    def start_process_output_engine(self):
        # launch process output engine
        self._process_output_task = asyncio.get_event_loop(
        ).create_task(self.run_process_output_engine())
        self._process_output_task.add_done_callback(
            _log_task_completion)
        self.process_output_engine = asyncio.shield(
            self._process_output_task)
