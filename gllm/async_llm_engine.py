import asyncio
import multiprocessing as mp
import time
from logger import logger
from typing import List, Dict
from multiprocessing import Queue,Process

from gllm.utils import make_async
from gllm.sequence import Sequence
from gllm.llm_engine import LLM
from gllm.model_runner import ModelRunner

class AsyncStream:

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

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

    def __init__(self, model_path, gpu_memory_utilization=0.9, page_size=16, max_decode_seqs=256, max_batch_tokens=8192, ratio_threshold_free_pages=0.2):
        super().__init__(model_path, gpu_memory_utilization, page_size,
                         max_decode_seqs, max_batch_tokens, ratio_threshold_free_pages)

        self.async_streams: Dict[int, AsyncStream] = {}
        self.schedule_engine = None
        self.process_output_engine = None
        self.gpu_engine = None

        self.ctx = mp.get_context('spawn')
        self.schedule_outputs: Queue = self.ctx.Queue()
        self.run_outputs: Queue = self.ctx.Queue()
        self.control_schedule : asyncio.Queue = asyncio.Queue()

        self.start_gpu_engine()
        

    async def add_requests_async(self, token_ids: List[int], output_len: int):
        seq = self.allocate_seq(token_ids, output_len)
        stream = AsyncStream()
        assert seq.seq_id not in self.async_streams
        self.async_streams[seq.seq_id] = stream
        await make_async(self.add_requests)(requests=[seq])
        if self.schedule_engine is None:
            self.start_schedule_engine()
        return stream       
        
    async def run_schedule_engine(self):
        while True:
            num_free_pages = await self.control_schedule.get()
            # print("SCHEDULE: start",time.time())
            if not self.scheduler.has_seqs():
                self.schedule_engine = None
                self.control_schedule.put_nowait(num_free_pages)
                return
            if not self.scheduler.has_seqs_cur() or self.scheduler.delay_schedule():
                self.control_schedule.put_nowait(num_free_pages)
                await asyncio.sleep(0)
                continue
            scheduled_seqs = self.scheduler.schedule(num_free_pages)
            # time.sleep(0.015)
            # print("SCHEDULE: end",time.time())
            self.schedule_outputs.put_nowait(scheduled_seqs)
        
    def run_gpu_engine(schedule_outputs:Queue,run_outputs:Queue,model_runner:ModelRunner):
        logger.info('start gpu engine process')
        while True:
            # print("GPU: wait",time.time())
            scheduled_seqs = schedule_outputs.get()
            # print("GPU: start",time.time())
            next_tokens = model_runner.step_once(seqs=scheduled_seqs, temperature=0.6, top_p=0.9)
            # print("GPU: end",time.time())
            run_outputs.put_nowait((scheduled_seqs,next_tokens,model_runner.memory_manager.get_num_free_pages()))
            
    async def run_process_output_engine(self):
        # print("start output process")
        num_free_pages = self.model_runner.memory_manager.get_num_free_pages()
        self.control_schedule.put_nowait(num_free_pages)
        while True:
            self.control_schedule.put_nowait(num_free_pages)
            # print("OUTPUT: wait",time.time())
            while self.run_outputs.empty():
                await asyncio.sleep(0)
            outputs = self.run_outputs.get()
            # print("OUTPUT: start",time.time())
            scheduled_seqs:List[Sequence] = outputs[0]
            next_tokens:List[int] = outputs[1]
            num_free_pages = outputs[2]
            self.scheduler.update_seqs(scheduled_seqs, next_tokens)
            for seq in scheduled_seqs:
                self.async_streams[seq.seq_id].put(seq.detokenize_inc(self.model_runner.tokenizer))
            finished_seqs = []
            for seq in self.scheduler.finish_lists:
                self.async_streams[seq.seq_id].finish()
                del self.async_streams[seq.seq_id]
                finished_seqs.append(seq)
            self.free_requests(finished_seqs)
            # time.sleep(0.015)
            # print("OUTPUT: end",time.time())


    def start_gpu_engine(self):
        self.gpu_engine = self.ctx.Process(target=AsyncLLM.run_gpu_engine,args=(self.schedule_outputs,self.run_outputs,self.model_runner))
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
