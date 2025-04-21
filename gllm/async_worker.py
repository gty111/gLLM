import pickle
import asyncio

from logger import logger
from typing import List

from gllm.input_data import InputData
from gllm.sequence import Sequence
from gllm.dist_utils import send_pp_data, recv_pp_data
from gllm.scheduler import SchedulerOutput
from gllm.utils import make_async
from gllm.worker import Worker


def async_wrapper(func):
    async def wrapper(*args, **kwargs):
        while True:
            await func(*args, **kwargs)
            await asyncio.sleep(0)
    return wrapper

# Used with PipeAsyncLLM
class AsyncWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(args,kwargs)

    # rank except 0
    @async_wrapper
    async def recv_schedule_seqs(self):
        if self.gpu_schedule_socket.poll(timeout=0) != 0:
            recv_bytes = self.gpu_schedule_socket.recv(copy=False)
            seqs = pickle.loads(recv_bytes)
            self.schedule_queue.append(
                InputData(seqs, self.model_runner.memory_manager))

    # rank except 0
    @async_wrapper
    async def recv_intermediate_data(self):
        if len(self.schedule_queue) != 0:
            input_data = self.schedule_queue.popleft()
            intermediate_data = await make_async(recv_pp_data)(
                self.get_pp_last_rank(), self.dtype,
                [input_data.tokens.shape[0], self.hidden_size], self.ret_residual)
            self.run_queue.append((input_data, intermediate_data))

    # rank except 0
    @async_wrapper
    async def run(self):
        # model forward
        if len(self.run_queue) != 0:
            hidden_states = None
            residual = None

            if self.ret_residual:
                input_data, (hidden_states_future, residual_future,
                             hidden_states, residual) = self.run_queue[0]
                if not hidden_states_future.is_completed() or not residual_future.is_completed():
                    return
            else:
                input_data, (hidden_states_future,
                             hidden_states) = self.run_queue[0]
                if not hidden_states_future.is_completed():
                    return

            self.run_queue.popleft()

            output = self.model_runner.step_once(
                input_data, hidden_states, residual)
            if self.pp_rank == self.pp_size - 1:
                assert type(output) == list
                token_bytes = pickle.dumps(output)
                self.token_socket.send(token_bytes, copy=False)
            else:
                send_pp_data(output, self.get_pp_next_rank())
                
    @async_wrapper
    async def recv_requests(self):
        super().recv_requests()

    # rank 0
    @async_wrapper
    async def schedule_run(self):
        if len(self.seqs_to_decode) + len(self.seqs_to_prefill) != 0 and len(self.batch_running) < self.pp_size:
            schedule_seqs = self.schedule() if not self.use_naive_schedule else self.schedule_naive()
            if len(schedule_seqs) != 0:
                input_data = InputData(
                    schedule_seqs, self.model_runner.memory_manager)
                if self.pp_size > 1:
                    seqs_bytes = pickle.dumps(schedule_seqs)
                    for i in range(1, self.pp_size):
                        self.gpu_schedule_socket[i - 1].send(seqs_bytes, copy=False)
                self.batch_running.append(schedule_seqs)
                output = self.model_runner.step_once(input_data)

                if type(output) != list:
                    send_pp_data(output, self.get_pp_next_rank())
                else:
                    self.next_tokens_queue.append(output)

    # rank 0
    @async_wrapper
    async def recv_next_tokens(self):
        if self.pp_size != 1:  # recv tokens from last rank
            recv_bytes = await make_async(self.token_socket.recv)(copy=False)
            next_tokens = pickle.loads(recv_bytes)
            self.next_tokens_queue.append(next_tokens)

    # rank 0
    @async_wrapper
    async def process_output(self):
        if len(self.next_tokens_queue) != 0:
            next_tokens = self.next_tokens_queue.popleft()

            schedule_seqs: List[Sequence] = self.batch_running.popleft()
            assert len(next_tokens) == len(schedule_seqs)

            send_tokens = []
            schedulerOutput = SchedulerOutput([])

            for idx, seq in enumerate(schedule_seqs):
                seq.computed_token_num += seq.to_compute_token_num
                if seq.computed_prompt():
                    schedulerOutput.act_schedule_ids.append(seq.seq_id)
                    send_tokens.append(next_tokens[idx])
                    seq.token_ids.append(next_tokens[idx])
                if seq.is_finish():
                    schedulerOutput.free_ids.append(seq.seq_id)
                    self.model_runner.memory_manager.free(seq)
                elif seq.computed_prompt():
                    self.seqs_to_decode.appendleft(seq)
                else:
                    self.seqs_to_prefill.appendleft(seq)

            output_bytes = pickle.dumps((schedulerOutput, send_tokens))
            self.output_socket.send(output_bytes, copy=False)

def create_worker_task(func):
    return asyncio.get_event_loop().create_task(func())

async def run_worker_async(worker: AsyncWorker):
    worker.init()
    logger.info(f'Init')

    tasks = []
    if worker.pp_rank == 0:
        tasks.append(create_worker_task(worker.recv_requests))
        tasks.append(create_worker_task(worker.schedule_run))
        tasks.append(create_worker_task(worker.process_output))
        tasks.append(create_worker_task(worker.recv_next_tokens))
    else:
        tasks.append(create_worker_task(worker.recv_schedule_seqs))
        tasks.append(create_worker_task(worker.recv_intermediate_data))
        tasks.append(create_worker_task(worker.run))
    await asyncio.gather(*tasks)


def run_worker_async(worker: AsyncWorker):
    try:
        asyncio.run(run_worker_async(worker))
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
