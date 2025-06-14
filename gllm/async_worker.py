import asyncio

from gllm.worker import Worker

def async_wrapper(func):
    async def wrapper(*args, **kwargs):
        while True:
            await func(*args, **kwargs)
            await asyncio.sleep(0)
    return wrapper

# Async wrapper for worker
class AsyncWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @async_wrapper
    async def run_driver(self):
        return super().run_driver()
    
    @async_wrapper
    async def run_first_tp(self):
        return super().run_first_tp()
    
    @async_wrapper
    async def run_other(self):
        return super().run_other()

class AsyncTasks():
    def __init__(self):
        self.tasks = []
        
    def add_task(self, func):
        self.tasks.append(asyncio.get_event_loop().create_task(func()))
        
    async def wait(self):
        await asyncio.gather(*self.tasks)

async def launch_async_tasks(worker: AsyncWorker):
    worker.init()

    ats = AsyncTasks()
    if worker.rank == 0:
        ats.add_task(worker.run_driver)
    elif worker.pp_rank == 0:
        ats.add_task(worker.run_first_tp)
    else:
        ats.add_task(worker.run_other)
    await ats.wait()


def run_worker_async(worker: AsyncWorker):
    try:
        asyncio.run(launch_async_tasks(worker))
    except KeyboardInterrupt as e:
        worker.handle_keyboardInterrupt()
    except Exception as e:
        worker.handle_exception(e)
