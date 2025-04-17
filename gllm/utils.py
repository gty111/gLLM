import asyncio
import uuid
import torch
import zmq
import time
import sys
import os
import hashlib
import filelock
import tempfile
import logging

from logger import logger
from functools import partial
from typing import Awaitable, Callable, ParamSpec, TypeVar, Union, Optional
from pathlib import Path
from gllm.dist_utils import get_pp_rank

P = ParamSpec('P')
K = TypeVar("K")
T = TypeVar("T")

formater = logging.Formatter(f"[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s - %(message)s")
for handler in logger.handlers:
    handler.setFormatter(formater)

def make_async(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)


def make_socket(ctx, ipc_path: str, type):
    if type == zmq.PUSH:
        socket = ctx.socket(type)
        socket.connect(ipc_path)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, int(0.5 * 1024**3))
        return socket
    elif type == zmq.PULL:
        socket = ctx.socket(type)
        socket.bind(ipc_path)
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, int(0.5 * 1024**3))
        return socket
    else:
        assert 0

def wait_worker(mp_alive,num_worker):
    while True:
        num_worker_start = 0
        for i in mp_alive:
            if i==-1:
                sys.exit()
            num_worker_start += i
        if num_worker_start == num_worker:
            break
        time.sleep(1)
        
def check_worker_alive(mp_alive):
    for i in mp_alive:
        if i==-1:
            sys.exit()
            

temp_dir = tempfile.gettempdir()

def get_lock(model_name_or_path: Union[str, Path],
             cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock