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
import tqdm

from logger import logger
from functools import partial
from typing import (Awaitable, Callable, ParamSpec, TypeVar, Union, Optional, Dict, Any,
                    List, Tuple)
from pathlib import Path
from torch.library import Library

P = ParamSpec('P')
K = TypeVar("K")
T = TypeVar("T")

def init_logger():
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


def make_socket(ctx, path: str, type):
    if type == zmq.PUSH:
        socket = ctx.socket(type)
        socket.connect(path)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, int(0.5 * 1024**3))
        return socket
    elif type == zmq.PULL:
        socket = ctx.socket(type)
        socket.bind(path)
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

def get_model_load_pbar(num_totals):
    return tqdm.tqdm(total=num_totals,ncols=100,
                    bar_format='Loading model weights ... {l_bar}{bar}{r_bar}')
    
def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)

gllm_lib = Library("gllm", "FRAGMENT")  # noqa
 
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    tags: Tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    import torch.library

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

    my_lib = target_lib or gllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, "CUDA")
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
        
def get_device_name(device_id: int = 0) -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device_id)
    
def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

def ceil_div(a, b):
    return (a + b - 1) // b

def get_dtype_bytes(dtype):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    else:
        info = torch.iinfo(dtype)
    return info.bits // 8  # bits => bytes