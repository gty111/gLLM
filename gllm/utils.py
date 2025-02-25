import asyncio
import uuid
import torch
import zmq

from functools import partial
from typing import Awaitable, Callable, ParamSpec, TypeVar, Union

P = ParamSpec('P')
K = TypeVar("K")
T = TypeVar("T")

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
