import asyncio
import uuid
from functools import partial
from typing import Awaitable, Callable, ParamSpec, TypeVar

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