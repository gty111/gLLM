import asyncio
import hashlib
import logging
import math
import os
import tempfile
import uuid
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
)

import filelock
import torch
import tqdm
import zmq
from logger import logger
from torch.library import Library

P = ParamSpec("P")
K = TypeVar("K")
T = TypeVar("T")


def init_logger():
    formatter = logging.Formatter(
        f"[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)


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


temp_dir = tempfile.gettempdir()


def get_lock(model_name_or_path: Union[str, Path], cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


def get_model_load_pbar(num_totals):
    return tqdm.tqdm(
        total=num_totals,
        ncols=100,
        bar_format="Loading model weights ... {l_bar}{bar}{r_bar}",
    )


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


def round_down(x: int, y: int) -> int:
    return (x // y) * y


def ceil_div(a, b):
    return (a + b - 1) // b


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def get_dtype_bytes(dtype):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    else:
        info = torch.iinfo(dtype)
    return info.bits // 8  # bits => bytes


def get_device_capability():
    device = torch.cuda.current_device()
    capability_arr = torch.cuda.get_device_capability(device)
    return capability_arr[0] * 10 + capability_arr[1]


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    from gllm.vllm_flash_attn.flash_attn_interface import (
        fa_version_unsupported_reason,
        is_fa_version_supported,
    )

    device_capability = get_device_capability()

    assert device_capability is not None

    # 1. default version depending on platform
    fa_version = (
        3 if (device_capability // 10 == 9 and is_fa_version_supported(3)) else 2
    )

    # 2. fallback for unsupported combinations
    if device_capability // 10 == 10 and fa_version == 3:
        logger.warning_once(
            "Cannot use FA version 3 on Blackwell platform "
            "defaulting to FA version 2."
        )
        fa_version = 2

    if requires_alibi and fa_version == 3:
        logger.warning_once(
            "Cannot use FA version 3 with ALiBi, " "defaulting to FA version 2."
        )
        fa_version = 2

    if not is_fa_version_supported(fa_version):
        logger.error(
            "Cannot use FA version %d is not supported due to %s",
            fa_version,
            fa_version_unsupported_reason(fa_version),
        )

    assert is_fa_version_supported(fa_version)
    return fa_version


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors
