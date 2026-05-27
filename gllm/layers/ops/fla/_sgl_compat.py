"""Minimal compatibility shim for the Triton kernels vendored from sglang.

The kernels in :mod:`gllm.layers.ops.fla` and :mod:`gllm.layers.ops.mamba` are
copied verbatim from sglang's flash-linear-attention port and depend on a
handful of utility symbols that originally live in :mod:`sglang.srt.utils`
and :mod:`sglang.srt.server_args`. To keep the in-tree port self-contained
we replicate just those symbols here. Bulk-copying the kernels and rewriting
the imports to point at this shim is materially safer than hand-porting
8000+ lines of Triton.
"""

from __future__ import annotations

import contextlib
import importlib.metadata as _md
from functools import lru_cache
from typing import Tuple

import torch


def cdiv(a: int, b: int) -> int:
    """Integer ceiling division. Mirrors ``sglang.srt.utils.cdiv``."""
    return -(-a // b)


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= ``n``. Mirrors ``sglang.srt.utils.next_power_of_2``."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


@lru_cache(maxsize=1)
def is_cpu() -> bool:
    return not torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_npu() -> bool:
    # gllm targets NVIDIA/AMD GPUs only; mirror sglang's flag so the
    # vendored kernels keep their NPU code-paths disabled.
    return False


@lru_cache(maxsize=1)
def cpu_has_amx_support() -> bool:
    return False


@contextlib.contextmanager
def device_context(device):
    """Context manager that pins the current CUDA device. Mirrors
    ``sglang.srt.utils.device_context``.
    """
    if device is None or not torch.cuda.is_available():
        yield
        return
    if isinstance(device, torch.Tensor):
        device = device.device
    if hasattr(device, "type") and device.type != "cuda":
        yield
        return
    with torch.cuda.device(device):
        yield


@lru_cache(maxsize=1)
def _torch_release() -> Tuple[int, int]:
    """Parse the major/minor of the installed ``torch`` version.

    Mirrors ``sglang.srt.utils.common.torch_release`` so the FLA ``utils.py``
    can keep its version-gated autotune branches.
    """
    try:
        v = _md.version("torch")
    except Exception:  # pragma: no cover - defensive
        v = torch.__version__
    parts = []
    for token in v.split("+", 1)[0].split("."):
        try:
            parts.append(int(token))
        except ValueError:
            break
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])


# Match the name FLA's ``utils.py`` actually imports.
torch_release = _torch_release()


class _ServerArgsShim:
    """Stand-in for ``sglang.srt.server_args.get_global_server_args()``.

    The vendored ``layernorm_gated.py`` only reads
    ``disable_piecewise_cuda_graph`` to decide whether to register the kernel
    as a custom op. gllm does not have piecewise-CUDA-graph mode, so the
    safe default is "True" — i.e. always go through the eager path and skip
    the custom-op registration. This matches the behaviour you'd get from
    sglang with ``--disable-piecewise-cuda-graph``.
    """

    disable_piecewise_cuda_graph: bool = True


_SHIM_INSTANCE = _ServerArgsShim()


def get_global_server_args() -> _ServerArgsShim:
    return _SHIM_INSTANCE
