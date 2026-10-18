"""Custom distributed primitives for gLLM.

Currently provides ``CustomAllreduce``, an NVLink P2P fast path for small
TP all-reduces that bypasses NCCL's RING_LL kernel. See
``custom_all_reduce.py`` for the implementation.
"""

from gllm.distributed.custom_all_reduce import (
    CustomAllreduce,
    get_custom_allreduce,
    init_custom_allreduce,
    shutdown_custom_allreduce,
)

__all__ = [
    "CustomAllreduce",
    "get_custom_allreduce",
    "init_custom_allreduce",
    "shutdown_custom_allreduce",
]
