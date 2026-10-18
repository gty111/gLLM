"""Common shared-expert/routed-expert CUDA stream overlap.

The runner is intentionally model agnostic: callers provide the routed and
shared callables, while this module owns stream lifetime and synchronization.
Keeping the auxiliary stream alive before CUDA graph capture also makes the
same path usable by eager, full-graph, and piecewise-graph forwards.
"""

from collections.abc import Callable
from typing import TypeVar

import torch


T = TypeVar("T")

# Above this measured cutoff the two GEMM pipelines tend to compete for the
# whole device instead of hiding the shared-expert work.
DEFAULT_SHARED_EXPERT_OVERLAP_TOKENS = 256

_AUX_STREAMS: dict[int, torch.cuda.Stream] = {}


def _shared_expert_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _AUX_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _AUX_STREAMS[index] = stream
    return stream


class SharedExpertRunner:
    """Execute a routed branch and a dense shared branch concurrently.

    One auxiliary stream is shared by every MoE layer on a device.  Each call
    joins that stream before returning, so layers remain ordered and tensors
    can be consumed or released safely.  Distributed MoE callers should pass
    ``allow_overlap=False`` unless their collective ordering explicitly
    supports this overlap.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        max_overlap_tokens: int = DEFAULT_SHARED_EXPERT_OVERLAP_TOKENS,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.stream = _shared_expert_stream(device)
        self.max_overlap_tokens = max_overlap_tokens

    def run(
        self,
        hidden_states: torch.Tensor,
        routed_fn: Callable[[], T],
        shared_fn: Callable[[], T],
        *,
        allow_overlap: bool = True,
    ) -> tuple[T, T]:
        if not allow_overlap or hidden_states.shape[0] > self.max_overlap_tokens:
            return routed_fn(), shared_fn()

        current = torch.cuda.current_stream(hidden_states.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            shared_output = shared_fn()

        routed_output = routed_fn()
        current.wait_stream(self.stream)
        return routed_output, shared_output
