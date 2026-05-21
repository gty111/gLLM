"""GPU-side overlap scheduling primitives (FutureMap + CUDA streams)."""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class FutureIndices:
    # NOTE: ``indices`` is intentionally optional. Allocating a GPU tensor for
    # the future slot ids forced a ``.cpu().tolist()`` on the calling stream,
    # which inserted a hidden host-side sync on every batch. The slot ids are
    # purely a CPU concept (used by the scheduler) and ``interval`` carries the
    # information ``store_to_map`` and ``resolve_future`` actually need, so we
    # avoid materializing the GPU tensor by default.
    indices: Optional[torch.Tensor] = None
    interval: Optional[slice] = None


class FutureMap:
    """Circular GPU buffer for sampled token IDs consumed by the next batch."""

    def __init__(
        self,
        max_running_requests: int,
        context_len: int = 8192,
        chunked_prefill_size: Optional[int] = None,
        device: Union[torch.device, str] = "cuda:0",
    ):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if chunked_prefill_size and chunked_prefill_size > 0:
            max_num_chunks = (
                context_len + chunked_prefill_size - 1
            ) // chunked_prefill_size
        else:
            max_num_chunks = 1
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.future_ct = 0
        self.token_ids_buf = torch.empty(
            (self.future_buffer_len,), dtype=torch.int64, device=self.device
        )

    def alloc_future_indices(self, batch_size: int) -> FutureIndices:
        cur = self.future_ct
        self.future_ct = (cur + batch_size) % self.future_limit
        start = cur + 1
        end = cur + 1 + batch_size
        # ``store_to_map`` and ``resolve_future`` only need ``interval``;
        # ``indices`` is lazily materialized on the rare paths that want it.
        return FutureIndices(indices=None, interval=slice(start, end))

    def resolve_future(self, input_ids: torch.Tensor) -> None:
        input_ids[:] = torch.where(
            input_ids < 0,
            self.token_ids_buf[torch.clamp(-input_ids, min=0)],
            input_ids,
        )

    def store_to_map(
        self, future_indices: FutureIndices, next_token_ids: torch.Tensor
    ) -> None:
        if future_indices.interval is None:
            raise ValueError("FutureIndices.interval is required")
        self.token_ids_buf[future_indices.interval] = next_token_ids

    def reset(self) -> None:
        self.future_ct = 0


class OverlapRuntime:
    """CUDA streams used to overlap scheduling, forward, and D2H copies.

    Three streams form the GPU-side pipeline:

    * ``prep_stream``    — H2D copies of input metadata + (for VL) multimodal
                           embed work for the *next* batch. GPU-waits for the
                           previous forward via ``input_consumed_event`` so it
                           can be enqueued from the host without ever blocking.
    * ``forward_stream`` — model forward + sample for the current batch.
                           GPU-waits for ``prep_stream`` via
                           ``input_ready_event``.
    * ``copy_stream``    — D2H sample-id copy back to a pinned host buffer.

    With this pipeline the host thread never calls ``cudaEventSynchronize``
    between batches: every cross-stream dependency is expressed through events
    that the GPU itself blocks on, so dispatch for batch N+1 can race ahead of
    forward N as far as the schedule queue allows.
    """

    def __init__(self, device: Union[torch.device, str]):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.forward_stream = torch.cuda.Stream(device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.prep_stream = torch.cuda.Stream(device=self.device)
        # ``input_consumed_event``: signalled at the end of forward+sample on
        # ``forward_stream``; the next batch's ``prep_stream`` waits on it
        # before overwriting the shared input buffers.
        self.input_consumed_event = torch.cuda.Event()
        # ``input_ready_event``: signalled by ``prep_stream`` after H2D and any
        # VL embed work; ``forward_stream`` waits on it before reading the
        # input buffers.
        self.input_ready_event = torch.cuda.Event()
