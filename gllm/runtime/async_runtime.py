"""GPU-side overlap scheduling primitives (FutureMap + CUDA streams)."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _resolve_future_kernel(ids_ptr, buf_ptr, n, BLOCK: tl.constexpr):
    """In-place substitution of FutureMap placeholders by their sampled token.

    A negative id ``-slot`` stands for "whatever the producer wrote into slot
    ``slot``"; anything else is already a real token.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(ids_ptr + offs, mask=mask, other=0)
    is_future = x < 0
    slot = tl.where(is_future, -x, 0)
    tok = tl.load(buf_ptr + slot, mask=mask & is_future, other=0)
    tl.store(ids_ptr + offs, tl.where(is_future, tok, x), mask=mask)


def resolve_future_inplace(input_ids: torch.Tensor, token_ids_buf: torch.Tensor):
    """Replace placeholders in ``input_ids`` with the tokens they reference.

    One launch rather than the six that the equivalent ``torch.where`` over a
    ``clamp``/gather expression compiles to.  That matters here and nowhere
    else: this runs at the launch boundary, where the GPU has just drained the
    previous forward and every extra launch is a gap the host has to fill.
    """
    n = input_ids.numel()
    if n == 0:
        return
    BLOCK = 256
    _resolve_future_kernel[(triton.cdiv(n, BLOCK),)](
        input_ids, token_ids_buf, n, BLOCK=BLOCK, num_warps=4
    )


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
        # PP stages before the sampler receive completed future values on a
        # dedicated feedback stream.  Keep the completion event per circular
        # slot so a dependent microbatch waits for exactly the producer it
        # consumes, rather than for every newer feedback receive already queued.
        self._slot_ready_events: list[Optional[torch.cuda.Event]] = [
            None
        ] * self.future_buffer_len

    def alloc_future_indices(self, batch_size: int) -> FutureIndices:
        cur = self.future_ct
        self.future_ct = (cur + batch_size) % self.future_limit
        start = cur + 1
        end = cur + 1 + batch_size
        # ``store_to_map`` and ``resolve_future`` only need ``interval``;
        # ``indices`` is lazily materialized on the rare paths that want it.
        return FutureIndices(indices=None, interval=slice(start, end))

    def resolve_future(self, input_ids: torch.Tensor) -> None:
        resolve_future_inplace(input_ids, self.token_ids_buf)

    def store_to_map(
        self, future_indices: FutureIndices, next_token_ids: torch.Tensor
    ) -> None:
        if future_indices.interval is None:
            raise ValueError("FutureIndices.interval is required")
        self.token_ids_buf[future_indices.interval] = next_token_ids

    def mark_ready(
        self, future_indices: FutureIndices, event: torch.cuda.Event
    ) -> None:
        """Associate remotely-produced future slots with their CUDA event."""
        interval = future_indices.interval
        if interval is None:
            raise ValueError("FutureIndices.interval is required")
        for slot in range(interval.start, interval.stop):
            self._slot_ready_events[slot] = event

    def wait_for_inputs(
        self, input_ids_cpu: torch.Tensor, stream: torch.cuda.Stream
    ) -> bool:
        """Wait only for remote futures referenced by this input batch.

        ``input_ids_cpu`` is the pinned staging tensor built by the scheduler,
        so discovering negative FutureMap placeholders is CPU-only and does not
        introduce a device synchronization.
        """
        slots = self.referenced_slots(input_ids_cpu)
        if not slots:
            return False
        seen = set()
        for slot in slots:
            event = self._slot_ready_events[slot]
            if event is None:
                raise RuntimeError(f"FutureMap slot {slot} has no ready event")
            event_id = id(event)
            if event_id not in seen:
                stream.wait_event(event)
                seen.add(event_id)
        return True

    @staticmethod
    def has_futures(input_ids_cpu: torch.Tensor) -> bool:
        """Return whether a pinned CPU token batch contains placeholders."""
        return bool(np.any(input_ids_cpu.numpy() < 0))

    @staticmethod
    def referenced_slots(input_ids_cpu: torch.Tensor) -> list[int]:
        """Return the distinct FutureMap slots a pinned token batch reads."""
        values = input_ids_cpu.numpy()
        return [int(slot) for slot in np.unique(-values[values < 0])]

    def reset(self) -> None:
        self.future_ct = 0
        self._slot_ready_events[:] = [None] * self.future_buffer_len


class OverlapRuntime:
    """CUDA streams used to overlap scheduling, forward, and D2H copies.

    Input H2D, model forward, and sampling share ``forward_stream``. The input
    buffers are shared across batches, so prep cannot legally begin before the
    previous forward has consumed them; FIFO ordering on one stream expresses
    that dependency without a pair of per-batch CUDA events. The host can still
    prepare and enqueue batch N+1 while batch N is executing.

    ``copy_stream`` handles sampled-token D2H, and ``feedback_stream`` carries
    PP sampled-token broadcasts independently of model execution.
    """

    def __init__(self, device: Union[torch.device, str]):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.forward_stream = torch.cuda.Stream(device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        # PP sampled-token feedback must not sit behind the next model forward.
        # A slot-specific event reconnects the dependency only when a later
        # microbatch actually consumes that future token.
        self.feedback_stream = torch.cuda.Stream(device=self.device)
