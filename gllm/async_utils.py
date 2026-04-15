"""
Async scheduling utilities for gLLM.

This module provides GPU-side circular buffers and synchronization primitives
for non-blocking async scheduling patterns inspired by SGLang's event_loop_overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class FutureIndices:
    """Represents allocated future token slots in the circular buffer.
    
    Attributes:
        indices: Tensor of indices pointing to allocated buffer slots
        interval: Optional slice representing the buffer interval for direct access
    """
    indices: torch.Tensor
    interval: Optional[slice] = None


class FutureMap:
    """GPU circular buffer for storing future token predictions.
    
    This implements a GPU-side circular buffer that stores sampled token IDs
    from one batch to be consumed by the next batch without CPU synchronization.
    
    The mechanism works as follows:
    1. Batch N is executed, producing next_token_ids
    2. These tokens are stored in circular buffer at allocated indices
    3. Batch N+1 is prepared with references (-1, -2, -3) to these indices
    4. On GPU forward, these negative indices are replaced with actual buffer lookups
    5. GPU never needs to wait for CPU to process results
    
    This enables true CPU/GPU overlap without synchronization stalls.
    """
    
    def __init__(
        self,
        max_running_requests: int,
        context_len: int = 4096,
        chunked_prefill_size: Optional[int] = None,
        device: str | torch.device = "cuda:0",
    ):
        """Initialize the FutureMap circular buffer.
        
        Args:
            max_running_requests: Maximum number of concurrent requests
            context_len: Maximum context length for sequences
            chunked_prefill_size: Prefill chunk size (for calculating max chunks)
            device: GPU device to allocate buffer on
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Calculate buffer size based on SGLang's formula
        # The circular buffer needs space for:
        # - Running decode batch (1)
        # - Multiple prefill chunks (calculated from context_len / chunked_prefill_size)
        # - Safety margin for wraparound
        
        if chunked_prefill_size is not None and chunked_prefill_size > 0:
            max_num_chunks = (context_len + chunked_prefill_size - 1) // chunked_prefill_size
        else:
            max_num_chunks = 1
        
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large
        # This provides extra slots for wraparound edge cases
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        
        # Current position in circular buffer (incremented with modulo arithmetic)
        self.future_ct = 0
        
        # Allocate GPU tensor for storing token IDs
        # int64 for compatibility with PyTorch indexing
        self.token_ids_buf = torch.empty(
            (self.future_buffer_len,),
            dtype=torch.int64,
            device=self.device,
        )
        
        logger.debug(
            f"FutureMap initialized: "
            f"future_limit={self.future_limit}, "
            f"future_buffer_len={self.future_buffer_len}, "
            f"buffer_memory={self.future_buffer_len * 8 / 1024:.1f} KB"
        )
    
    def alloc_future_indices(self, batch_size: int) -> FutureIndices:
        """Allocate future indices for a batch.
        
        Uses circular wraparound arithmetic to reuse buffer space. The allocated
        indices are negative values that serve as placeholders during batch
        preparation. They are resolved to actual buffer lookups during forward pass.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            FutureIndices with allocated indices and buffer interval
        """
        cur_future_ct = self.future_ct
        
        # Advance position with modulo wraparound
        self.future_ct = (cur_future_ct + batch_size) % self.future_limit
        
        # Calculate buffer interval (1-indexed for negative offset encoding)
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + batch_size
        
        # Create indices tensor on GPU
        indices = torch.arange(
            start, end,
            dtype=torch.int64,
            device=self.device,
        )
        
        return FutureIndices(
            indices=indices,
            interval=slice(start, end),
        )
    
    def resolve_future(
        self,
        input_ids: torch.Tensor,
    ) -> None:
        """Resolve future token references to actual buffer values (GPU D2D).

        Replaces all negative index placeholders in input_ids with actual
        token values from the circular buffer.  Positive values pass through
        unchanged.

        IMPORTANT: This must be a pure GPU operation with NO CPU
        synchronization.  We use an unconditional torch.where instead of
        mask.any() (which would trigger cudaStreamSynchronize) so the GPU
        kernel queue is never stalled.

        Args:
            input_ids: GPU tensor with potential negative indices (in-place)
        """
        # Unconditional D2D: negative → buffer lookup, non-negative → keep.
        # torch.where launches a single elementwise kernel, no CPU sync.
        input_ids[:] = torch.where(
            input_ids < 0,
            self.token_ids_buf[torch.clamp(-input_ids, min=0)],
            input_ids,
        )
    
    def store_to_map(
        self,
        future_indices: FutureIndices,
        next_token_ids: torch.Tensor,
    ) -> None:
        """Store computed token IDs in the circular buffer.
        
        Called after GPU forward+sampling to store results for the next batch
        to reference. Uses the allocated interval from alloc_future_indices.
        
        Args:
            future_indices: FutureIndices returned from alloc_future_indices
            next_token_ids: Tensor of sampled token IDs to store
        """
        intv = future_indices.interval
        
        if intv is None:
            raise ValueError("FutureIndices must have interval set")
        
        # Store tokens in the allocated buffer interval
        self.token_ids_buf[intv] = next_token_ids
    
    def is_empty(self) -> bool:
        """Check if buffer has wrapped around (all slots reused).
        
        Returns:
            True if future_ct is at initial position
        """
        return self.future_ct == 0
    
    def reset(self) -> None:
        """Reset buffer to initial state.
        
        Used when restarting request batches or for testing.
        """
        self.future_ct = 0
        logger.debug("FutureMap reset to initial state")


class AsyncSchedulerContext:
    """Context manager for async scheduling operations.
    
    Manages CUDA streams and synchronization for overlapped CPU/GPU execution.
    """
    
    def __init__(
        self,
        schedule_stream_priority: int = 0,
        device: str | torch.device = "cuda:0",
    ):
        """Initialize async scheduler context.
        
        Args:
            schedule_stream_priority: Priority for schedule stream (0=high, -1=low)
            device: GPU device to use
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # High priority stream for CPU scheduling work
        self.schedule_stream = torch.cuda.Stream(
            device=self.device,
            priority=schedule_stream_priority,
        )
        
        # Default priority stream for GPU forward pass
        self.forward_stream = torch.cuda.Stream(device=self.device)
        
        # Optional copy stream for H2D/D2H transfers
        self.copy_stream = torch.cuda.Stream(device=self.device)
        
        logger.debug(
            f"AsyncSchedulerContext initialized on {self.device} "
            f"with schedule_stream_priority={schedule_stream_priority}"
        )
    
    def synchronize(self) -> None:
        """Synchronize all streams."""
        torch.cuda.synchronize(device=self.device)
    
    def synchronize_forward_stream(self) -> None:
        """Synchronize only the forward stream."""
        self.forward_stream.synchronize()
    
    def synchronize_schedule_stream(self) -> None:
        """Synchronize only the schedule stream."""
        self.schedule_stream.synchronize()
