"""Overlap decode applies GPU-resolved tokens to repetition penalties without host waits."""
from types import SimpleNamespace

import pytest
import torch

from gllm.runtime.memory_manager import MemoryManager
from gllm.runtime.sequence import GenerationSequence

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _manager(vocab=64, capacity=4):
    manager = MemoryManager.__new__(MemoryManager)
    manager.dtype = torch.bfloat16
    manager.vocab_size = vocab
    manager.max_running_seqs = capacity
    manager._rep_pool = None
    manager._rep_free_slots = None
    return manager


@cuda
def test_resolved_tokens_update_pool_and_mask_without_host_sync():
    manager = _manager()
    a = GenerationSequence(1, [3], [], output_len=16, repetition_penalty=1.05)
    neutral = GenerationSequence(2, [5], [], output_len=16, repetition_penalty=1.0)
    b = GenerationSequence(3, [9], [], output_len=16, repetition_penalty=0.8)
    seqs = [a, neutral, b]
    mask = manager.build_repetition_penalty_mask(seqs)
    # The CPU view holds FutureMap placeholders; the real tokens live on GPU.
    input_data = SimpleNamespace(
        repetition_penalty=mask,
        num_decodes=len(seqs),
        seqs=seqs,
        tokens=torch.tensor([13, 17, 19], device="cuda"),
    )
    torch.cuda.synchronize()
    # This runs right after the forward is enqueued; any blocking copy here
    # stalls the host for the whole forward and serializes the overlap pipeline.
    torch.cuda.set_sync_debug_mode("error")
    try:
        manager.apply_resolved_repetition_tokens(input_data)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    pen_a = torch.tensor(1.05, dtype=manager.dtype)
    pen_b = torch.tensor(0.8, dtype=manager.dtype)
    assert mask[0, 13].cpu() == pen_a and mask[0, 3].cpu() == pen_a
    assert mask[2, 19].cpu() == pen_b and mask[2, 9].cpu() == pen_b
    assert torch.all(mask[1] == 1)
    assert manager._rep_pool[a.rep_slot, 13].cpu() == pen_a
    assert manager._rep_pool[b.rep_slot, 19].cpu() == pen_b
