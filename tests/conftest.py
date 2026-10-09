"""Shared factories for the scheduler/cache-pressure test scaffolding.

The fixtures below each return a factory function so tests can keep calling
them with arguments, e.g. ``prefix_manager(pages=4)`` or ``make_seq(1, 64)``.
gLLM imports stay inside the factory bodies so collecting an unrelated test
module never pays for the runtime stack.
"""
from types import SimpleNamespace

import pytest


def _make_seq(seq_id, num_tokens, output=(999,), output_len=16):
    """A prompt of deterministic token ids waiting to be scheduled."""
    from gllm.runtime.sequence import GenerationSequence
    return GenerationSequence(
        seq_id, list(range(num_tokens)), list(output), output_len=output_len
    )


def _populate(segment, seq):
    """Allocate one page per 16-token span so the whole prompt is cached."""
    seq.page_table = [segment.allocate(seq, end)
                      for end in range(16, len(seq.token_ids) + 1, 16)]


def _prefix_manager(pages=128):
    """A one-layer PrefixMemoryManager backed by a real arena of `pages` pages."""
    import torch

    from gllm.runtime.cache_arena import CacheArena
    from gllm.runtime.memory_manager import (
        MemoryManager, PrefixMemoryManager, PrefixSegment,
    )

    base = MemoryManager(0.85, num_layers=1, dtype=torch.bfloat16, page_size=16,
                         kv_head_num=1, kv_head_dim=8, vocab_size=1024)
    layout = base._kv_cache_layout()
    arena = CacheArena(torch.empty(pages * layout.entry_bytes, dtype=torch.uint8),
                       physical_page_bytes=layout.entry_bytes)
    cache = arena.register_cache(layout)
    segment = PrefixSegment(1, 16, 1, 8, False, cache)
    arena.allocator.set_evictor(cache.name, segment.evict_arena_slot)
    manager = PrefixMemoryManager.__new__(PrefixMemoryManager)
    manager.segment = segment
    manager.page_size = 16
    manager.ssm_segment = manager.dsv4_state_segment = None
    manager.num_allocated_pages = manager.num_hit_pages = 0
    return manager, arena, cache


def _make_scheduler(manager, method='chunked_prefill'):
    """A Scheduler over `manager` with the standard fake model runner."""
    from gllm.scheduling.scheduler import Scheduler

    manager._rep_pool = None
    manager._pending_ssm_restores = {}
    freed = []

    def free(seq):
        freed.append(seq.seq_id)
        manager.free(seq)

    runner = SimpleNamespace(
        memory_manager=manager, maxd=8, maxp=128, max_num_batched_tokens=128,
        minp=1, iterp=1, page_size=16, init_new_token_ratio=0.1,
        min_new_token_ratio=0.1, mtp_enabled=False, free=free,
        _mm_precompute_hash=lambda seq: None, disagg_prefill_limit=lambda seq: None,
        register_decode_page_hash=manager.register_decode_boundary,
    )
    scheduler = Scheduler(1, runner, method)
    scheduler.log = False
    scheduler.freed = freed
    return scheduler


def _stalled_requests(*, method="chunked_prefill", count=2, pages_each=5):
    """A scheduler whose decoding requests jointly exhaust the page pool."""
    from gllm.runtime.sequence import GenerationSequence

    mm, _, _ = _prefix_manager(pages=count * pages_each)
    s = _make_scheduler(mm, method)
    requests = []
    for sid in range(1, count + 1):
        request = GenerationSequence(sid, [sid] * 128, [], output_len=3)
        request.page_table = [
            mm.segment.allocate(request, end)
            for end in range(16, pages_each * 16 + 1, 16)
        ]
        request.computed_token_num = pages_each * 16
        requests.append(request)
    s.add_new_requests(requests)
    return s, requests


def _make_mm_runner(uses_mrope, tuple_output):
    """A ModelRunner stub whose embedding table is a known arange weight."""
    import torch

    from gllm.runtime.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.uses_mrope = uses_mrope
    runner.hidden_size = 4
    runner.input_hidden_states = torch.empty((32, 4))
    runner.embedding_cache = {}
    runner._init_disagg_state()
    weight = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
    calls = []

    def embed(ids, media, mask):
        calls.append(ids.clone())
        value = torch.nn.functional.embedding(ids.masked_fill(mask, 0), weight)
        deepstack = None
        if media is not None:
            rows = torch.cat(media)
            value[mask] = rows[:, :4]
            if rows.shape[1] > 4:
                deepstack = torch.zeros((1, ids.numel(), 4), dtype=value.dtype)
                deepstack[0, mask] = rows[:, 4:]
        return (value, deepstack) if tuple_output else value

    runner.model = SimpleNamespace(
        get_mm_placeholder_token_ids=lambda: [126, 127],
        embed_input_ids=embed,
    )
    return runner, weight, calls


@pytest.fixture
def make_seq():
    return _make_seq


@pytest.fixture
def populate():
    return _populate


@pytest.fixture
def prefix_manager():
    return _prefix_manager


@pytest.fixture
def make_scheduler():
    return _make_scheduler


@pytest.fixture
def stalled_requests():
    return _stalled_requests


@pytest.fixture
def make_mm_runner():
    return _make_mm_runner
