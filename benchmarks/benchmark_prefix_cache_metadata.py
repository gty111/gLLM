"""CPU-only prefix metadata benchmark; no model or GPU allocations.

Run from an installed checkout, optionally pinning the process with taskset.
The retained prefix covers the whole input; hashing stops at the first miss
for ordinary requests. Timings include arena ownership operations.
"""
import argparse
import json
import statistics
import time

import torch

from gllm.runtime.cache_arena import CacheArena
from gllm.runtime.memory_manager import MemoryManager, PrefixSegment
from gllm.runtime.sequence import GenerationSequence


def measure(tokens, repetitions, warmup):
    page_size = 16
    base = MemoryManager(0.85, num_layers=1, dtype=torch.bfloat16,
                         page_size=page_size, kv_head_num=1, kv_head_dim=8,
                         vocab_size=1024)
    layout = base._kv_cache_layout()
    arena = CacheArena(torch.empty((tokens // page_size + 64) * layout.entry_bytes,
                                  dtype=torch.uint8, device="cpu"),
                       physical_page_bytes=layout.entry_bytes)
    cache = arena.register_cache(layout)
    segment = PrefixSegment(1, page_size, 1, 8, False, cache)
    arena.allocator.set_evictor(cache.name, segment.evict_arena_slot)
    source = list(range(tokens))
    resident = GenerationSequence(0, source.copy(), [], output_len=1)
    resident.page_table = [segment.allocate(resident, end)
                           for end in range(page_size, tokens + 1, page_size)]
    segment.free_many(resident.page_table)
    samples = {"fresh_request_lookup_ms": [], "cached_hash_lookup_ms": [],
               "prefix_release_ms": []}
    for index in range(warmup + repetitions):
        seq = GenerationSequence(index + 1, source.copy(), [], output_len=1)
        start = time.perf_counter()
        pages = segment.retain_computed_prefix(seq)
        fresh = time.perf_counter() - start
        assert len(pages) == tokens // page_size
        start = time.perf_counter()
        segment.free_many(pages)
        release = time.perf_counter() - start
        start = time.perf_counter()
        pages = segment.retain_computed_prefix(seq)
        cached = time.perf_counter() - start
        segment.free_many(pages)
        assert arena.allocator.num_used_physical_pages == 0
        if index >= warmup:
            for key, value in zip(samples, (fresh, cached, release)):
                samples[key].append(value * 1000)
    print(json.dumps({"input_tokens": tokens, "page_size": page_size,
                      "repetitions": repetitions, "device": "cpu",
                      **{key: round(statistics.median(values), 3)
                         for key, values in samples.items()}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32768, 144000])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0 or any(n < 16 or n % 16 for n in args.tokens):
        parser.error("tokens must be positive multiples of 16; repetitions >= 1 and warmup >= 0")
    for tokens in args.tokens:
        measure(tokens, args.repetitions, args.warmup)
