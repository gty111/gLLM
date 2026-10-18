# Native cache arena metadata

The arena uses one Rust allocator for ordinary and prefix-retained cache slots.
Rust owns physical occupancy, group slot indexes, retention ages and allocation
priority. Rust also owns the prefix hash index, per-request incremental hash
chains, KV reference counts, snapshot slot assignments, publication flags,
restore pins and snapshot LRU. Python only coordinates tensor resets/copies
and invokes the allocator; it does not maintain shadow ownership tables. The
extension handles CPU metadata and does not access tensors or synchronize CUDA.

Before any prefix is retained, an ordered bitset selects free slots. The first
`mark_cached` call switches the same allocator to age-aware heaps. Uncached
slots remain the first allocation candidates, followed by older retained slots;
`prefer_high` still controls the slot-order tie break. Releasing a request is
synchronous: all ownership and candidate indexes are updated before returning.

## Build

Install Rust (minimum 1.83) and use the normal Linux package installation:

```sh
pip install -e . --no-deps
```

The PEP 517 build environment installs `setuptools-rust`; Cargo compiles the
PyO3 extension in release mode. Source distributions include the Cargo sources
and lockfile. Wheels carry the Python 3.9+ stable ABI extension.

For an existing environment with build isolation disabled, install
`setuptools-rust>=1.10,<2` first. Rust and the compiled extension are required;
there is no Python allocator fallback.

On hosts that need a mirror, configure the toolchain before installing gLLM:

```sh
export RUSTUP_DIST_SERVER=https://rsproxy.cn
export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
```

Add the following to `$CARGO_HOME/config.toml` (normally `~/.cargo/config.toml`):

```toml
[source.crates-io]
replace-with = "rsproxy"

[source.rsproxy]
registry = "sparse+https://rsproxy.cn/index/"
```

The mirror is a host setting and is not hardcoded into the package build.

## Checks

The arena always uses the Rust extension. Import fails immediately when the
extension is missing or incompatible; there is no backend selection setting.
The Python adapter exposes batch operations and policy callbacks without
maintaining a second ownership map.

```sh
pytest -q \
  tests/test_cache_arena.py tests/test_native_cache_arena.py \
  tests/test_memory_manager_arena.py tests/test_cache_capacity.py \
  tests/test_prefix_cache_reuse.py tests/test_cache_arena_granularity.py
```

Native cache-type properties such as `free_slots`, `live_slots` and
`retained_slots` return diagnostic snapshots. Use allocator batch operations to
change ownership; mutating a snapshot does not update the allocator.

## Physical page granularity

`MemoryManager._cache_layouts()` collects persistent cache layouts before the
arena is allocated. The physical page size is the greatest common divisor of
those layouts' logical entry sizes, with a minimum of 1 KiB and scalar alignment.
When the GCD falls below that floor, each logical slot is rounded up to whole
physical pages; its tensor shapes and packed bank offsets are preserved.
New persistent cache groups must be included in this layout plan.

For a 1 MiB target KV entry, a 320 KiB draft KV entry and a 153944064-byte
recurrent state, the grain is 64 KiB. Target entries occupy 16 physical pages;
draft entries occupy five. Each group retains its own logical page IDs and
fixed tensor strides. The grain is computed once at initialization and remains
fixed for the lifetime of the arena, including later cache registrations.

Across ranks, initialization agrees on logical target KV slot capacity rather
than physical page count: different PP-local layouts can choose different
physical grains. Allocation is rounded down to a whole number of target slots,
so the backing allocation stays within each rank's budget. This is startup-only
CPU layout arithmetic; it adds no device synchronization to request processing.

## Prefix cache metadata

`PrefixCore` and `PrefixHashes` are part of the required `_cache_arena`
extension. Prefix scans stay lazy: only complete pages through the first miss
are hashed. Hash keys retain Python's integer tuple-hash semantics and the
first-eight-token canary, including multimodal content-derived IDs. A native
hash chain holds its source list and invalidates on source replacement; it
supports independent deep copies and pickling for scheduler/request transport.
As before, already-hashed token boundaries must remain unchanged; decode may
append committed tokens beyond those boundaries.

Lookup and arena retain execute in one native operation. All hit pages are
pinned before Python may allocate recurrent working state or reclaim snapshots.
Release validates the whole reference batch first, then updates references,
retention ages and physical ownership without sending intermediate page lists
back through Python. The existing policy remains: use clean slots first, then
older cached requests, preserving the beginning of each prefix longest.

Snapshot publication and restore pins are separate from KV references. Pending
writes never enter the reclaimable LRU; only the assigned slot can be published.
Nested pins protect restore sources, and their final unpin returns published
snapshots to the newest end. An intrusive native list provides constant-time
LRU updates without stale heap entries. Python enqueues the tensor reset before
releasing the arena extent; Rust borrows end before callbacks can reenter cache
metadata. PP followers mirror slot assignments in a validated native batch.
PP payload generation reads slots by index without copying the entire table.

Prefix properties such as `page_ref_num`, `hash2page`, `page2ssm_snapshot` and
`_ssm_snapshot_lru` are diagnostic snapshots, just like the arena properties.
Mutating returned containers does not change ownership. Runtime code uses
explicit native operations; there is no Python metadata backend or switch.

```sh
pytest -q tests/test_native_prefix_cache.py tests/test_prefix_cache_reuse.py \
  tests/test_ssm_snapshot_replacement.py tests/test_ssm_snapshot_publication.py \
  tests/test_cache_capacity.py tests/test_worker_nonoverlap.py
python benchmarks/benchmark_prefix_cache_metadata.py --tokens 32768 144000
```

The benchmark uses CPU backing tensors and full-prefix hits. Fresh-request
lookup includes hash generation; cached-hash lookup excludes it. Both include
arena retain, and release includes retention policy and physical ownership
updates. These are metadata measurements, not TTFT or model throughput.
