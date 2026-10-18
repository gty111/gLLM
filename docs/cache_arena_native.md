# Native cache arena metadata

The arena uses one Rust allocator for ordinary and prefix-retained cache slots.
Rust owns physical occupancy, group slot indexes, retention ages and allocation
priority. Existing Python prefix lookup and eviction callbacks retain their
policy; they do not maintain a second arena ownership map. The extension only
handles CPU metadata and does not access tensors or synchronize CUDA.

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
`setuptools-rust>=1.10,<2` first. A source-only development installation is
available with `GLLM_BUILD_RUST=0 pip install -e . --no-deps`.

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

## Backend selection and checks

`GLLM_CACHE_ARENA_BACKEND=auto` (the default) selects Rust when the extension is
installed, otherwise the Python reference. `rust` requires the extension and
fails immediately if it is missing. `python` selects the reference for
comparison. Import or ABI errors in an installed extension are not hidden by an
automatic fallback. Set the variable before importing gLLM or starting workers.

```sh
GLLM_CACHE_ARENA_BACKEND=rust pytest -q \
  tests/test_cache_arena.py tests/test_native_cache_arena.py \
  tests/test_memory_manager_arena.py tests/test_cache_capacity.py \
  tests/test_prefix_cache_reuse.py
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
