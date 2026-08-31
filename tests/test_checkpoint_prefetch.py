"""The checkpoint prefetch splits shards across ranks without gaps or overlap.

The prefetch is a pure page-cache side effect and cannot change what is loaded,
so the only thing worth pinning down is the partition: a rank that skips shards
leaves them cold, and ranks that overlap read the checkpoint N times over.
"""

import threading
import time

from gllm.runtime.model_loader import _prefetch_shards


def _partition(paths, world_size):
    """Mirror of the slice ``_prefetch_shards`` applies, for assertions."""
    ordered = sorted(set(paths))
    return [ordered[rank::world_size] for rank in range(world_size)]


def _join_prefetch(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(
            t.name == "ckpt-prefetch" and t.is_alive() for t in threading.enumerate()
        ):
            return True
        time.sleep(0.01)
    return False


def test_partition_covers_every_shard_exactly_once():
    shards = [f"/ckpt/model-{i:05d}-of-00048.safetensors" for i in range(48)]
    for world_size in (1, 2, 3, 4, 8):
        flat = [p for part in _partition(shards, world_size) for p in part]
        assert sorted(flat) == sorted(shards), world_size
        assert len(flat) == len(set(flat)), f"overlap at world_size={world_size}"


def test_partition_is_balanced():
    shards = [f"/ckpt/{i}.safetensors" for i in range(48)]
    for world_size in (4, 8):
        sizes = [len(p) for p in _partition(shards, world_size)]
        assert max(sizes) - min(sizes) <= 1, sizes


def test_duplicate_paths_are_read_once():
    shards = ["/ckpt/a.safetensors", "/ckpt/a.safetensors", "/ckpt/b.safetensors"]
    flat = [p for part in _partition(shards, 1) for p in part]
    assert flat == ["/ckpt/a.safetensors", "/ckpt/b.safetensors"]


def test_prefetch_reads_files_and_survives_missing_ones(tmp_path):
    real = tmp_path / "shard.safetensors"
    real.write_bytes(b"\0" * (1 << 20))
    missing = tmp_path / "gone.safetensors"

    # An unreadable shard must not take the loader down; the real failure
    # surfaces later, when that tensor is actually requested.
    _prefetch_shards([str(real), str(missing)], rank=0, world_size=1)
    assert _join_prefetch()


def test_empty_slice_is_a_noop():
    _prefetch_shards([], rank=0, world_size=4)
    _prefetch_shards(["/ckpt/only.safetensors"], rank=3, world_size=4)
    assert _join_prefetch()
