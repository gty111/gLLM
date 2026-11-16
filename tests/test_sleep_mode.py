"""GPU tests for the sleep-mode VMM allocator (gllm.runtime.sleep_mode)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_KEEP_ALIVE = []


@pytest.fixture(scope="module")
def allocator():
    from gllm.runtime.sleep_mode import SleepAllocator

    return SleepAllocator.get()


def _used():
    free, total = torch.cuda.mem_get_info()
    return total - free


def test_normalize_tags():
    from gllm.runtime.sleep_mode import ALL_TAGS, normalize_tags

    assert normalize_tags(None) == tuple(sorted(ALL_TAGS))
    assert normalize_tags([]) == tuple(sorted(ALL_TAGS))
    assert normalize_tags(["weights", "weights"]) == ("weights",)
    with pytest.raises(ValueError):
        normalize_tags(["bogus"])


def test_sleep_wake_preserves_weights_and_addresses(allocator):
    with allocator.use_pool("weights"):
        w = torch.randn(64 << 20, device="cuda")  # 256 MiB
        small = torch.arange(1000, device="cuda", dtype=torch.float32)
    with allocator.use_pool("kv_cache"):
        kv = torch.full((128 << 20,), 7, dtype=torch.uint8, device="cuda")
    ref = w.clone()
    ptrs = (w.data_ptr(), small.data_ptr(), kv.data_ptr())

    # A CUDA graph captured against pool memory must survive a sleep cycle.
    out = torch.empty_like(w)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out.copy_(w * 2)
        with torch.cuda.graph(g, stream=s):
            out.copy_(w * 2)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    before = _used()
    freed = allocator.sleep(["weights", "kv_cache"])
    assert freed["weights"] >= w.numel() * 4
    assert freed["kv_cache"] >= kv.numel()
    assert allocator.is_sleeping("weights") and allocator.is_sleeping("kv_cache")
    assert before - _used() >= (w.numel() * 4 + kv.numel()) * 0.95

    allocator.wake_up(None)
    assert not allocator.is_sleeping()
    assert (w.data_ptr(), small.data_ptr(), kv.data_ptr()) == ptrs
    torch.testing.assert_close(w, ref)
    torch.testing.assert_close(
        small, torch.arange(1000, device="cuda", dtype=torch.float32)
    )
    # KV contents are discarded and zero-filled on wake.
    assert int(kv.max()) == 0

    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref * 2)


def test_partial_tags_and_repeat_cycles(allocator):
    with allocator.use_pool("weights"):
        w = torch.randn(1 << 20, device="cuda")
    ref = w.clone()
    for _ in range(3):
        allocator.sleep(["kv_cache"])
        assert allocator.sleeping_tags == ("kv_cache",)
        allocator.sleep(["weights"])
        allocator.wake_up(["weights"])
        torch.testing.assert_close(w, ref)
        assert allocator.sleeping_tags == ("kv_cache",)
        allocator.wake_up(["kv_cache"])
    assert allocator.sleeping_tags == ()


def test_close_and_trim_releases_free_segments():
    from gllm.runtime.sleep_mode import SleepAllocator

    allocator = SleepAllocator()  # private instance: closing its pool is final
    _KEEP_ALIVE.append(allocator)  # allocators live for the whole process
    with allocator.use_pool("weights"):
        keep = torch.randn(16 << 20, device="cuda")  # 64 MiB, survives load
        tmp = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")  # temporary
    ref = keep.clone()
    del tmp
    before = _used()
    released = allocator.close_and_trim("weights")
    assert released >= 1 << 30
    assert before - _used() >= (1 << 30) * 0.95
    assert allocator.tag_bytes("weights") < 1 << 30
    # A reopened tag gets a fresh pool; trimmed segments are never handed out.
    with allocator.use_pool("weights"):
        more = torch.full((1 << 20,), 3.0, device="cuda")

    # Trimmed segments are neither backed up nor remapped.
    freed = allocator.sleep(["weights"])
    assert freed["weights"] == allocator.tag_bytes("weights")
    allocator.wake_up(["weights"])
    torch.testing.assert_close(keep, ref)
    assert float(more.min()) == 3.0
