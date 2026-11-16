"""GPU memory release/restore for an idle engine ("sleep mode").

Allocations made inside :meth:`SleepAllocator.use_pool` are backed by CUDA
virtual memory management (VMM): every allocation reserves a virtual address
range and maps one physical handle into it. Sleeping unmaps and releases the
physical handle but keeps the virtual reservation, so every tensor, view and
captured CUDA graph that refers to the address remains valid; waking maps a
fresh physical handle at the same address. No graph has to be re-captured.

Two tags are managed:

* ``weights``: model parameters/buffers created during model loading. Their
  bytes are copied to a pinned host buffer before release and copied back on
  wake.
* ``kv_cache``: the cache arena backing. Its contents are discarded (the
  caller must flush any prefix-cache metadata first) and zero-filled on wake.

Only allocations made inside ``use_pool`` are affected. Everything else
(CUDA graph pools, sampler/runtime buffers, NCCL and custom all-reduce
buffers) stays resident.

The allocator is wired into PyTorch through ``CUDAPluggableAllocator``'s
underlying ``_cuda_customAllocator`` hook with ctypes callbacks, so no native
extension is needed; the callbacks only run when the caching allocator grows
or releases a pool segment, never per tensor.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from logger import logger

WEIGHTS_TAG = "weights"
KV_CACHE_TAG = "kv_cache"
ALL_TAGS: Tuple[str, ...] = (WEIGHTS_TAG, KV_CACHE_TAG)
# Tags whose contents are preserved across sleep (copied to host memory).
_BACKUP_TAGS = frozenset({WEIGHTS_TAG})


def normalize_tags(tags: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Validate ``tags``; ``None``/empty means every tag. Returns a sorted tuple."""
    tags = set(tags or ()) or set(ALL_TAGS)
    unknown = sorted(tags.difference(ALL_TAGS))
    if unknown:
        raise ValueError(
            f"unknown memory tags {unknown}; valid tags are {list(ALL_TAGS)}"
        )
    return tuple(sorted(tags))


def _check(result):
    """Unpack a ``cuda.bindings.driver`` result tuple, raising on error."""
    from cuda.bindings import driver

    err, *values = result
    if err != driver.CUresult.CUDA_SUCCESS:
        _, name = driver.cuGetErrorName(err)
        raise RuntimeError(f"CUDA driver error: {name.decode() if name else err}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


@dataclass
class _Allocation:
    size: int  # mapped (granularity-aligned) size
    tag: str
    handle: Optional[object]  # CUmemGenericAllocationHandle while mapped
    host_offset: int = -1  # offset into the host backup buffer (weights)
    # Released for good by ``trim``: the segment holds no live tensor and its
    # pool is closed, so it is never mapped again.
    trimmed: bool = False


_ALLOC_FN = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
)
_FREE_FN = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
)


class SleepAllocator:
    """Per-process VMM allocator behind the tagged sleep-mode memory pools."""

    _instance: Optional["SleepAllocator"] = None

    @classmethod
    def get(cls) -> "SleepAllocator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from cuda.bindings import driver

        self._driver = driver
        self.device = torch.cuda.current_device()
        # Make sure the primary context exists and is current on this thread.
        torch.cuda.init()
        torch.empty(1, device=f"cuda:{self.device}")

        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device
        self._prop = prop
        self._granularity = int(
            _check(
                driver.cuMemGetAllocationGranularity(
                    prop,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            )
        )
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._access = access

        self._allocations: Dict[int, _Allocation] = {}
        self._current_tag: Optional[str] = None
        self._sleeping: set = set()
        # Pinned host backup for the ``weights`` tag, kept across cycles so a
        # second sleep does not pay cudaHostAlloc again.
        self._host_ptr: Optional[int] = None
        self._host_size = 0

        # Keep the ctypes trampolines alive for the life of the process.
        self._alloc_cb = _ALLOC_FN(self._malloc)
        self._free_cb = _FREE_FN(self._free)
        alloc_ptr = ctypes.cast(self._alloc_cb, ctypes.c_void_p).value
        free_ptr = ctypes.cast(self._free_cb, ctypes.c_void_p).value
        self._torch_allocator = torch._C._cuda_customAllocator(alloc_ptr, free_ptr)
        self._pools: Dict[str, torch.cuda.MemPool] = {}
        # Pools closed by ``close_and_trim``. Kept alive: their remaining
        # segments still back live tensors.
        self._closed_pools = []

    # ------------------------------------------------------------------
    # PyTorch pluggable-allocator callbacks
    # ------------------------------------------------------------------
    def _map(self, ptr: int, size: int):
        driver = self._driver
        handle = _check(driver.cuMemCreate(size, self._prop, 0))
        try:
            _check(driver.cuMemMap(ptr, size, 0, handle, 0))
        except Exception:
            _check(driver.cuMemRelease(handle))
            raise
        _check(driver.cuMemSetAccess(ptr, size, [self._access], 1))
        return handle

    def _unmap(self, ptr: int, alloc: _Allocation) -> None:
        driver = self._driver
        _check(driver.cuMemUnmap(ptr, alloc.size))
        _check(driver.cuMemRelease(alloc.handle))
        alloc.handle = None

    def _malloc(self, size, device, stream):
        try:
            tag = self._current_tag
            if tag is None:
                raise RuntimeError("sleep-mode pool allocation outside use_pool()")
            if tag in self._sleeping:
                raise RuntimeError(f"allocation into sleeping pool {tag!r}")
            g = self._granularity
            aligned = (int(size) + g - 1) // g * g
            ptr = int(_check(self._driver.cuMemAddressReserve(aligned, g, 0, 0)))
            try:
                handle = self._map(ptr, aligned)
            except Exception:
                _check(self._driver.cuMemAddressFree(ptr, aligned))
                raise
            self._allocations[ptr] = _Allocation(aligned, tag, handle)
            return ptr
        except Exception:
            logger.exception("sleep-mode allocation of %d bytes failed", size)
            return None

    def _free(self, ptr, size, device, stream):
        allocations = self.__dict__.get("_allocations")
        if sys.is_finalizing() or allocations is None:
            # Process or allocator teardown: the driver releases everything
            # with the context, and our state may already be gone.
            return
        try:
            alloc = allocations.pop(int(ptr), None)
            if alloc is None:
                logger.error("sleep-mode free of unknown pointer 0x%x", ptr)
                return
            if alloc.handle is not None:
                self._unmap(int(ptr), alloc)
            _check(self._driver.cuMemAddressFree(int(ptr), alloc.size))
        except Exception:
            logger.exception("sleep-mode free of 0x%x failed", ptr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def use_pool(self, tag: str):
        """Route this thread's CUDA allocations to the VMM pool for ``tag``."""
        if tag not in ALL_TAGS:
            raise ValueError(tag)
        if self._current_tag is not None:
            raise RuntimeError("sleep-mode pools cannot be nested")
        pool = self._pools.get(tag)
        if pool is None:
            pool = torch.cuda.MemPool(self._torch_allocator)
            self._pools[tag] = pool
        self._current_tag = tag
        try:
            with torch.cuda.use_mem_pool(pool):
                yield
        finally:
            self._current_tag = None

    def is_sleeping(self, tag: Optional[str] = None) -> bool:
        return bool(self._sleeping) if tag is None else tag in self._sleeping

    @property
    def sleeping_tags(self) -> Tuple[str, ...]:
        return tuple(sorted(self._sleeping))

    def tag_bytes(self, tag: str) -> int:
        """Physical bytes ``tag`` holds while awake."""
        return sum(
            a.size for a in self._allocations.values() if a.tag == tag and not a.trimmed
        )

    def _live(self, tag: str):
        return [
            (p, a)
            for p, a in self._allocations.items()
            if a.tag == tag and not a.trimmed
        ]

    def close_and_trim(self, tag: str) -> int:
        """Close ``tag``'s pool and release its segments that hold no tensor.

        The caching allocator keeps a private pool's free segments cached even
        across ``empty_cache``, so load-time temporaries would otherwise stay
        resident (and be copied on every sleep). Once the pool is closed no
        allocation can be served from those segments again, so their physical
        memory can go. A later ``use_pool(tag)`` opens a fresh pool under the
        same tag. Returns the bytes released.
        """
        pool = self._pools.pop(tag, None)
        if pool is None:
            return 0
        self._closed_pools.append(pool)
        released = 0
        for seg in pool.snapshot():
            if any(b["state"] != "inactive" for b in seg["blocks"]):
                continue
            alloc = self._allocations.get(int(seg["address"]))
            if alloc is None or alloc.trimmed or alloc.handle is None:
                continue
            self._unmap(int(seg["address"]), alloc)
            alloc.trimmed = True
            released += alloc.size
        return released

    def _ensure_host_buffer(self, size: int) -> None:
        if size == 0 or (self._host_ptr is not None and self._host_size >= size):
            return
        if self._host_ptr is not None:
            _check(self._driver.cuMemFreeHost(self._host_ptr))
            self._host_ptr = None
            self._host_size = 0
        t0 = time.perf_counter()
        self._host_ptr = int(_check(self._driver.cuMemAllocHost(size)))
        self._host_size = size
        logger.info(
            "Sleep mode: pinned %.2f GiB host backup in %.2fs",
            size / (1 << 30),
            time.perf_counter() - t0,
        )

    def prepare_sleep(self, tags: Iterable[str]) -> None:
        """Allocate the host backup ``sleep(tags)`` needs, changing nothing else.

        The pinned buffer is the step most likely to fail (host memory), so
        callers run this before any destructive preparation of their own.
        """
        backup = [
            t
            for t in normalize_tags(tags)
            if t in _BACKUP_TAGS and t not in self._sleeping
        ]
        self._ensure_host_buffer(sum(a.size for t in backup for _, a in self._live(t)))

    def sleep(self, tags: Iterable[str]) -> Dict[str, int]:
        """Release the physical memory of ``tags``. Returns bytes freed per tag.

        The caller must guarantee no kernel is running or will run against
        these allocations until :meth:`wake_up`.
        """
        driver = self._driver
        tags = [t for t in normalize_tags(tags) if t not in self._sleeping]
        self.prepare_sleep(tags)
        torch.cuda.synchronize()
        freed: Dict[str, int] = {}
        for tag in tags:
            items = self._live(tag)
            if tag in _BACKUP_TAGS:
                host = self._host_ptr
                offset = 0
                for ptr, alloc in items:
                    _check(driver.cuMemcpyDtoH(host + offset, ptr, alloc.size))
                    alloc.host_offset = offset
                    offset += alloc.size
            for ptr, alloc in items:
                self._unmap(ptr, alloc)
            freed[tag] = sum(a.size for _, a in items)
            self._sleeping.add(tag)
        return freed

    def wake_up(self, tags: Iterable[str]) -> Dict[str, int]:
        """Re-map physical memory for ``tags`` and restore preserved contents."""
        driver = self._driver
        tags = [t for t in normalize_tags(tags) if t in self._sleeping]
        restored: Dict[str, int] = {}
        for tag in tags:
            items = self._live(tag)
            mapped = []
            try:
                for ptr, alloc in items:
                    alloc.handle = self._map(ptr, alloc.size)
                    mapped.append((ptr, alloc))
            except Exception:
                # Roll back so the tag is consistently asleep and can be retried.
                for ptr, alloc in mapped:
                    self._unmap(ptr, alloc)
                raise
            for ptr, alloc in items:
                if tag in _BACKUP_TAGS:
                    _check(
                        driver.cuMemcpyHtoD(
                            ptr, self._host_ptr + alloc.host_offset, alloc.size
                        )
                    )
                else:
                    # Discarded contents: start from a deterministic state.
                    _check(driver.cuMemsetD8(ptr, 0, alloc.size))
            restored[tag] = sum(a.size for _, a in items)
            self._sleeping.discard(tag)
        torch.cuda.synchronize()
        return restored


def sleep_mode_enabled() -> bool:
    """Sleep mode is on in this process once its allocator exists.

    The model runner creates it (``SleepAllocator.get()``) before loading.
    """
    return SleepAllocator._instance is not None


def memory_pool(tag: str):
    """``use_pool(tag)`` when sleep mode is enabled, else a no-op context."""
    if not sleep_mode_enabled():
        return contextlib.nullcontext()
    return SleepAllocator.get().use_pool(tag)
