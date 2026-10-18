"""Extensible GPU cache arena with typed, strided tensor views.

The arena owns one byte tensor and divides it into fixed-size physical pages.
Cache consumers register a *type* describing the number of physical pages in
one logical slot.  Types can have different slot sizes and can reuse the same
bytes at different times; KV and recurrent state are merely the first two
users of this mechanism.

Every type uses an aligned fixed-stride slot grid.  That gives kernels a stable
base pointer and stride (required by CUDA Graph) while the allocator decides
which slots are physically live.  An eviction callback lets a cache type drop
unpinned metadata when another type claims overlapping bytes.
"""

from dataclasses import dataclass
from math import gcd, lcm
from typing import Dict, Iterable, Optional, Tuple

import torch

from gllm.runtime.native_cache_arena import (
    NativeCacheArenaAllocator as CacheArenaAllocator,
    NativeCacheType as ArenaCacheType,
)
from gllm.utils import get_dtype_bytes


def _numel(shape) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CacheTensorLayout:
    """One tensor bank stored inside every logical cache entry."""

    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if not self.name or any(dim <= 0 for dim in self.shape):
            raise ValueError((self.name, self.shape))

    @property
    def nbytes(self) -> int:
        return _numel(self.shape) * get_dtype_bytes(self.dtype)


@dataclass(frozen=True)
class CacheLayout:
    """Complete layout of one registered logical cache slot.

    A cache can contain several banks that share one lifetime and page id. For
    example, explicit attention registers K and V banks, while DSA registers
    its MLA latent plus BF16 and FP8 index banks as one cache entry.
    """

    name: str
    tensors: Tuple[CacheTensorLayout, ...]
    prefer_high: bool = False

    def __post_init__(self):
        object.__setattr__(self, "tensors", tuple(self.tensors))
        names = [tensor.name for tensor in self.tensors]
        if not self.name or not names or len(names) != len(set(names)):
            raise ValueError((self.name, names))

    def packed_offsets(self) -> Dict[str, int]:
        offsets: Dict[str, int] = {}
        offset = 0
        for tensor in self.tensors:
            offset = _align_up(offset, get_dtype_bytes(tensor.dtype))
            offsets[tensor.name] = offset
            offset += tensor.nbytes
        return offsets

    @property
    def entry_bytes(self) -> int:
        offsets = self.packed_offsets()
        # Keep every physical entry suitable for a later registered cache with
        # wider scalar types. This also makes entry strides friendly to vector
        # loads without imposing alignment rules on individual consumers.
        alignment = max(16, *(get_dtype_bytes(tensor.dtype) for tensor in self.tensors))
        last = self.tensors[-1]
        return _align_up(offsets[last.name] + last.nbytes, alignment)


def select_physical_page_bytes(
    layouts: Iterable[CacheLayout], *, min_page_bytes: int = 1024
) -> int:
    """Choose one fixed accounting grain before allocating the arena.

    Use the GCD of logical entry sizes when it meets the metadata-budget
    floor. Below the floor, registration rounds each entry up to whole
    physical pages; the logical tensor shape and packed bank offsets remain
    unchanged. Include scalar alignment so every slot has valid typed views.
    """
    layouts = tuple(layouts)
    if not layouts or min_page_bytes <= 0:
        raise ValueError("cache layouts and a positive minimum page size are required")
    alignment = lcm(16, *(
        get_dtype_bytes(tensor.dtype)
        for layout in layouts for tensor in layout.tensors
    ))
    common = gcd(*(layout.entry_bytes for layout in layouts))
    floor = _align_up(min_page_bytes, alignment)
    return common if common >= floor else floor


class RegisteredCache:
    """A registered cache layout and its stable tensor views."""

    def __init__(
        self,
        arena: "CacheArena",
        layout: CacheLayout,
        cache_type: ArenaCacheType,
    ):
        self.arena = arena
        self.layout = layout
        self.cache_type = cache_type
        offsets = layout.packed_offsets()
        self._tensors = {
            tensor.name: arena.entry_view(
                layout.name,
                tensor.dtype,
                tensor.shape,
                entry_offset_bytes=offsets[tensor.name],
            )
            for tensor in layout.tensors
        }

    @property
    def name(self) -> str:
        return self.layout.name

    @property
    def num_slots(self) -> int:
        return self.cache_type.num_slots

    @property
    def entry_bytes(self) -> int:
        return self.layout.entry_bytes

    @property
    def slot_bytes(self) -> int:
        """Physical slot stride, including any page-grain padding."""
        return self.cache_type.pages_per_slot * self.arena.physical_page_bytes

    def tensor(self, name: str) -> torch.Tensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"cache {self.name!r} has no tensor {name!r}") from exc

    def slot_allocator(self) -> "ArenaSlotAllocator":
        return ArenaSlotAllocator(self.arena, self.name)


class CacheArena:
    """One GPU storage allocation plus its generic physical-page allocator."""

    def __init__(self, backing: torch.Tensor, physical_page_bytes: int):
        if backing.dtype != torch.uint8 or not backing.is_contiguous():
            raise ValueError("cache arena backing must be contiguous uint8")
        if physical_page_bytes <= 0 or backing.numel() % physical_page_bytes:
            raise ValueError((backing.numel(), physical_page_bytes))
        self.backing = backing
        self.physical_page_bytes = int(physical_page_bytes)
        self.allocator = CacheArenaAllocator(
            backing.numel() // self.physical_page_bytes
        )
        self._registered_caches: Dict[str, RegisteredCache] = {}

    def register_cache(self, layout: CacheLayout) -> RegisteredCache:
        """Register a multi-bank cache and materialize all of its views."""
        if layout.name in self._registered_caches:
            raise ValueError(f"arena cache {layout.name!r} already exists")
        cache_type = self.register_cache_type(
            layout.name,
            layout.entry_bytes,
            prefer_high=layout.prefer_high,
        )
        cache = RegisteredCache(self, layout, cache_type)
        self._registered_caches[layout.name] = cache
        return cache

    def cache(self, name: str) -> RegisteredCache:
        try:
            return self._registered_caches[name]
        except KeyError as exc:
            raise KeyError(f"unknown registered cache {name!r}") from exc

    def register_cache_type(
        self,
        name: str,
        entry_bytes: int,
        *,
        prefer_high: bool = False,
    ) -> ArenaCacheType:
        return self.allocator.register_type(
            name,
            entry_bytes,
            self.physical_page_bytes,
            prefer_high=prefer_high,
        )

    def entry_view(
        self,
        name: str,
        dtype: torch.dtype,
        trailing_shape,
        *,
        entry_offset_bytes: int = 0,
    ) -> torch.Tensor:
        """Return ``[num_slots, *trailing_shape]`` over a cache-type grid."""
        cache_type = self.allocator.cache_type(name)
        item_size = get_dtype_bytes(dtype)
        if entry_offset_bytes % item_size:
            raise ValueError((entry_offset_bytes, dtype))
        numel = 1
        for dim in trailing_shape:
            numel *= int(dim)
        if entry_offset_bytes + numel * item_size > cache_type.entry_bytes:
            raise ValueError(
                f"view exceeds {name} entry: offset={entry_offset_bytes}, "
                f"bytes={numel * item_size}, entry={cache_type.entry_bytes}"
            )
        inner_stride = []
        running = 1
        for dim in reversed(tuple(trailing_shape)):
            inner_stride.append(running)
            running *= int(dim)
        return self.strided_view(
            dtype,
            (cache_type.num_slots, *tuple(trailing_shape)),
            (
                cache_type.pages_per_slot * self.physical_page_bytes // item_size,
                *reversed(inner_stride),
            ),
            storage_offset_bytes=entry_offset_bytes,
        )

    def strided_view(
        self,
        dtype: torch.dtype,
        shape,
        stride,
        *,
        storage_offset_bytes: int = 0,
    ) -> torch.Tensor:
        """Build an arbitrary stable tensor layout over the arena storage.

        ``entry_view`` covers the common slot-major layout. This lower-level
        primitive supports cache shapes with additional leading axes (for
        example layer-major recurrent state) without teaching the allocator
        about those axes.
        """
        item_size = get_dtype_bytes(dtype)
        if storage_offset_bytes % item_size:
            raise ValueError((storage_offset_bytes, dtype))
        return torch.as_strided(
            self.backing.view(dtype),
            size=tuple(shape),
            stride=tuple(stride),
            storage_offset=storage_offset_bytes // item_size,
        )


class ArenaSlotAllocator:
    """``IDAllocator``-compatible facade for one registered cache type."""

    def __init__(self, arena: CacheArena, cache_type: str):
        self.arena = arena
        self.cache_type = cache_type
        self.size = arena.allocator.cache_type(cache_type).num_slots

    def allocate(self, id: Optional[int] = None):
        slots = self.arena.allocator.allocate(
            self.cache_type, slot=id, retain=id is not None
        )
        if slots is None:
            raise RuntimeError(f"no free {self.cache_type} arena slot")
        return slots[0]

    def free(self, id: int):
        self.arena.allocator.free(self.cache_type, [id])

    def retain_many(self, ids: Iterable[int]) -> None:
        if self.arena.allocator.retain_many(self.cache_type, ids) is None:
            raise RuntimeError(f"cannot retain occupied {self.cache_type} arena slots")

    def free_many(self, ids: Iterable[int]) -> None:
        """Return a cohort in one allocator transaction.

        Sequence teardown commonly releases a contiguous KV page table.  The
        arena allocator can coalesce those physical extents before updating
        every registered cache-type grid, which is lost when callers invoke
        ``free`` once per page.
        """
        self.arena.allocator.free(self.cache_type, ids)

    def is_free(self, id: int) -> bool:
        return self.arena.allocator.is_free(self.cache_type, id)

    def get_num_used_ids(self):
        return self.arena.allocator.num_used_physical_pages

    def get_num_free_ids(self):
        return self.arena.allocator.num_available_slots(self.cache_type)
