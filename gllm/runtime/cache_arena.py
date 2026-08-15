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

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch

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


@dataclass
class ArenaCacheType:
    """One logical cache layout registered in a :class:`CacheArena`."""

    name: str
    entry_bytes: int
    pages_per_slot: int
    num_slots: int
    prefer_high: bool = False
    free_slots: Set[int] = field(default_factory=set)
    free_heap: List[int] = field(default_factory=list)
    live_slots: Set[int] = field(default_factory=set)
    # Number of physical pages in each logical slot currently owned by any
    # cache type. Maintaining this incrementally makes the allocator's hot
    # ``is extent free?`` query O(1), independent of entry size.
    occupied_pages: List[int] = field(default_factory=list)
    evictor: Optional[Callable[[int], None]] = None
    reclaimer: Optional[Callable[[], bool]] = None

    def heap_key(self, slot: int) -> int:
        return -slot if self.prefer_high else slot

    def add_free(self, slot: int) -> None:
        if slot in self.free_slots:
            return
        self.free_slots.add(slot)
        heapq.heappush(self.free_heap, self.heap_key(slot))

    def discard_free(self, slot: int) -> None:
        self.free_slots.discard(slot)

    def take_free(self) -> Optional[int]:
        while self.free_heap:
            key = heapq.heappop(self.free_heap)
            slot = -key if self.prefer_high else key
            if slot in self.free_slots:
                self.free_slots.remove(slot)
                return slot
        return None


class CacheArenaAllocator:
    """Generic aligned-extent allocator over physical arena pages.

    Allocation is expressed in logical slot ids, not byte pointers.  A slot's
    physical extent is ``[slot * pages_per_slot, (slot + 1) * pages_per_slot)``.
    Different cache types therefore expose independent logical id spaces while
    sharing one physical ownership map.
    """

    def __init__(self, num_physical_pages: int):
        if num_physical_pages <= 0:
            raise ValueError(num_physical_pages)
        self.num_physical_pages = int(num_physical_pages)
        self._owners: List[Optional[Tuple[str, int]]] = [None] * self.num_physical_pages
        self._types: Dict[str, ArenaCacheType] = {}
        self._used_physical_pages = 0

    def register_type(
        self,
        name: str,
        entry_bytes: int,
        physical_page_bytes: int,
        *,
        prefer_high: bool = False,
    ) -> ArenaCacheType:
        if name in self._types:
            raise ValueError(f"arena cache type {name!r} already exists")
        if entry_bytes <= 0:
            raise ValueError(entry_bytes)
        pages = (int(entry_bytes) + physical_page_bytes - 1) // physical_page_bytes
        cache_type = ArenaCacheType(
            name=name,
            entry_bytes=int(entry_bytes),
            pages_per_slot=pages,
            num_slots=self.num_physical_pages // pages,
            prefer_high=prefer_high,
            occupied_pages=[0] * (self.num_physical_pages // pages),
        )
        self._types[name] = cache_type
        # Types are normally registered before the first claim. Populate from
        # the owner map as a correctness-preserving fallback for dynamically
        # registered cache shapes.
        for slot in range(cache_type.num_slots):
            start, end = self._extent(cache_type, slot)
            occupied = sum(owner is not None for owner in self._owners[start:end])
            cache_type.occupied_pages[slot] = occupied
            if occupied == 0:
                cache_type.add_free(slot)
        return cache_type

    def cache_type(self, name: str) -> ArenaCacheType:
        try:
            return self._types[name]
        except KeyError as exc:
            raise KeyError(f"unknown arena cache type {name!r}") from exc

    @property
    def cache_type_names(self) -> Tuple[str, ...]:
        return tuple(self._types)

    def set_evictor(self, name: str, callback: Callable[[int], None]) -> None:
        self.cache_type(name).evictor = callback

    def set_reclaimer(self, name: str, callback: Callable[[], bool]) -> None:
        """Register pressure-driven reclamation for an evictable cache type."""
        self.cache_type(name).reclaimer = callback

    def _reclaim_for(self, request: ArenaCacheType, count: int) -> bool:
        while len(request.free_slots) < count:
            progressed = False
            for donor in self._types.values():
                # Replacing one entry with another entry of the same cache type
                # only churns metadata and cannot increase its free capacity.
                if donor.name == request.name or donor.reclaimer is None:
                    continue
                before = self._used_physical_pages
                if donor.reclaimer():
                    progressed = True
                    if self._used_physical_pages >= before:
                        raise RuntimeError(
                            f"arena reclaimer {donor.name!r} freed no physical pages"
                        )
                    if len(request.free_slots) >= count:
                        return True
            if not progressed:
                return False
        return True

    @staticmethod
    def _extent(cache_type: ArenaCacheType, slot: int) -> Tuple[int, int]:
        if not 0 <= slot < cache_type.num_slots:
            raise IndexError(
                f"{cache_type.name} slot {slot} outside [0, {cache_type.num_slots})"
            )
        start = slot * cache_type.pages_per_slot
        return start, start + cache_type.pages_per_slot

    def _is_extent_free(self, cache_type: ArenaCacheType, slot: int) -> bool:
        if not 0 <= slot < cache_type.num_slots:
            self._extent(cache_type, slot)  # raise the canonical IndexError
        return cache_type.occupied_pages[slot] == 0

    def _overlapping_slots(
        self, cache_type: ArenaCacheType, start: int, end: int
    ) -> range:
        span = cache_type.pages_per_slot
        first = start // span
        last = min(cache_type.num_slots - 1, (end - 1) // span)
        return range(first, last + 1)

    def _update_candidates(self, start: int, end: int, delta: int) -> None:
        """Apply one physical ownership transition to every typed slot grid."""
        if delta not in (-1, 1):
            raise ValueError(delta)
        for cache_type in self._types.values():
            for slot in self._overlapping_slots(cache_type, start, end):
                slot_start, slot_end = self._extent(cache_type, slot)
                overlap = min(end, slot_end) - max(start, slot_start)
                occupied = cache_type.occupied_pages[slot] + delta * overlap
                if not 0 <= occupied <= cache_type.pages_per_slot:
                    raise RuntimeError(
                        f"arena occupancy index is inconsistent for "
                        f"{cache_type.name} slot {slot}: {occupied}"
                    )
                cache_type.occupied_pages[slot] = occupied
                if occupied == 0:
                    cache_type.add_free(slot)
                else:
                    cache_type.discard_free(slot)

    def _claim(self, cache_type: ArenaCacheType, slot: int) -> None:
        self._claim_many(cache_type, [slot])

    def _claim_many(self, cache_type: ArenaCacheType, slots: Iterable[int]) -> None:
        """Claim a cohort and coalesce adjacent candidate-index updates."""
        slot_list = [int(slot) for slot in slots]
        if len(set(slot_list)) != len(slot_list):
            raise RuntimeError(
                f"duplicate {cache_type.name} slots in one claim: {slot_list}"
            )
        extents = []
        for slot in slot_list:
            start, end = self._extent(cache_type, slot)
            if not self._is_extent_free(cache_type, slot):
                raise RuntimeError(
                    f"{cache_type.name} slot {slot} is not physically free"
                )
            extents.append((start, end, slot))

        for start, end, slot in extents:
            owner = (cache_type.name, slot)
            self._owners[start:end] = [owner] * (end - start)
            cache_type.live_slots.add(slot)
            self._used_physical_pages += end - start
        merged = []
        for start, end, _ in sorted(extents):
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        for start, end in merged:
            self._update_candidates(start, end, 1)

    def _evict_stale_views(
        self,
        allocations: Iterable[Tuple[str, int]],
        *,
        preserved: Optional[Set[Tuple[str, int]]] = None,
    ) -> None:
        # Claims are already visible, so callbacks that free some other cache
        # entry cannot accidentally reselect the just-claimed extent.
        notified: Set[Tuple[str, int]] = set()
        for name, slot in allocations:
            source = self.cache_type(name)
            start, end = self._extent(source, slot)
            for cache_type in self._types.values():
                if cache_type.evictor is None:
                    continue
                for stale_slot in self._overlapping_slots(cache_type, start, end):
                    key = (cache_type.name, stale_slot)
                    if key in notified or (preserved is not None and key in preserved):
                        continue
                    notified.add(key)
                    cache_type.evictor(stale_slot)

    def allocate(
        self,
        name: str,
        count: int = 1,
        *,
        slot: Optional[int] = None,
        retain: bool = False,
    ) -> Optional[List[int]]:
        """Atomically claim logical slots.

        ``slot`` is used for prefix-cache retention. If that exact slot is
        already owned by the same cache type, physical ownership is unchanged;
        the caller maintains its own logical reference count.
        """
        cache_type = self.cache_type(name)
        count = int(count)
        if count < 0 or (slot is not None and count != 1) or (retain and slot is None):
            raise ValueError((count, slot, retain))
        if count == 0:
            return []

        if slot is not None:
            start, end = self._extent(cache_type, int(slot))
            owners = set(self._owners[start:end])
            if owners == {(name, int(slot))}:
                if retain:
                    return [int(slot)]
                raise RuntimeError(f"{name} slot {slot} is already live")
            if owners != {None}:
                return None
            cache_type.discard_free(int(slot))
            selected = [int(slot)]
            retain_existing_contents = retain
        else:
            if len(cache_type.free_slots) < count and not self._reclaim_for(
                cache_type, count
            ):
                return None
            selected = []
            for _ in range(count):
                candidate = cache_type.take_free()
                if candidate is None:
                    raise RuntimeError("arena free-slot index is inconsistent")
                selected.append(candidate)
            retain_existing_contents = False

        self._claim_many(cache_type, selected)
        preserved = {(name, selected[0])} if retain_existing_contents else None
        self._evict_stale_views(
            ((name, selected_slot) for selected_slot in selected),
            preserved=preserved,
        )
        return selected

    def free(self, name: str, slots: Iterable[int]) -> None:
        cache_type = self.cache_type(name)
        slot_list = [int(value) for value in slots]
        if len(set(slot_list)) != len(slot_list):
            raise RuntimeError(f"duplicate {name} slots in one free: {slot_list}")
        extents = []
        for slot in slot_list:
            start, end = self._extent(cache_type, slot)
            expected = (name, slot)
            if any(owner != expected for owner in self._owners[start:end]):
                raise RuntimeError(f"{name} slot {slot} is not live")
            extents.append((start, end, slot))

        # Release metadata atomically, then coalesce adjacent physical ranges.
        # MTP block tables are allocated as neighboring arena entries; updating
        # every overlapping cache grid once for the combined range avoids
        # repeating the generic candidate walk for each checkpoint column.
        for start, end, slot in extents:
            self._owners[start:end] = [None] * (end - start)
            cache_type.live_slots.remove(slot)
            self._used_physical_pages -= end - start
        merged = []
        for start, end, _ in sorted(extents):
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        for start, end in merged:
            self._update_candidates(start, end, -1)

    def is_free(self, name: str, slot: int) -> bool:
        cache_type = self.cache_type(name)
        return self._is_extent_free(cache_type, int(slot))

    def num_free_slots(self, name: str) -> int:
        return len(self.cache_type(name).free_slots)

    def num_available_slots(self, name: str) -> int:
        """Free slots plus pressure-reclaimable capacity (admission estimate)."""
        request = self.cache_type(name)
        reclaimable_pages = sum(
            len(cache_type.live_slots) * cache_type.pages_per_slot
            for cache_type in self._types.values()
            if cache_type.name != name and cache_type.reclaimer is not None
        )
        return min(
            request.num_slots,
            len(request.free_slots) + reclaimable_pages // request.pages_per_slot,
        )

    @property
    def num_used_physical_pages(self) -> int:
        return self._used_physical_pages


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
