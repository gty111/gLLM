"""Rust ownership/age metadata with Python prefix-policy callbacks.

Snapshot properties are diagnostics only; allocation and release never build
Python owner maps or per-slot metadata mirrors.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from gllm._cache_arena import ArenaCore


@dataclass
class NativeCacheType:
    allocator: object
    index: int
    name: str
    entry_bytes: int
    pages_per_slot: int
    num_slots: int
    prefer_high: bool = False
    evictor: Optional[Callable] = None
    reclaimer: Optional[Callable] = None
    reclaimable_count: Optional[Callable] = None

    @property
    def free_slots(self):
        return set(self.allocator._core.slots(self.index, "free"))

    @property
    def live_slots(self):
        return set(self.allocator._core.slots(self.index, "live"))

    @property
    def retained_slots(self):
        return set(self.allocator._core.slots(self.index, "retained"))

    @property
    def occupied_pages(self):
        return self.allocator._core.occupancy(self.index)

    @property
    def cached_priority(self):
        return self.allocator._core.priorities(self.index)

    @property
    def free_heap(self):
        return self.allocator._core.heap(self.index)


class NativeCacheArenaAllocator:
    backend = "rust"

    def __init__(self, num_physical_pages):
        if num_physical_pages <= 0:
            raise ValueError(num_physical_pages)
        self.num_physical_pages = int(num_physical_pages)
        self._core = ArenaCore(self.num_physical_pages)
        self._types = {}
        self._groups = []

    def register_type(self, name, entry_bytes, physical_page_bytes, *, prefer_high=False):
        if name in self._types:
            raise ValueError(f"arena cache type {name!r} already exists")
        if entry_bytes <= 0 or physical_page_bytes <= 0:
            raise ValueError((entry_bytes, physical_page_bytes))
        span = (int(entry_bytes) + physical_page_bytes - 1) // physical_page_bytes
        index = self._core.register(span, prefer_high)
        group = NativeCacheType(self, index, name, int(entry_bytes), span,
                                self.num_physical_pages // span, prefer_high)
        self._types[name] = group
        self._groups.append(group)
        return group

    def cache_type(self, name):
        try:
            return self._types[name]
        except KeyError as exc:
            raise KeyError(f"unknown arena cache type {name!r}") from exc

    @property
    def cache_type_names(self):
        return tuple(self._types)

    def set_evictor(self, name, callback):
        self.cache_type(name).evictor = callback

    def set_reclaimer(self, name, callback, reclaimable_count=None):
        group = self.cache_type(name)
        group.reclaimer = callback
        group.reclaimable_count = reclaimable_count

    @staticmethod
    def _extent(group, slot):
        if not 0 <= slot < group.num_slots:
            raise IndexError(f"{group.name} slot {slot} outside [0, {group.num_slots})")
        return slot * group.pages_per_slot, (slot + 1) * group.pages_per_slot

    @staticmethod
    def _slots(slots):
        return slots if isinstance(slots, (list, tuple)) else list(slots)

    def _call(self, method, *args):
        try:
            return method(*args)
        except OverflowError as exc:
            raise IndexError("arena slot out of range") from exc

    def _update_candidates(self, start, end, delta):
        """Compatibility observation hook; Rust has already updated the grids."""

    def _update_cached_priorities(self, start, end):
        """Compatibility observation hook; priorities are updated in Rust."""

    def _emit_changes(self, delta):
        # Preserve diagnostic hooks without crossing Python once per page.
        if "_update_candidates" in self.__dict__:
            for start, end in self._core.changed_ranges():
                self._update_candidates(start, end, delta)

    def _notify(self, group, slots, preserve=False):
        callbacks = [g.evictor is not None for g in self._groups]
        if not slots or (not any(callbacks) and self._core.status()[1] == 0):
            return
        notifications = self._core.invalidate(group.index, slots, callbacks, preserve)
        for index, slot in notifications:
            callback = self._groups[index].evictor
            if callback is not None:
                callback(slot)

    def _reclaim_for(self, group, count):
        while self.num_free_slots(group.name) < count:
            progressed = False
            for donor in self._groups:
                if donor.name == group.name or donor.reclaimer is None:
                    continue
                before = self.num_used_physical_pages
                if donor.reclaimer():
                    progressed = True
                    if self.num_used_physical_pages >= before:
                        raise RuntimeError(f"arena reclaimer {donor.name!r} freed no physical pages")
                    if self.num_free_slots(group.name) >= count:
                        return True
            if not progressed:
                return False
        return True

    def allocate(self, name, count=1, *, slot=None, retain=False):
        group = self.cache_type(name)
        count = int(count)
        if count < 0 or (slot is not None and count != 1) or (retain and slot is None):
            raise ValueError((count, slot, retain))
        if slot is None and self.num_free_slots(name) < count:
            if not self._reclaim_for(group, count):
                return None
        result = self._call(self._core.allocate, group.index, count, slot, retain)
        if result:
            self._emit_changes(1)
            fresh = self._core.claimed_slots() if retain else result
            self._notify(group, fresh, preserve=retain)
        return result

    def retain_many(self, name, slots):
        group = self.cache_type(name)
        result = self._call(self._core.retain_many, group.index, self._slots(slots))
        if result:
            self._emit_changes(1)
            self._notify(group, self._core.claimed_slots(), preserve=True)
        return result

    def free(self, name, slots):
        group = self.cache_type(name)
        self._call(self._core.free, group.index, self._slots(slots))
        self._emit_changes(-1)

    def mark_cached(self, name, slots):
        self._call(self._core.mark_cached, self.cache_type(name).index, self._slots(slots))

    def discard_cached(self, name, slots):
        self._call(self._core.discard_cached, self.cache_type(name).index, self._slots(slots))

    def is_free(self, name, slot):
        return self._call(self._core.is_free, self.cache_type(name).index, int(slot))

    def num_free_slots(self, name):
        return self._core.counts(self.cache_type(name).index)[0]

    def _reclaimable_pages(self, group):
        if group.reclaimer is None:
            return 0
        count = (group.reclaimable_count() if group.reclaimable_count is not None
                 else self._core.counts(group.index)[1])
        return count * group.pages_per_slot

    def num_available_slots(self, name):
        group = self.cache_type(name)
        reclaimable = sum(self._reclaimable_pages(g) for g in self._groups if g is not group)
        return min(group.num_slots, self.num_free_slots(name) + reclaimable // group.pages_per_slot)

    @property
    def num_used_physical_pages(self):
        return self._core.status()[0]

    @property
    def num_non_reclaimable_physical_pages(self):
        return self.num_used_physical_pages - sum(self._reclaimable_pages(g) for g in self._groups)

    @property
    def _owners(self):
        owners = [None] * self.num_physical_pages
        for index, slot, start, end in self._core.ranges():
            owners[start:end] = [(self._groups[index].name, slot)] * (end - start)
        return owners

    @property
    def _cached_pages(self):
        return self._core.ages()

    @property
    def _cache_clock(self):
        return self._core.status()[1]
