from collections import OrderedDict


class IDAllocator:
    """FIFO id pool with O(1) random removal.

    Internally backed by an ``OrderedDict`` so all three operations the
    prefix cache hits the most --- ``popleft``-style allocation,
    ``allocate(id)`` for cache hits, and ``free`` --- are O(1). The
    previous ``deque``-based implementation paid O(n) per cache hit
    because ``deque.remove`` is linear, which dominated CPU time during
    long-context prefill (see profile in benchmarks/results: with
    ~70k free pages and a 48% cache hit rate the deque path can spend
    seconds per request on remove() alone).

    ``free`` re-inserts at the back to keep the FIFO order so recently
    freed pages stay warm for the prefix cache, mirroring the original
    ``append`` behavior.
    """

    def __init__(self, start_num=1, end_num=1000):
        self.size = end_num - start_num + 1
        # ``OrderedDict.fromkeys`` preserves insertion order, giving us the
        # same FIFO popleft semantics as the previous deque.
        self.free_ids: "OrderedDict[int, None]" = OrderedDict.fromkeys(
            range(start_num, end_num + 1)
        )

    def allocate(self, id: int = None):
        if id is None:
            assert len(self.free_ids) != 0
            id, _ = self.free_ids.popitem(last=False)
        else:
            # ``pop(id, None)`` matches the prior "noop if not free" semantics
            # without paying a separate __contains__ lookup.
            self.free_ids.pop(id, None)
        return id

    def free(self, id: int):
        # Insert at the back (FIFO) to keep already-warm pages cached
        # longer; mirrors the original ``deque.append`` behavior.
        self.free_ids[id] = None

    def get_num_used_ids(self):
        return self.size - len(self.free_ids)

    def get_num_free_ids(self):
        return len(self.free_ids)
