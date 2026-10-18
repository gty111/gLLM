"""Bounded, in-memory storage for Responses API results."""

import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class ResponseStore:
    """Keep recent responses for ``previous_response_id`` lookups."""

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 86400):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = Lock()

    def put(self, response_id: str, value: dict[str, Any]) -> None:
        """Store a response record, evicting expired and least recently used entries."""
        with self._lock:
            now = time.monotonic()
            for key, (created, _) in list(self._entries.items()):
                if now - created >= self.ttl_seconds:
                    del self._entries[key]
            self._entries[response_id] = (now, value)
            self._entries.move_to_end(response_id)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get(self, response_id: str) -> dict[str, Any] | None:
        """Return a live response record and refresh its LRU position."""
        with self._lock:
            entry = self._entries.get(response_id)
            if entry is None:
                return None
            created, value = entry
            if time.monotonic() - created >= self.ttl_seconds:
                del self._entries[response_id]
                return None
            self._entries.move_to_end(response_id)
            return value

    def delete(self, response_id: str) -> None:
        """Remove a response if present."""
        with self._lock:
            self._entries.pop(response_id, None)
