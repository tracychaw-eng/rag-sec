"""In-process TTL+LRU caches for query embeddings, plans, and answers.

Deliberately dependency-free: at single-instance scale an in-process cache
is correct and free. The `CacheBackend` protocol is the seam where Redis
slots in when the service scales horizontally (Phase 2+): implement the
same three methods against a Redis client and swap in config.

Keys always include the model/corpus/prompt version that produced the
value, so a model swap or re-ingest can never serve stale entries.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any) -> None: ...
    def stats(self) -> dict: ...


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(self, max_items: int = 2048, ttl_s: float = 3600.0):
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires, value = entry
            if time.monotonic() > expires:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "items": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }


def cache_key(*parts: str) -> str:
    """Stable key from any number of namespacing parts."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
