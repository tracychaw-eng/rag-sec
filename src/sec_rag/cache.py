"""Caches for query embeddings and answers: in-process TTL+LRU by default,
Redis when SECRAG_REDIS_URL is set (required once the service runs more
than one replica).

Cached values must be JSON-serializable (lists, dicts, strings) so the two
backends are interchangeable. Keys always include the model/corpus/prompt
version that produced the value, so a model swap or re-ingest can never
serve stale entries.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Protocol


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any) -> None: ...
    def stats(self) -> dict: ...


class TTLCache:
    """Thread-safe in-process LRU cache with per-entry TTL."""

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
            "backend": "memory",
            "items": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }


class RedisCache:
    """Redis-backed cache. Values stored as JSON with per-key TTL.

    Uses the sync redis client: individual GET/SETEX round-trips are
    sub-millisecond on a local/near network and not worth the complexity
    of a second (async) client in the same codebase yet.
    """

    def __init__(self, url: str, namespace: str, ttl_s: float = 3600.0):
        import redis
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self.ns = namespace
        self.ttl_s = int(ttl_s)
        self.hits = 0
        self.misses = 0

    def _k(self, key: str) -> str:
        return f"secrag:{self.ns}:{key}"

    def get(self, key: str) -> Optional[Any]:
        raw = self._r.get(self._k(key))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any) -> None:
        self._r.setex(self._k(key), self.ttl_s, json.dumps(value))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "backend": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }


def make_cache(namespace: str, max_items: int, ttl_s: float,
               redis_url: str | None = None) -> CacheBackend:
    if redis_url:
        return RedisCache(redis_url, namespace, ttl_s)
    return TTLCache(max_items=max_items, ttl_s=ttl_s)


def cache_key(*parts: str) -> str:
    """Stable key from any number of namespacing parts."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
