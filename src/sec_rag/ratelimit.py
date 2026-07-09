"""Per-caller rate limiting: in-process sliding window by default, Redis
fixed-window when SECRAG_REDIS_URL is set (so the limit holds across
replicas instead of multiplying by replica count).

Interface: check(key) -> None if allowed, else seconds until retry.
"""

import threading
import time
from collections import defaultdict, deque
from typing import Protocol


class RateLimiter(Protocol):
    def check(self, key: str) -> float | None: ...


class SlidingWindowLimiter:
    """In-process sliding window (exact within one process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            while q and q[0] <= now - 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(0.0, 60 - (now - q[0]))
            q.append(now)
            return None


class RedisFixedWindowLimiter:
    """Redis INCR-per-minute-window. Slightly bursty at window boundaries
    (up to 2x limit across a boundary) but atomic, O(1), and shared across
    replicas — the standard tradeoff for a first distributed limiter."""

    def __init__(self, url: str, per_minute: int):
        import redis
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self.per_minute = per_minute

    def check(self, key: str) -> float | None:
        window = int(time.time() // 60)
        rkey = f"secrag:rl:{key}:{window}"
        pipe = self._r.pipeline()
        pipe.incr(rkey)
        pipe.expire(rkey, 120)
        count, _ = pipe.execute()
        if count > self.per_minute:
            return max(1.0, 60 - (time.time() % 60))
        return None


def make_limiter(per_minute: int, redis_url: str | None = None) -> RateLimiter:
    if redis_url:
        return RedisFixedWindowLimiter(redis_url, per_minute)
    return SlidingWindowLimiter(per_minute)
