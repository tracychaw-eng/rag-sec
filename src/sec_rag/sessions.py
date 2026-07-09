"""Chat session history: in-process by default, Redis when configured.

History is a list of {"role", "content"} dicts, trimmed to the last
MAX_TURNS exchanges. The planner reads it to condense follow-ups into
standalone queries.
"""

import json
import threading
from collections import defaultdict
from typing import Protocol

MAX_TURNS = 10  # user+assistant pairs kept per session


class SessionStore(Protocol):
    def get(self, session_id: str) -> list[dict]: ...
    def append(self, session_id: str, role: str, content: str) -> None: ...
    def clear(self, session_id: str) -> None: ...


class InMemorySessionStore:
    def __init__(self):
        self._sessions: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._sessions[session_id])

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            s = self._sessions[session_id]
            s.append({"role": role, "content": content})
            del s[:-2 * MAX_TURNS]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class RedisSessionStore:
    def __init__(self, url: str, ttl_s: int = 24 * 3600):
        import redis
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self.ttl_s = ttl_s

    def _k(self, session_id: str) -> str:
        return f"secrag:session:{session_id}"

    def get(self, session_id: str) -> list[dict]:
        return [json.loads(m) for m in self._r.lrange(self._k(session_id), 0, -1)]

    def append(self, session_id: str, role: str, content: str) -> None:
        k = self._k(session_id)
        pipe = self._r.pipeline()
        pipe.rpush(k, json.dumps({"role": role, "content": content}))
        pipe.ltrim(k, -2 * MAX_TURNS, -1)
        pipe.expire(k, self.ttl_s)
        pipe.execute()

    def clear(self, session_id: str) -> None:
        self._r.delete(self._k(session_id))


def make_session_store(redis_url: str | None = None) -> SessionStore:
    if redis_url:
        return RedisSessionStore(redis_url)
    return InMemorySessionStore()
