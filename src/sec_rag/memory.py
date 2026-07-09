"""Per-user long-term memory: semantic facts + episodic event log.

Semantic memory = durable facts about the USER ("credit analyst",
"prefers tables", "covers financials") extracted from chat turns by an
LLM pass. Episodic memory = what they asked and when.

Governance rules (enforced in code, not just prompt):
  - Facts are about the user, never document content — the corpus stores
    facts about companies, memory stores facts about people (pollution
    rule from the architecture review).
  - Every fact carries provenance (session, timestamp) and a confidence;
    writes below the confidence floor are dropped.
  - Case-insensitive dedupe; capped per user (oldest evicted).
  - Fully user-visible and erasable via GET/DELETE /memory/{user_id}.
"""

import json
import logging
import threading
import time
from collections import defaultdict
from typing import Protocol

from openai import AsyncOpenAI

logger = logging.getLogger("sec_rag.memory")

MAX_FACTS = 50
MAX_EVENTS = 200

EXTRACTION_PROMPT = """You extract durable facts about a USER from a chat exchange
with a financial-filings assistant.

Extract ONLY facts about the user themselves: their role, expertise level,
preferences (format, depth, style), and what companies/topics they focus on.

NEVER extract:
- facts about companies or filings (that is the document corpus's job)
- one-off situational details ("user asked about X today")
- sensitive personal data

Return JSON: {"facts": [{"fact": "short sentence", "confidence": 0.0-1.0}]}
Return {"facts": []} when the exchange reveals nothing durable — most don't."""


class UserMemoryStore(Protocol):
    def get_facts(self, user_id: str) -> list[dict]: ...
    def add_facts(self, user_id: str, facts: list[dict]) -> None: ...
    def log_event(self, user_id: str, event: dict) -> None: ...
    def get_events(self, user_id: str, limit: int = 50) -> list[dict]: ...
    def clear(self, user_id: str) -> None: ...


class InMemoryUserMemory:
    def __init__(self):
        self._facts: dict[str, list[dict]] = defaultdict(list)
        self._events: dict[str, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def get_facts(self, user_id):
        with self._lock:
            return list(self._facts[user_id])

    def add_facts(self, user_id, facts):
        with self._lock:
            existing = self._facts[user_id]
            known = {f["fact"].casefold() for f in existing}
            for f in facts:
                if f["fact"].casefold() not in known:
                    existing.append(f)
                    known.add(f["fact"].casefold())
            del existing[:-MAX_FACTS]

    def log_event(self, user_id, event):
        with self._lock:
            self._events[user_id].append(event)
            del self._events[user_id][:-MAX_EVENTS]

    def get_events(self, user_id, limit=50):
        with self._lock:
            return list(self._events[user_id][-limit:])

    def clear(self, user_id):
        with self._lock:
            self._facts.pop(user_id, None)
            self._events.pop(user_id, None)


class RedisUserMemory:
    def __init__(self, url: str):
        import redis
        self._r = redis.Redis.from_url(url, decode_responses=True)

    def _fk(self, uid):
        return f"secrag:memfacts:{uid}"

    def _ek(self, uid):
        return f"secrag:memevents:{uid}"

    def get_facts(self, user_id):
        return [json.loads(x) for x in self._r.lrange(self._fk(user_id), 0, -1)]

    def add_facts(self, user_id, facts):
        known = {f["fact"].casefold() for f in self.get_facts(user_id)}
        pipe = self._r.pipeline()
        for f in facts:
            if f["fact"].casefold() not in known:
                pipe.rpush(self._fk(user_id), json.dumps(f))
                known.add(f["fact"].casefold())
        pipe.ltrim(self._fk(user_id), -MAX_FACTS, -1)
        pipe.execute()

    def log_event(self, user_id, event):
        pipe = self._r.pipeline()
        pipe.rpush(self._ek(user_id), json.dumps(event))
        pipe.ltrim(self._ek(user_id), -MAX_EVENTS, -1)
        pipe.execute()

    def get_events(self, user_id, limit=50):
        return [json.loads(x)
                for x in self._r.lrange(self._ek(user_id), -limit, -1)]

    def clear(self, user_id):
        self._r.delete(self._fk(user_id), self._ek(user_id))


def make_user_memory(redis_url: str | None = None) -> UserMemoryStore:
    if redis_url:
        return RedisUserMemory(redis_url)
    return InMemoryUserMemory()


class MemoryExtractor:
    def __init__(self, openai_client: AsyncOpenAI, model: str,
                 store: UserMemoryStore, confidence_floor: float = 0.7):
        self.openai = openai_client
        self.model = model
        self.store = store
        self.confidence_floor = confidence_floor

    async def extract(self, user_id: str, session_id: str,
                      question: str, answer: str) -> None:
        """Background task — must never raise into the caller."""
        try:
            resp = await self.openai.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content":
                        f"User said: {question}\nAssistant replied: "
                        f"{answer[:500]}"},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            accepted = [
                {"fact": f["fact"].strip(),
                 "confidence": float(f["confidence"]),
                 "source_session": session_id,
                 "ts": time.time()}
                for f in data.get("facts", [])
                if isinstance(f.get("fact"), str) and f["fact"].strip()
                and float(f.get("confidence", 0)) >= self.confidence_floor
            ]
            if accepted:
                self.store.add_facts(user_id, accepted)
                logger.info("memory facts stored",
                            extra={"user_id": user_id, "n": len(accepted)})
        except Exception:
            logger.warning("memory extraction failed", exc_info=True,
                           extra={"user_id": user_id})


def format_user_context(facts: list[dict]) -> str | None:
    if not facts:
        return None
    lines = "\n".join(f"- {f['fact']}" for f in facts[-10:])
    return f"Known about this user (apply when relevant):\n{lines}"
