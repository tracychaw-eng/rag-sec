import fakeredis

from sec_rag.memory import (MAX_FACTS, InMemoryUserMemory, RedisUserMemory,
                            format_user_context)


def _fact(text, conf=0.9):
    return {"fact": text, "confidence": conf, "source_session": "s1",
            "ts": 1.0}


def test_facts_roundtrip_and_dedupe():
    m = InMemoryUserMemory()
    m.add_facts("u1", [_fact("User is a credit analyst")])
    m.add_facts("u1", [_fact("user is a credit analyst"),   # dupe (casefold)
                       _fact("Prefers tables")])
    facts = [f["fact"] for f in m.get_facts("u1")]
    assert facts == ["User is a credit analyst", "Prefers tables"]


def test_facts_capped():
    m = InMemoryUserMemory()
    m.add_facts("u1", [_fact(f"fact {i}") for i in range(MAX_FACTS + 20)])
    assert len(m.get_facts("u1")) == MAX_FACTS


def test_clear_erases_everything():
    m = InMemoryUserMemory()
    m.add_facts("u1", [_fact("x")])
    m.log_event("u1", {"ts": 1, "question": "q"})
    m.clear("u1")
    assert m.get_facts("u1") == []
    assert m.get_events("u1") == []


def test_users_isolated():
    m = InMemoryUserMemory()
    m.add_facts("u1", [_fact("u1 fact")])
    assert m.get_facts("u2") == []


def test_redis_memory_roundtrip():
    m = RedisUserMemory.__new__(RedisUserMemory)
    m._r = fakeredis.FakeRedis(decode_responses=True)
    m.add_facts("u1", [_fact("User covers financials")])
    m.add_facts("u1", [_fact("user covers financials")])  # dupe
    assert len(m.get_facts("u1")) == 1
    m.log_event("u1", {"ts": 1, "question": "q"})
    assert len(m.get_events("u1")) == 1
    m.clear("u1")
    assert m.get_facts("u1") == []


def test_format_user_context():
    assert format_user_context([]) is None
    ctx = format_user_context([_fact("Prefers tables")])
    assert "Prefers tables" in ctx and "Known about this user" in ctx
