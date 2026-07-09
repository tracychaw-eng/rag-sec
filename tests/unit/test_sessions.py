import fakeredis

from sec_rag.sessions import (MAX_TURNS, InMemorySessionStore,
                              RedisSessionStore)


def test_inmemory_roundtrip_and_isolation():
    s = InMemorySessionStore()
    s.append("a", "user", "hi")
    s.append("a", "assistant", "hello")
    assert s.get("a") == [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "hello"}]
    assert s.get("b") == []


def test_inmemory_trims_history():
    s = InMemorySessionStore()
    for i in range(MAX_TURNS * 3):
        s.append("a", "user", f"q{i}")
        s.append("a", "assistant", f"a{i}")
    assert len(s.get("a")) == 2 * MAX_TURNS


def test_inmemory_clear():
    s = InMemorySessionStore()
    s.append("a", "user", "hi")
    s.clear("a")
    assert s.get("a") == []


def test_redis_store_roundtrip(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    store = RedisSessionStore.__new__(RedisSessionStore)
    store._r = fake
    store.ttl_s = 60

    store.append("a", "user", "hi")
    store.append("a", "assistant", "hello")
    assert store.get("a") == [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}]
    store.clear("a")
    assert store.get("a") == []


def test_redis_store_trims(monkeypatch):
    store = RedisSessionStore.__new__(RedisSessionStore)
    store._r = fakeredis.FakeRedis(decode_responses=True)
    store.ttl_s = 60
    for i in range(MAX_TURNS * 3):
        store.append("a", "user", f"q{i}")
        store.append("a", "assistant", f"a{i}")
    assert len(store.get("a")) == 2 * MAX_TURNS
