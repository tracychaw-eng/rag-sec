import fakeredis

from sec_rag.cache import RedisCache, make_cache


def _fake_cache(ttl_s=60):
    c = RedisCache.__new__(RedisCache)
    c._r = fakeredis.FakeRedis(decode_responses=True)
    c.ns = "test"
    c.ttl_s = ttl_s
    c.hits = 0
    c.misses = 0
    return c


def test_roundtrip_json_values():
    c = _fake_cache()
    c.set("k", [1.0, 2.5])
    assert c.get("k") == [1.0, 2.5]
    c.set("s", "a json string")
    assert c.get("s") == "a json string"


def test_miss_returns_none_and_counts():
    c = _fake_cache()
    assert c.get("nope") is None
    assert c.stats()["misses"] == 1
    assert c.stats()["backend"] == "redis"


def test_make_cache_selects_backend():
    mem = make_cache("ns", 10, 60, redis_url=None)
    assert mem.stats()["backend"] == "memory"
