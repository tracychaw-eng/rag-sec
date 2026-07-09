import time

from sec_rag.cache import TTLCache, cache_key


def test_get_set_roundtrip():
    c = TTLCache(max_items=10, ttl_s=60)
    c.set("k", [1.0, 2.0])
    assert c.get("k") == [1.0, 2.0]


def test_ttl_expiry():
    c = TTLCache(max_items=10, ttl_s=0.01)
    c.set("k", "v")
    time.sleep(0.05)
    assert c.get("k") is None


def test_lru_eviction():
    c = TTLCache(max_items=2, ttl_s=60)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")          # a is now most-recently used
    c.set("c", 3)       # evicts b
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_stats_hit_rate():
    c = TTLCache()
    c.set("k", 1)
    c.get("k")
    c.get("missing")
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["hit_rate"] == 0.5


def test_cache_key_distinguishes_parts():
    assert cache_key("a", "bc") != cache_key("ab", "c")
    assert cache_key("x", "y") == cache_key("x", "y")
