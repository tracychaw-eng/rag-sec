import fakeredis

from sec_rag import ratelimit
from sec_rag.ratelimit import (RedisFixedWindowLimiter, SlidingWindowLimiter,
                               make_limiter)


def _fake_limiter(per_minute):
    lim = RedisFixedWindowLimiter.__new__(RedisFixedWindowLimiter)
    lim._r = fakeredis.FakeRedis(decode_responses=True)
    lim.per_minute = per_minute
    return lim


def _freeze_time(monkeypatch, epoch=1_751_000_010.0):
    # Pin the clock mid-window: the fixed window must not roll over mid-test
    monkeypatch.setattr(ratelimit.time, "time", lambda: epoch)


def test_redis_limiter_allows_up_to_limit(monkeypatch):
    _freeze_time(monkeypatch)
    lim = _fake_limiter(3)
    assert lim.check("k") is None
    assert lim.check("k") is None
    assert lim.check("k") is None
    wait = lim.check("k")
    assert wait is not None and 0 < wait <= 60


def test_redis_limiter_is_per_key(monkeypatch):
    _freeze_time(monkeypatch)
    lim = _fake_limiter(1)
    assert lim.check("a") is None
    assert lim.check("b") is None
    assert lim.check("a") is not None


def test_make_limiter_selects_backend():
    assert isinstance(make_limiter(10, None), SlidingWindowLimiter)
