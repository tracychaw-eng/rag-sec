import pytest
from fastapi import HTTPException

from sec_rag.api import app as app_module
from sec_rag.api.app import require_caller
from sec_rag.ratelimit import SlidingWindowLimiter


class _StubCfg:
    def __init__(self, keys):
        self.api_key_set = frozenset(keys)


class _StubPipeline:
    def __init__(self, keys):
        self.cfg = _StubCfg(keys)


def _setup(keys, per_minute=100):
    app_module._state["pipeline"] = _StubPipeline(keys)
    app_module._state["limiter"] = SlidingWindowLimiter(per_minute)


def teardown_function():
    app_module._state.clear()


def test_valid_key_allowed():
    _setup({"k1", "k2"})
    assert require_caller("k1") == "k1"


def test_missing_key_401():
    _setup({"k1"})
    with pytest.raises(HTTPException) as e:
        require_caller(None)
    assert e.value.status_code == 401


def test_wrong_key_403():
    _setup({"k1"})
    with pytest.raises(HTTPException) as e:
        require_caller("bad")
    assert e.value.status_code == 403


def test_auth_disabled_when_no_keys():
    _setup(set())
    assert require_caller(None) == "anonymous"


def test_rate_limit_429_with_retry_after():
    _setup({"k1"}, per_minute=2)
    require_caller("k1")
    require_caller("k1")
    with pytest.raises(HTTPException) as e:
        require_caller("k1")
    assert e.value.status_code == 429
    assert "Retry-After" in e.value.headers


def test_rate_limit_is_per_key():
    _setup({"k1", "k2"}, per_minute=1)
    require_caller("k1")
    require_caller("k2")  # different key, own budget


def test_limiter_window_recovers():
    lim = SlidingWindowLimiter(per_minute=1)
    assert lim.check("x") is None
    wait = lim.check("x")
    assert wait is not None and 0 <= wait <= 60
