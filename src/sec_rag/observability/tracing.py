"""Request tracing + cost metering.

One `RequestTrace` per query, with named spans (plan / retrieve / generate)
and per-call LLM token+cost records. Components report usage through the
active trace held in a contextvar, so nothing threads a tracer argument
through every call.

Export: Langfuse when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are set,
otherwise JSONL to logs/traces.jsonl. Instrumentation is identical either
way — the sink is the only difference.
"""

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# $ per 1M tokens (update alongside model changes in config.py)
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}
# $ per rerank search (Cohere rerank-english-v3.0: $2.00 / 1k searches)
RERANK_PRICE_PER_SEARCH = 0.002

_current_trace: contextvars.ContextVar["RequestTrace | None"] = \
    contextvars.ContextVar("sec_rag_trace", default=None)


@dataclass
class Span:
    name: str
    start: float
    end: float = 0.0
    attrs: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round((self.end - self.start) * 1000, 1)


class RequestTrace:
    def __init__(self, name: str, **meta):
        self.id = uuid.uuid4().hex[:16]
        self.name = name
        self.meta = meta
        self.spans: list[Span] = []
        self.llm_calls: list[dict] = []
        self.rerank_searches = 0
        self._t0 = time.monotonic()
        self._started_at = time.time()

    @contextmanager
    def span(self, name: str, **attrs):
        s = Span(name=name, start=time.monotonic(), attrs=attrs)
        try:
            yield s
        finally:
            s.end = time.monotonic()
            self.spans.append(s)

    def record_llm(self, model: str, prompt_tokens: int,
                   completion_tokens: int, kind: str = "chat") -> None:
        p = PRICES.get(model, {"input": 0.0, "output": 0.0})
        cost = (prompt_tokens * p["input"]
                + completion_tokens * p["output"]) / 1_000_000
        self.llm_calls.append({
            "model": model, "kind": kind,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost, 8),
        })

    def record_rerank(self, n_searches: int = 1) -> None:
        self.rerank_searches += n_searches

    def summary(self) -> dict:
        llm_cost = sum(c["cost_usd"] for c in self.llm_calls)
        rerank_cost = self.rerank_searches * RERANK_PRICE_PER_SEARCH
        return {
            "trace_id": self.id,
            "name": self.name,
            "started_at": self._started_at,
            "total_ms": round((time.monotonic() - self._t0) * 1000, 1),
            "spans": [{"name": s.name, "ms": s.duration_ms, **s.attrs}
                      for s in self.spans],
            "llm_calls": self.llm_calls,
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.llm_calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.llm_calls),
            "rerank_searches": self.rerank_searches,
            "cost_usd": round(llm_cost + rerank_cost, 6),
            **self.meta,
        }


# ---------------------------------------------------------------------------
# Ambient access — components report into whatever trace is active
# ---------------------------------------------------------------------------
@contextmanager
def start_trace(name: str, **meta):
    trace = RequestTrace(name, **meta)
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


def active() -> "RequestTrace | None":
    return _current_trace.get()


def record_llm(model: str, usage, kind: str = "chat") -> None:
    """Report token usage from an OpenAI response into the active trace."""
    t = active()
    if t is not None and usage is not None:
        t.record_llm(model, getattr(usage, "prompt_tokens", 0),
                     getattr(usage, "completion_tokens", 0), kind)


def record_rerank(n: int = 1) -> None:
    t = active()
    if t is not None:
        t.record_rerank(n)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
class JsonlSink:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, summary: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")


class LangfuseSink:
    def __init__(self):
        from langfuse import Langfuse   # reads LANGFUSE_* env vars
        self._lf = Langfuse()

    def export(self, summary: dict) -> None:
        with self._lf.start_as_current_span(
                name=summary["name"],
                input={"question": summary.get("question")},
                output={"answer_preview": summary.get("answer_preview")},
        ) as root:
            root.update_trace(metadata={
                k: v for k, v in summary.items()
                if k not in ("spans", "llm_calls")})
            for s in summary["spans"]:
                with self._lf.start_as_current_span(name=s["name"]) as child:
                    child.update(metadata=s)
        self._lf.flush()


def make_sink(langfuse_enabled: bool, jsonl_path: Path):
    if langfuse_enabled:
        try:
            return LangfuseSink()
        except Exception as e:
            print(f"Langfuse init failed ({e}) — falling back to JSONL")
    return JsonlSink(jsonl_path)
