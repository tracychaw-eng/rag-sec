import json
import logging

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter)

from sec_rag.observability.logs import JsonFormatter
from sec_rag.observability.tracing import MultiSink, OtelSink

SUMMARY = {
    "trace_id": "abc123", "name": "rag.answer", "started_at": 1751000000.0,
    "total_ms": 5000.0, "engine": "hybrid+rerank",
    "prompt_tokens": 900, "completion_tokens": 100,
    "rerank_searches": 1, "cost_usd": 0.001,
    "spans": [
        {"name": "plan", "ms": 1000.0, "offset_ms": 0.0},
        {"name": "retrieve", "ms": 800.0, "offset_ms": 1000.0,
         "intent": "factual"},
        {"name": "generate", "ms": 3000.0, "offset_ms": 1900.0},
    ],
}


def _otel_sink_with_memory_exporter():
    exporter = InMemorySpanExporter()
    return OtelSink(exporter=exporter), exporter


def test_otel_sink_exports_root_and_children():
    sink, exporter = _otel_sink_with_memory_exporter()
    sink.export(SUMMARY)
    spans = exporter.get_finished_spans()
    names = sorted(s.name for s in spans)
    assert names == ["generate", "plan", "rag.answer", "retrieve"]

    root = next(s for s in spans if s.name == "rag.answer")
    children = [s for s in spans if s.name != "rag.answer"]
    assert all(c.parent is not None and
               c.parent.span_id == root.context.span_id for c in children)
    assert root.attributes["cost_usd"] == 0.001

    retrieve = next(s for s in spans if s.name == "retrieve")
    # start offset honored: 1000ms after trace start
    assert retrieve.start_time == int(1751000000.0 * 1e9) + 1_000_000_000
    assert retrieve.end_time - retrieve.start_time == 800_000_000


def test_multisink_isolates_failures():
    class Boom:
        def export(self, s):
            raise RuntimeError("x")

    class Ok:
        def __init__(self):
            self.got = None

        def export(self, s):
            self.got = s

    ok = Ok()
    MultiSink([Boom(), ok]).export(SUMMARY)
    assert ok.got is SUMMARY


def test_json_formatter_includes_extras():
    rec = logging.LogRecord("sec_rag.api", logging.WARNING, "f.py", 1,
                            "rate limited", (), None)
    rec.caller = "key-1"
    out = json.loads(JsonFormatter().format(rec))
    assert out["level"] == "WARNING"
    assert out["message"] == "rate limited"
    assert out["caller"] == "key-1"
    assert out["ts"].endswith("Z")
