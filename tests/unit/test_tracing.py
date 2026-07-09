import json
from types import SimpleNamespace

from sec_rag.observability import tracing


def test_spans_and_costs_accumulate():
    with tracing.start_trace("t", question="q") as trace:
        with trace.span("plan"):
            tracing.record_llm(
                "gpt-4o-mini",
                SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
                kind="planner")
        with trace.span("retrieve"):
            tracing.record_rerank(2)
    s = trace.summary()
    assert [sp["name"] for sp in s["spans"]] == ["plan", "retrieve"]
    assert s["prompt_tokens"] == 1000
    assert s["completion_tokens"] == 500
    # 1000*0.15/1M + 500*0.60/1M + 2*0.002
    assert abs(s["cost_usd"] - (0.00015 + 0.0003 + 0.004)) < 1e-9
    assert s["question"] == "q"


def test_record_outside_trace_is_noop():
    tracing.record_llm("gpt-4o-mini",
                       SimpleNamespace(prompt_tokens=5, completion_tokens=5))
    tracing.record_rerank()
    assert tracing.active() is None


def test_jsonl_sink_writes(tmp_path):
    sink = tracing.JsonlSink(tmp_path / "traces.jsonl")
    sink.export({"trace_id": "abc", "cost_usd": 0.1})
    line = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["trace_id"] == "abc"
