"""FastAPI service for the SEC RAG pipeline.

Run:
    uvicorn sec_rag.api.app:app --port 8000

Endpoints:
    GET  /health              — pipeline readiness + corpus/cache stats
    POST /query               — single-shot question
    POST /chat/{session_id}   — multi-turn; history feeds the query planner's
                                rewrite step (standalone-query condensation)

The pipeline is synchronous (OpenAI/Cohere sync clients), so endpoints are
plain `def` — FastAPI runs them on its threadpool. Session history is
in-process (dict); swap for Redis when running more than one replica.
"""

import threading
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..pipeline import RAGPipeline

MAX_HISTORY_TURNS = 10

_state: dict = {}
_sessions: dict[str, list[dict]] = defaultdict(list)
_sessions_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = RAGPipeline()   # loads chunk store + BM25 once
    yield
    _state.clear()


app = FastAPI(title="sec-rag", version="0.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ContextOut(BaseModel):
    ticker: str
    item: str
    filing_date: str
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    tickers: list[str]
    engine: str
    cached: bool
    contexts: list[ContextOut]
    latency_ms: float | None = None
    cost_usd: float | None = None
    trace_id: str | None = None


def _to_response(ans) -> QueryResponse:
    return QueryResponse(
        answer=ans.text,
        intent=ans.plan.intent,
        tickers=ans.plan.tickers,
        engine=ans.engine_used,
        cached=ans.cached,
        contexts=[
            ContextOut(
                ticker=c.parent.ticker, item=c.parent.item,
                filing_date=c.parent.filing_date,
                score=round(c.best_child_score, 4),
                preview=c.parent.text[:200],
            )
            for c in ans.contexts
        ],
        latency_ms=(ans.trace or {}).get("total_ms"),
        cost_usd=(ans.trace or {}).get("cost_usd"),
        trace_id=(ans.trace or {}).get("trace_id"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    pipeline: RAGPipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return {
        "status": "ok",
        "tickers": pipeline.retriever.store.available_tickers(),
        "children_indexed": len(pipeline.retriever.children_by_id),
        "parents": len(pipeline.retriever.parents_by_id),
        "collection": pipeline.cfg.collection,
        "caches": pipeline.cache_stats(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    pipeline: RAGPipeline = _state["pipeline"]
    ans = pipeline.answer(req.question)
    return _to_response(ans)


@app.post("/chat/{session_id}", response_model=QueryResponse)
def chat(session_id: str, req: QueryRequest):
    pipeline: RAGPipeline = _state["pipeline"]
    with _sessions_lock:
        history = list(_sessions[session_id])

    ans = pipeline.answer(req.question, history=history or None)

    with _sessions_lock:
        s = _sessions[session_id]
        s.append({"role": "user", "content": req.question})
        s.append({"role": "assistant", "content": ans.text})
        del s[:-2 * MAX_HISTORY_TURNS]
    return _to_response(ans)


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
