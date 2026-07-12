"""FastAPI service for the SEC RAG pipeline.

Run:
    uvicorn sec_rag.api.app:app --port 8000

Endpoints (all under X-API-Key auth when SECRAG_API_KEYS is set):
    GET    /health                  — readiness + corpus/cache stats (no auth)
    POST   /query                   — single-shot question
    POST   /chat/{session_id}        — multi-turn (planner condenses follow-ups)
    POST   /chat/{session_id}/stream — same, but SSE token streaming
    DELETE /chat/{session_id}        — clear session

Sessions and caches are in-process by default; set SECRAG_REDIS_URL to
share them across replicas.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..memory import MemoryExtractor, format_user_context, make_user_memory
from ..observability.logs import setup_logging
from ..pipeline import RAGPipeline
from ..ratelimit import make_limiter
from ..sessions import make_session_store

logger = logging.getLogger("sec_rag.api")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = RAGPipeline()          # loads chunk store + BM25 once
    setup_logging(json_output=pipeline.cfg.log_json)
    _state["pipeline"] = pipeline
    _state["sessions"] = make_session_store(pipeline.cfg.redis_url)
    _state["limiter"] = make_limiter(pipeline.cfg.rate_limit_per_minute,
                                     pipeline.cfg.redis_url)
    _state["user_memory"] = make_user_memory(pipeline.cfg.redis_url)
    _state["memory_extractor"] = MemoryExtractor(
        pipeline.openai, pipeline.cfg.llm_model, _state["user_memory"],
        pipeline.cfg.memory_confidence_floor)
    logger.info("sec-rag api ready", extra={
        "children": len(pipeline.retriever.children_by_id),
        "auth": bool(pipeline.cfg.api_key_set),
        "redis": bool(pipeline.cfg.redis_url),
    })
    if not pipeline.cfg.api_key_set:
        logger.warning(
            "SECRAG_API_KEYS not set — API auth is DISABLED. "
            "Do not expose this server beyond localhost.")
    yield
    _state.clear()


app = FastAPI(title="sec-rag", version="0.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Auth + rate-limit dependency
# ---------------------------------------------------------------------------
def require_caller(x_api_key: str | None = Header(default=None)) -> str:
    pipeline: RAGPipeline = _state["pipeline"]
    keys = pipeline.cfg.api_key_set

    if keys:
        if x_api_key is None:
            raise HTTPException(status_code=401,
                                detail="Missing X-API-Key header")
        if x_api_key not in keys:
            raise HTTPException(status_code=403, detail="Invalid API key")
        caller = x_api_key
    else:
        caller = "anonymous"          # dev mode — auth disabled

    retry_after = _state["limiter"].check(caller)
    if retry_after is not None:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1)})
    return caller


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
@app.get("/", include_in_schema=False)
def ui():
    """Chat UI — a single self-contained page; the API calls it makes are
    the same authenticated endpoints as any other client."""
    return FileResponse(Path(__file__).parent / "ui.html",
                        media_type="text/html")


@app.get("/health")
def health():
    pipeline: RAGPipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return {
        "status": "ok",
        "auth": "enabled" if pipeline.cfg.api_key_set else "DISABLED",
        "tickers": pipeline.retriever.store.available_tickers(),
        "filings": [f"{t} {d}" for t, d
                    in pipeline.retriever.store.available_filings()],
        "children_indexed": len(pipeline.retriever.children_by_id),
        "parents": len(pipeline.retriever.parents_by_id),
        "collection": pipeline.cfg.collection,
        "qdrant": pipeline.cfg.qdrant_url or "embedded",
        "caches": pipeline.cache_stats(),
        "judge": pipeline.judge.stats() if pipeline.judge else {"sample_rate": 0},
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, caller: str = Depends(require_caller)):
    pipeline: RAGPipeline = _state["pipeline"]
    ans = await pipeline.answer(req.question)
    return _to_response(ans)


def _user_context_for(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return format_user_context(_state["user_memory"].get_facts(user_id))


def _after_chat_turn(user_id: str | None, session_id: str,
                     question: str, answer_text: str, intent: str,
                     tickers: list[str]) -> None:
    """Episodic log + fire-and-forget semantic fact extraction."""
    if not user_id:
        return
    _state["user_memory"].log_event(user_id, {
        "ts": time.time(), "session_id": session_id,
        "question": question[:300], "intent": intent, "tickers": tickers,
    })
    asyncio.create_task(_state["memory_extractor"].extract(
        user_id, session_id, question, answer_text))


@app.post("/chat/{session_id}", response_model=QueryResponse)
async def chat(session_id: str, req: QueryRequest,
               caller: str = Depends(require_caller),
               x_user_id: str | None = Header(default=None)):
    pipeline: RAGPipeline = _state["pipeline"]
    sessions = _state["sessions"]
    history = sessions.get(session_id)

    ans = await pipeline.answer(req.question, history=history or None,
                                user_context=_user_context_for(x_user_id))

    sessions.append(session_id, "user", req.question)
    sessions.append(session_id, "assistant", ans.text)
    _after_chat_turn(x_user_id, session_id, req.question, ans.text,
                     ans.plan.intent, ans.plan.tickers)
    return _to_response(ans)


@app.post("/chat/{session_id}/stream")
async def chat_stream(session_id: str, req: QueryRequest,
                      caller: str = Depends(require_caller),
                      x_user_id: str | None = Header(default=None)):
    """Server-Sent Events: `meta` → `delta`* → `done`."""
    pipeline: RAGPipeline = _state["pipeline"]
    sessions = _state["sessions"]
    history = sessions.get(session_id)

    async def event_stream():
        answer_text, meta = "", {}
        async for event in pipeline.answer_stream(
                req.question, history=history or None,
                user_context=_user_context_for(x_user_id)):
            if event["type"] == "meta":
                meta = event
            if event["type"] == "done":
                answer_text = event["answer"]
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        sessions.append(session_id, "user", req.question)
        sessions.append(session_id, "assistant", answer_text)
        _after_chat_turn(x_user_id, session_id, req.question, answer_text,
                         meta.get("intent", ""), meta.get("tickers", []))

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str, caller: str = Depends(require_caller)):
    _state["sessions"].clear(session_id)
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# User memory — governance: fully visible, fully erasable
# ---------------------------------------------------------------------------
@app.get("/memory/{user_id}")
def get_memory(user_id: str, caller: str = Depends(require_caller)):
    mem = _state["user_memory"]
    return {
        "user_id": user_id,
        "facts": mem.get_facts(user_id),
        "recent_events": mem.get_events(user_id, limit=20),
    }


@app.delete("/memory/{user_id}")
def delete_memory(user_id: str, caller: str = Depends(require_caller)):
    _state["user_memory"].clear(user_id)
    logger.info("user memory erased", extra={"user_id": user_id})
    return {"status": "erased", "user_id": user_id}
