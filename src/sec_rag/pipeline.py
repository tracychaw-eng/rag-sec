"""End-to-end RAG pipeline: plan → retrieve (intent-aware) → generate.

This replaces the legacy per-question-ID routing with runtime decisions:

  intent      retrieval                                  generation
  ---------   ----------------------------------------   ------------------
  factual     hybrid → rerank (floor + min_keep)         direct + citations
  reasoning   hybrid, NO reranker (indirect evidence)    evidence/inference
  comparison  per-ticker hybrid+rerank loops             grouped synthesis

Fully async; per-ticker fan-outs (comparison/reasoning) run concurrently.
`answer_sync()` is the facade for CLI/eval callers. `answer_stream()`
yields SSE-ready events (meta → delta* → done).

Cross-cutting: per-request tracing with token/cost metering (Langfuse or
local JSONL), answer + query-embedding caching (in-process or Redis),
retries on every external call.
"""

import asyncio

from openai import AsyncOpenAI

from .cache import cache_key, make_cache
from .config import Settings, get_settings
from .generation.generator import PROMPT_VERSION, Generator
from .models import Answer, ContextBlock, QueryPlan
from .observability import tracing
from .retrieval.planner import QueryPlanner
from .retrieval.retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or get_settings()
        openai_client = AsyncOpenAI(
            api_key=self.cfg.openai_api_key,
            timeout=self.cfg.openai_timeout_s,
            max_retries=self.cfg.openai_max_retries,
        )
        embed_cache = make_cache("embed", self.cfg.cache_max_items,
                                 self.cfg.cache_ttl_s, self.cfg.redis_url)
        self.planner = QueryPlanner(self.cfg, openai_client)
        self.retriever = HybridRetriever(self.cfg, openai_client=openai_client,
                                         embed_cache=embed_cache)
        self.generator = Generator(self.cfg, openai_client)
        self.answer_cache = make_cache("answer", self.cfg.cache_max_items,
                                       self.cfg.cache_ttl_s, self.cfg.redis_url)
        self.trace_sink = tracing.make_sink(
            self.cfg.langfuse_enabled, self.cfg.traces_path)
        self._sync_loop: asyncio.AbstractEventLoop | None = None

    def cache_stats(self) -> dict:
        return {
            "answer_cache": self.answer_cache.stats(),
            "embed_cache": self.retriever.embed_cache.stats(),
        }

    # ------------------------------------------------------------------
    def _answer_cache_key(self, question: str,
                          history: list[dict] | None) -> str | None:
        # Single-shot questions only — follow-ups depend on conversation
        # state. Key includes corpus + prompt versions.
        if not self.cfg.enable_answer_cache or history:
            return None
        return cache_key("answer", self.cfg.collection, PROMPT_VERSION,
                         question.strip().lower())

    def _cached_answer(self, key: str | None) -> Answer | None:
        if key is None:
            return None
        hit = self.answer_cache.get(key)
        if hit is None:
            return None
        return Answer.model_validate_json(hit).model_copy(
            update={"cached": True})

    # ------------------------------------------------------------------
    async def answer(self, question: str,
                     history: list[dict] | None = None) -> Answer:
        ans_key = self._answer_cache_key(question, history)
        cached = self._cached_answer(ans_key)
        if cached is not None:
            return cached

        with tracing.start_trace("rag.answer", question=question) as trace:
            with trace.span("plan"):
                plan = await self.planner.plan(question, history)
            queries = [plan.rewritten_query, *plan.paraphrases]

            with trace.span("retrieve", intent=plan.intent,
                            tickers=",".join(plan.tickers)):
                contexts, engine = await self._retrieve(plan, queries)

            with trace.span("generate", n_contexts=len(contexts)):
                text = await self.generator.generate(question, plan, contexts)

            summary = trace.summary()
            summary["engine"] = engine
            summary["answer_preview"] = text[:200]
            self.trace_sink.export(summary)

        answer = Answer(question=question, text=text, plan=plan,
                        contexts=contexts, engine_used=engine, trace=summary)
        if ans_key is not None:
            self.answer_cache.set(ans_key, answer.model_dump_json())
        return answer

    def answer_sync(self, question: str,
                    history: list[dict] | None = None) -> Answer:
        """Blocking facade for CLI and eval callers.

        Reuses one private event loop across calls — asyncio.run() per call
        would close the loop each time, orphaning the async clients' pooled
        connections (observed as intermittent RuntimeError in the Cohere
        client on the 2nd+ call).
        """
        if self._sync_loop is None:
            self._sync_loop = asyncio.new_event_loop()
        return self._sync_loop.run_until_complete(
            self.answer(question, history))

    # ------------------------------------------------------------------
    async def answer_stream(self, question: str,
                            history: list[dict] | None = None):
        """Async generator of events:
            {"type": "meta", intent, tickers, engine, cached}
            {"type": "delta", "text": ...}          (0..n)
            {"type": "done", answer, latency_ms, cost_usd, trace_id, contexts}
        """
        ans_key = self._answer_cache_key(question, history)
        cached = self._cached_answer(ans_key)
        if cached is not None:
            yield {"type": "meta", "intent": cached.plan.intent,
                   "tickers": cached.plan.tickers,
                   "engine": cached.engine_used, "cached": True}
            yield {"type": "delta", "text": cached.text}
            yield {"type": "done", "answer": cached.text,
                   "latency_ms": 0, "cost_usd": 0,
                   "trace_id": (cached.trace or {}).get("trace_id"),
                   "contexts": _context_meta(cached.contexts)}
            return

        with tracing.start_trace("rag.answer_stream",
                                 question=question) as trace:
            with trace.span("plan"):
                plan = await self.planner.plan(question, history)
            queries = [plan.rewritten_query, *plan.paraphrases]

            with trace.span("retrieve", intent=plan.intent,
                            tickers=",".join(plan.tickers)):
                contexts, engine = await self._retrieve(plan, queries)

            yield {"type": "meta", "intent": plan.intent,
                   "tickers": plan.tickers, "engine": engine, "cached": False}

            parts: list[str] = []
            with trace.span("generate", n_contexts=len(contexts)):
                async for delta in self.generator.stream(
                        question, plan, contexts):
                    parts.append(delta)
                    yield {"type": "delta", "text": delta}

            text = "".join(parts).strip()
            summary = trace.summary()
            summary["engine"] = engine
            summary["answer_preview"] = text[:200]
            self.trace_sink.export(summary)

        answer = Answer(question=question, text=text, plan=plan,
                        contexts=contexts, engine_used=engine, trace=summary)
        if ans_key is not None:
            self.answer_cache.set(ans_key, answer.model_dump_json())

        yield {"type": "done", "answer": text,
               "latency_ms": summary["total_ms"],
               "cost_usd": summary["cost_usd"],
               "trace_id": summary["trace_id"],
               "contexts": _context_meta(contexts)}

    # ------------------------------------------------------------------
    async def _retrieve(self, plan: QueryPlan,
                        queries: list[str]) -> tuple[list[ContextBlock], str]:
        if plan.intent == "comparison":
            # Per-ticker retrieval: the only guaranteed source-diversity fix.
            # A comparison naming no specific companies compares all of them.
            tickers = (plan.tickers if len(plan.tickers) >= 2
                       else self.retriever.store.available_tickers())
            results = await asyncio.gather(*[
                self.retriever.retrieve(queries, mode="per_ticker",
                                        tickers=[t])
                for t in tickers
            ])
            contexts = [b for blocks in results for b in blocks]
            return contexts, f"per-ticker({','.join(tickers)})"

        if plan.intent == "reasoning":
            # "Which company is most X" needs evidence from all companies:
            # fan out per ticker (concurrently) to guarantee cross-company
            # evidence.
            tickers = plan.tickers or self.retriever.store.available_tickers()
            if len(tickers) >= 2:
                results = await asyncio.gather(*[
                    self.retriever.retrieve(queries, mode="reasoning",
                                            tickers=[t])
                    for t in tickers
                ])
                cap = self.cfg.reasoning_parents_per_ticker
                contexts = [b for blocks in results for b in blocks[:cap]]
            else:
                contexts = await self.retriever.retrieve(
                    queries, mode="reasoning", tickers=tickers)
            return contexts, f"reasoning({','.join(tickers)})"

        # factual
        contexts = await self.retriever.retrieve(
            queries, mode="precise", tickers=plan.tickers or None)
        return contexts, "hybrid+rerank"


def _context_meta(contexts: list[ContextBlock]) -> list[dict]:
    return [
        {"ticker": c.parent.ticker, "item": c.parent.item,
         "filing_date": c.parent.filing_date,
         "score": round(c.best_child_score, 4)}
        for c in contexts
    ]
