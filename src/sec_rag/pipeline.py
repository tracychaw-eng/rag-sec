"""End-to-end RAG pipeline: plan → retrieve (intent-aware) → generate.

This replaces the legacy per-question-ID routing with runtime decisions:

  intent      retrieval                                  generation
  ---------   ----------------------------------------   ------------------
  factual     hybrid → rerank (floor + min_keep)         direct + citations
  reasoning   hybrid, NO reranker (indirect evidence)    evidence/inference
  comparison  per-ticker hybrid+rerank loops             grouped synthesis

Cross-cutting (Phase 2): per-request tracing with token/cost metering
(exported to Langfuse or local JSONL), answer caching keyed on
(question, corpus, prompt version), query-embedding caching, retries on
every external call.
"""

from openai import OpenAI

from .cache import TTLCache, cache_key
from .config import Settings, get_settings
from .generation.generator import PROMPT_VERSION, Generator
from .models import Answer, ContextBlock
from .observability import tracing
from .retrieval.planner import QueryPlanner
from .retrieval.retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or get_settings()
        openai_client = OpenAI(
            api_key=self.cfg.openai_api_key,
            timeout=self.cfg.openai_timeout_s,
            max_retries=self.cfg.openai_max_retries,
        )
        self.planner = QueryPlanner(self.cfg, openai_client)
        self.retriever = HybridRetriever(self.cfg, openai_client=openai_client)
        self.generator = Generator(self.cfg, openai_client)
        self.answer_cache = TTLCache(
            max_items=self.cfg.cache_max_items, ttl_s=self.cfg.cache_ttl_s)
        self.trace_sink = tracing.make_sink(
            self.cfg.langfuse_enabled, self.cfg.traces_path)

    def cache_stats(self) -> dict:
        return {
            "answer_cache": self.answer_cache.stats(),
            "embed_cache": self.retriever.embed_cache.stats(),
        }

    # ------------------------------------------------------------------
    def answer(self, question: str,
               history: list[dict] | None = None) -> Answer:
        # Answer cache: single-shot questions only — follow-ups depend on
        # conversation state. Key includes corpus + prompt versions.
        ans_key = None
        if self.cfg.enable_answer_cache and not history:
            ans_key = cache_key("answer", self.cfg.collection,
                                PROMPT_VERSION, question.strip().lower())
            hit = self.answer_cache.get(ans_key)
            if hit is not None:
                return hit.model_copy(update={"cached": True})

        with tracing.start_trace("rag.answer", question=question) as trace:
            with trace.span("plan"):
                plan = self.planner.plan(question, history)
            queries = [plan.rewritten_query, *plan.paraphrases]

            with trace.span("retrieve", intent=plan.intent,
                            tickers=",".join(plan.tickers)):
                contexts, engine = self._retrieve(plan, queries)

            with trace.span("generate", n_contexts=len(contexts)):
                text = self.generator.generate(question, plan, contexts)

            summary = trace.summary()
            summary["engine"] = engine
            summary["answer_preview"] = text[:200]
            self.trace_sink.export(summary)

        answer = Answer(question=question, text=text, plan=plan,
                        contexts=contexts, engine_used=engine, trace=summary)
        if ans_key is not None:
            self.answer_cache.set(ans_key, answer)
        return answer

    # ------------------------------------------------------------------
    def _retrieve(self, plan, queries) -> tuple[list[ContextBlock], str]:
        if plan.intent == "comparison":
            # Per-ticker retrieval: the only guaranteed source-diversity fix.
            # A comparison naming no specific companies compares all of them.
            tickers = (plan.tickers if len(plan.tickers) >= 2
                       else self.retriever.store.available_tickers())
            contexts: list[ContextBlock] = []
            for ticker in tickers:
                contexts.extend(self.retriever.retrieve(
                    queries, mode="per_ticker", tickers=[ticker]))
            return contexts, f"per-ticker({','.join(tickers)})"

        if plan.intent == "reasoning":
            # "Which company is most X" needs evidence from all companies:
            # fan out per ticker to guarantee cross-company evidence.
            tickers = plan.tickers or self.retriever.store.available_tickers()
            if len(tickers) >= 2:
                contexts = []
                for ticker in tickers:
                    blocks = self.retriever.retrieve(
                        queries, mode="reasoning", tickers=[ticker])
                    contexts.extend(
                        blocks[:self.cfg.reasoning_parents_per_ticker])
            else:
                contexts = self.retriever.retrieve(
                    queries, mode="reasoning", tickers=tickers)
            return contexts, f"reasoning({','.join(tickers)})"

        # factual
        contexts = self.retriever.retrieve(
            queries, mode="precise", tickers=plan.tickers or None)
        return contexts, "hybrid+rerank"
