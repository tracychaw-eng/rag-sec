"""End-to-end RAG pipeline: plan → retrieve (intent-aware) → generate.

This replaces the legacy per-question-ID routing with runtime decisions:

  intent      retrieval                                  generation
  ---------   ----------------------------------------   ------------------
  factual     hybrid → rerank (floor)                    direct + citations
  reasoning   hybrid, NO reranker (indirect evidence)    evidence/inference
  comparison  per-ticker hybrid+rerank loops             grouped synthesis
"""

from openai import OpenAI

from .config import Settings, get_settings
from .generation.generator import Generator
from .models import Answer, ContextBlock
from .retrieval.planner import QueryPlanner
from .retrieval.retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or get_settings()
        openai_client = OpenAI(api_key=self.cfg.openai_api_key)
        self.planner = QueryPlanner(self.cfg, openai_client)
        self.retriever = HybridRetriever(self.cfg, openai_client=openai_client)
        self.generator = Generator(self.cfg, openai_client)

    def answer(self, question: str,
               history: list[dict] | None = None) -> Answer:
        plan = self.planner.plan(question, history)
        queries = [plan.rewritten_query, *plan.paraphrases]

        if plan.intent == "comparison":
            # Per-ticker retrieval: the only guaranteed source-diversity fix.
            # A comparison naming no specific companies compares all of them.
            tickers = (plan.tickers if len(plan.tickers) >= 2
                       else self.retriever.store.available_tickers())
            contexts: list[ContextBlock] = []
            for ticker in tickers:
                contexts.extend(self.retriever.retrieve(
                    queries, mode="per_ticker", tickers=[ticker]))
            engine = f"per-ticker({','.join(tickers)})"

        elif plan.intent == "reasoning":
            # "Which company is most X" needs evidence from all companies:
            # if multiple/no specific tickers, retrieve per ticker to
            # guarantee cross-company evidence, else single reasoning pass.
            tickers = plan.tickers or self.retriever.store.available_tickers()
            if len(tickers) >= 2:
                contexts = []
                per_ticker_cap = self.cfg.reasoning_parents_per_ticker
                for ticker in tickers:
                    blocks = self.retriever.retrieve(
                        queries, mode="reasoning", tickers=[ticker])
                    contexts.extend(blocks[:per_ticker_cap])
            else:
                contexts = self.retriever.retrieve(
                    queries, mode="reasoning", tickers=tickers)
            engine = f"reasoning({','.join(tickers)})"

        else:  # factual
            contexts = self.retriever.retrieve(
                queries, mode="precise", tickers=plan.tickers or None)
            engine = "hybrid+rerank"

        text = self.generator.generate(question, plan, contexts)
        return Answer(question=question, text=text, plan=plan,
                      contexts=contexts, engine_used=engine)
