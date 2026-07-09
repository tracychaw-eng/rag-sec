"""Domain models shared across ingestion, retrieval, and generation."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class ItemSection(BaseModel):
    """One Item section of a 10-K (e.g. Item 1A Risk Factors)."""
    item: str                    # "1", "1A", "7A", or "FULL" (fallback)
    title: str = ""
    text: str


class ParentChunk(BaseModel):
    """Large retrieval unit handed to the LLM (~parent_tokens)."""
    id: str
    ticker: str
    filing_date: str             # ISO date of the filing
    item: str
    item_title: str = ""
    text: str                    # includes the context header
    token_count: int = 0


class ChildChunk(BaseModel):
    """Small retrieval unit that is embedded and BM25-indexed (~child_tokens).

    text includes a context header ("NVDA 10-K filed 2026-02-25, Item 1A —
    Risk Factors: ...") so the vector, the BM25 tokens, and anything a judge
    sees all carry the company attribution that first-person 10-K prose lacks.
    """
    id: str
    parent_id: str
    ticker: str
    filing_date: str
    item: str
    item_title: str = ""
    text: str
    token_count: int = 0


# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------
QueryIntent = Literal["factual", "comparison", "reasoning"]


class QueryPlan(BaseModel):
    intent: QueryIntent = "factual"
    tickers: list[str] = Field(default_factory=list)  # [] = no restriction
    rewritten_query: str
    paraphrases: list[str] = Field(default_factory=list)
    reason: str = ""             # planner's one-line justification (for traces)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class ScoredChild(BaseModel):
    chunk: ChildChunk
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None


class ContextBlock(BaseModel):
    """A parent chunk selected for generation, with provenance."""
    parent: ParentChunk
    best_child_score: float


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class Answer(BaseModel):
    question: str
    text: str
    plan: QueryPlan
    contexts: list[ContextBlock]
    engine_used: str

    @property
    def retrieved_tickers(self) -> list[str]:
        return [c.parent.ticker for c in self.contexts]

    @property
    def generation_contexts(self) -> list[str]:
        return [c.parent.text for c in self.contexts]
