"""Cross-encoder reranking via Cohere (async), behind a small interface so
a self-hosted BGE reranker can slot in later without touching callers.

Retries: tenacity with exponential backoff + jitter on rate limits and
transient server errors. If retries are exhausted, callers get the
candidates back in fusion order — a degraded answer beats a failed request.
"""

import cohere
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from ..models import ScoredChild
from ..observability import tracing

_RETRYABLE = (
    cohere.TooManyRequestsError,
    cohere.ServiceUnavailableError,
    cohere.InternalServerError,
)


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self._client = cohere.AsyncClient(api_key=api_key)
        self.model = model

    @retry(retry=retry_if_exception_type(_RETRYABLE),
           wait=wait_random_exponential(multiplier=2, max=30),
           stop=stop_after_attempt(4), reraise=True)
    async def _call(self, query: str, docs: list[str], top_n: int):
        return await self._client.rerank(
            model=self.model, query=query, documents=docs, top_n=top_n)

    async def rerank(self, query: str, candidates: list[ScoredChild],
                     top_n: int, score_floor: float = 0.0,
                     min_keep: int = 0) -> list[ScoredChild]:
        """Keep top_n by cross-encoder score. The floor trims tail noise but
        never cuts below min_keep results — absolute reranker scores vary
        too much by query phrasing to be trusted as a hard gate (measured:
        a correct #1-ranked chunk scored 0.122 on one factual query)."""
        if not candidates:
            return []
        try:
            resp = await self._call(
                query, [c.chunk.text for c in candidates], top_n)
            tracing.record_rerank(1)
        except Exception as e:
            print(f"    reranker unavailable ({type(e).__name__}) — "
                  f"falling back to RRF order")
            return candidates[:top_n]

        out = []
        for r in resp.results:              # results arrive best-first
            if len(out) >= min_keep and r.relevance_score < score_floor:
                continue
            c = candidates[r.index].model_copy(
                update={"rerank_score": r.relevance_score})
            out.append(c)
        return out
