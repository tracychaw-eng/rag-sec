"""Cross-encoder reranking via Cohere, behind a small interface so a
self-hosted BGE reranker can slot in later without touching callers."""

import time

import cohere

from ..models import ScoredChild


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0",
                 max_retries: int = 3):
        self._client = cohere.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def rerank(self, query: str, candidates: list[ScoredChild],
               top_n: int, score_floor: float = 0.0,
               min_keep: int = 0) -> list[ScoredChild]:
        """Keep top_n by cross-encoder score. The floor trims tail noise but
        never cuts below min_keep results — absolute reranker scores vary
        too much by query phrasing to be trusted as a hard gate (measured:
        a correct #1-ranked chunk scored 0.122 on one factual query)."""
        if not candidates:
            return []
        docs = [c.chunk.text for c in candidates]

        for attempt in range(self.max_retries):
            try:
                resp = self._client.rerank(
                    model=self.model, query=query, documents=docs, top_n=top_n)
                break
            except cohere.errors.TooManyRequestsError:
                wait = 10 * (attempt + 1)
                print(f"    cohere rate limit — waiting {wait}s")
                time.sleep(wait)
        else:
            # Reranker unavailable — degrade gracefully to fusion order
            print("    cohere unavailable — falling back to RRF order")
            return candidates[:top_n]

        out = []
        for r in resp.results:              # results arrive best-first
            if len(out) >= min_keep and r.relevance_score < score_floor:
                continue
            c = candidates[r.index].model_copy(
                update={"rerank_score": r.relevance_score})
            out.append(c)
        return out
