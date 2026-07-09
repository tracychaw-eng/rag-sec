"""Hybrid retriever: dense (Qdrant) + BM25 → RRF → class-aware rerank →
parent expansion. Fully async: multi-query embeddings and dense searches
run concurrently, and callers can fan out per-ticker retrievals in
parallel with asyncio.gather.

Retrieval modes encode the project's empirical findings as policy:
  precise:   rerank hard with a floor + min_keep  (factual/adversarial)
  reasoning: NO cross-encoder — it penalizes the indirect evidence that
             multi-hop inference needs (legacy R003: score 0.920, yet the
             reranked pipeline abstained)
  per_ticker: caller loops over tickers with a filter — the only
             guaranteed fix for source diversity (legacy lesson #4)
"""

import asyncio

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from ..cache import CacheBackend, TTLCache, cache_key
from ..config import Settings
from ..ingestion.store import ChunkStore
from ..models import ChildChunk, ContextBlock, ScoredChild
from ..observability import tracing
from .bm25 import BM25Index
from .fusion import rrf
from .rerank import CohereReranker


class HybridRetriever:
    def __init__(self, cfg: Settings, store: ChunkStore | None = None,
                 qdrant: AsyncQdrantClient | None = None,
                 openai_client: AsyncOpenAI | None = None,
                 reranker: CohereReranker | None = None,
                 embed_cache: CacheBackend | None = None):
        self.cfg = cfg
        self.store = store or ChunkStore(cfg.store_dir)
        self.qdrant = qdrant or AsyncQdrantClient(path=str(cfg.qdrant_path))
        self.openai = openai_client or AsyncOpenAI(api_key=cfg.openai_api_key)
        self.reranker = reranker or CohereReranker(
            api_key=cfg.cohere_api_key, model=cfg.rerank_model)
        # Query-embedding cache: multi-query retrieval re-embeds the same
        # rewrites often (eval reruns, paraphrase overlap, repeat questions)
        self.embed_cache = embed_cache if embed_cache is not None else TTLCache(
            max_items=cfg.cache_max_items, ttl_s=cfg.cache_ttl_s)

        children = self.store.load_children()
        self.children_by_id: dict[str, ChildChunk] = {c.id: c for c in children}
        self.parents_by_id = self.store.parents_by_id()
        self.bm25 = BM25Index(children)

    # ------------------------------------------------------------------
    async def _embed(self, text: str) -> list[float]:
        key = cache_key("embed", self.cfg.embed_model, text)
        cached = self.embed_cache.get(key)
        if cached is not None:
            return cached
        resp = await self.openai.embeddings.create(
            model=self.cfg.embed_model, input=[text])
        tracing.record_llm(self.cfg.embed_model, resp.usage, kind="embedding")
        vec = resp.data[0].embedding
        self.embed_cache.set(key, vec)
        return vec

    async def _dense_search(self, query: str, top_k: int,
                            tickers: list[str] | None) -> list[str]:
        """Returns child chunk ids in rank order."""
        flt = None
        if tickers:
            flt = qm.Filter(must=[qm.FieldCondition(
                key="ticker", match=qm.MatchAny(any=[t.upper() for t in tickers]))])
        hits = (await self.qdrant.query_points(
            collection_name=self.cfg.collection,
            query=await self._embed(query),
            limit=top_k,
            query_filter=flt,
            with_payload=["chunk_id"],
        )).points
        return [h.payload["chunk_id"] for h in hits]

    def _bm25_search(self, query: str, top_k: int,
                     tickers: list[str] | None) -> list[str]:
        return [c.id for c, _ in self.bm25.search(query, top_k, tickers)]

    # ------------------------------------------------------------------
    async def retrieve_children(self, queries: list[str],
                                tickers: list[str] | None = None
                                ) -> list[ScoredChild]:
        """Hybrid retrieval for one or more query phrasings (multi-query),
        fused with RRF. Dense searches for all phrasings run concurrently."""
        dense_lists = await asyncio.gather(*[
            self._dense_search(q, self.cfg.dense_top_k, tickers)
            for q in queries
        ])
        rankings = []
        for q, dense in zip(queries, dense_lists):
            rankings.append(dense)
            rankings.append(self._bm25_search(q, self.cfg.bm25_top_k, tickers))

        fused = rrf(rankings, k=self.cfg.rrf_k)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [
            ScoredChild(chunk=self.children_by_id[cid], rrf_score=score)
            for cid, score in ordered if cid in self.children_by_id
        ]

    def to_parents(self, scored: list[ScoredChild],
                   max_parents: int | None = None) -> list[ContextBlock]:
        """Child → parent expansion with dedupe, ordered by best child."""
        max_parents = max_parents or self.cfg.max_parents
        blocks: dict[str, ContextBlock] = {}
        for s in scored:
            pid = s.chunk.parent_id
            score = s.rerank_score if s.rerank_score is not None else s.rrf_score
            if pid not in blocks:
                if pid not in self.parents_by_id:
                    continue
                blocks[pid] = ContextBlock(
                    parent=self.parents_by_id[pid], best_child_score=score)
            elif score > blocks[pid].best_child_score:
                blocks[pid] = blocks[pid].model_copy(
                    update={"best_child_score": score})
        ordered = sorted(blocks.values(),
                         key=lambda b: b.best_child_score, reverse=True)
        return ordered[:max_parents]

    # ------------------------------------------------------------------
    async def retrieve(self, queries: list[str], mode: str = "precise",
                       tickers: list[str] | None = None) -> list[ContextBlock]:
        candidates = await self.retrieve_children(queries, tickers)

        if mode == "precise":
            kept = await self.reranker.rerank(
                queries[0], candidates[:self.cfg.rerank_candidates],
                top_n=self.cfg.rerank_top_n,
                score_floor=self.cfg.rerank_score_floor,
                min_keep=self.cfg.rerank_min_keep)
            return self.to_parents(kept)

        if mode == "reasoning":
            # No cross-encoder: keep RRF order, wider net
            return self.to_parents(
                candidates[:self.cfg.reasoning_top_children])

        if mode == "per_ticker":
            # Caller passes a single ticker in `tickers`
            kept = await self.reranker.rerank(
                queries[0], candidates[:self.cfg.rerank_candidates],
                top_n=self.cfg.per_ticker_top_n,
                score_floor=0.0)
            return self.to_parents(kept, max_parents=3)

        raise ValueError(f"Unknown retrieval mode: {mode}")
