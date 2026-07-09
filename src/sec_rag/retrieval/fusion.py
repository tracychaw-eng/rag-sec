"""Reciprocal Rank Fusion.

Rank-based fusion sidesteps the project's hardest-won lesson: scores from
different retrieval systems (cosine vs BM25 vs cross-encoder) are never
comparable. Ranks always are.
"""


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse multiple ranked lists of ids into {id: rrf_score}.

    rankings: each inner list is ids in rank order (best first).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
