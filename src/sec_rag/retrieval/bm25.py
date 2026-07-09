"""In-process BM25 over child chunks.

At the current corpus size (a few thousand children) an in-process index
builds in well under a second and gives us full control over tokenization
and filtering. When the corpus outgrows memory, this module's interface
is the seam where a server-side sparse index (Qdrant sparse vectors,
OpenSearch) slots in.
"""

import re

from rank_bm25 import BM25Okapi

from ..models import ChildChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that "
    "the to was were will with we our us this these those such may could "
    "would should".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, children: list[ChildChunk]):
        self.children = children
        self._bm25 = BM25Okapi([tokenize(c.text) for c in children])

    def search(self, query: str, top_k: int = 50,
               tickers: list[str] | None = None,
               years: list[int] | None = None
               ) -> list[tuple[ChildChunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        allowed = set(t.upper() for t in tickers) if tickers else None
        year_set = set(years) if years else None
        ranked = sorted(
            (
                (c, float(s)) for c, s in zip(self.children, scores)
                if s > 0
                and (allowed is None or c.ticker in allowed)
                and (year_set is None or int(c.filing_date[:4]) in year_set)
            ),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]
