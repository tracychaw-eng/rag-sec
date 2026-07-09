"""Ingestion pipeline: filing HTML → sections → parent/child chunks →
embeddings → Qdrant (children) + chunk store (parents & children).

Idempotent per ticker: re-running deletes that ticker's vectors first.

Usage:
    python -m sec_rag.ingestion.pipeline            # all filings in data/filings
    python -m sec_rag.ingestion.pipeline NVDA MSFT  # specific tickers
"""

import re
import sys
import uuid
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..config import Settings, get_settings
from ..models import ChildChunk
from .chunker import chunk_section
from .parser import parse_filing
from .store import ChunkStore

FILENAME_RE = re.compile(r"^([A-Z]+)_10K_(\d{4}-\d{2}-\d{2})\.html$")


def _point_id(chunk_id: str) -> str:
    """Qdrant needs UUID/int ids — derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _embed_batches(client: OpenAI, model: str, texts: list[str],
                   batch_size: int = 128) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(d.embedding for d in resp.data)
        print(f"    embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return vectors


def ensure_collection(qdrant: QdrantClient, cfg: Settings) -> None:
    if not qdrant.collection_exists(cfg.collection):
        qdrant.create_collection(
            collection_name=cfg.collection,
            vectors_config=qm.VectorParams(
                size=cfg.embed_dim, distance=qm.Distance.COSINE),
        )


def upsert_children(qdrant: QdrantClient, cfg: Settings,
                    children: list[ChildChunk],
                    vectors: list[list[float]]) -> None:
    points = [
        qm.PointStruct(
            id=_point_id(c.id),
            vector=v,
            payload={
                "chunk_id": c.id, "parent_id": c.parent_id,
                "ticker": c.ticker, "filing_date": c.filing_date,
                "item": c.item,
            },
        )
        for c, v in zip(children, vectors)
    ]
    for i in range(0, len(points), 256):
        qdrant.upsert(collection_name=cfg.collection, points=points[i:i + 256])


def ingest_filing(filepath: Path, cfg: Settings, store: ChunkStore,
                  qdrant: QdrantClient, openai_client: OpenAI) -> dict:
    m = FILENAME_RE.match(filepath.name)
    if not m:
        raise ValueError(f"Unexpected filing filename: {filepath.name}")
    ticker, filing_date = m.group(1), m.group(2)

    print(f"\n=== {ticker} ({filepath.name}) ===")
    sections = parse_filing(filepath, cfg.min_section_chars)
    print(f"  sections: {[s.item for s in sections]}")

    parents, children = [], []
    for section in sections:
        p, c = chunk_section(
            section, ticker, filing_date,
            parent_tokens=cfg.parent_tokens,
            child_tokens=cfg.child_tokens,
            child_overlap_tokens=cfg.child_overlap_tokens,
        )
        parents.extend(p)
        children.extend(c)
    print(f"  parents: {len(parents)}  children: {len(children)}")

    store.save(ticker, parents, children)

    # Idempotency: remove this ticker's old vectors before upserting
    qdrant.delete(
        collection_name=cfg.collection,
        points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker))
        ])),
    )
    vectors = _embed_batches(openai_client, cfg.embed_model,
                             [c.text for c in children])
    upsert_children(qdrant, cfg, children, vectors)
    print(f"  ✓ {len(children)} vectors upserted to {cfg.collection}")

    return {"ticker": ticker, "sections": len(sections),
            "parents": len(parents), "children": len(children)}


def main():
    cfg = get_settings()
    tickers = [t.upper() for t in sys.argv[1:]]

    files = sorted(cfg.filings_dir.glob("*_10K_*.html"))
    if tickers:
        files = [f for f in files if f.name.split("_")[0] in tickers]
    if not files:
        print(f"No filings found in {cfg.filings_dir}")
        sys.exit(1)

    store = ChunkStore(cfg.store_dir)
    qdrant = QdrantClient(path=str(cfg.qdrant_path))
    openai_client = OpenAI(api_key=cfg.openai_api_key)
    ensure_collection(qdrant, cfg)

    summary = [ingest_filing(fp, cfg, store, qdrant, openai_client)
               for fp in files]

    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"  {s['ticker']}: {s['sections']} sections, "
              f"{s['parents']} parents, {s['children']} children")


if __name__ == "__main__":
    main()
