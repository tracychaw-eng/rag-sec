"""Migrate vectors from embedded Qdrant (local mode) to a Qdrant server
without re-embedding — the same zero-cost pattern as the old
chroma→pinecone migration, for the local→server scale-out step.

Usage:
    python -m sec_rag.ingestion.migrate_qdrant --to http://localhost:6333
"""

import argparse

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..config import get_settings


def migrate(to_url: str, batch: int = 256) -> int:
    cfg = get_settings()
    src = QdrantClient(path=str(cfg.qdrant_path))
    dst = QdrantClient(url=to_url)

    if not dst.collection_exists(cfg.collection):
        dst.create_collection(
            collection_name=cfg.collection,
            vectors_config=qm.VectorParams(
                size=cfg.embed_dim, distance=qm.Distance.COSINE),
        )

    total = 0
    offset = None
    while True:
        points, offset = src.scroll(
            collection_name=cfg.collection, limit=batch,
            with_vectors=True, with_payload=True, offset=offset)
        if not points:
            break
        dst.upsert(collection_name=cfg.collection, points=[
            qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ])
        total += len(points)
        print(f"  migrated {total} vectors")
        if offset is None:
            break
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True, help="Qdrant server URL")
    args = ap.parse_args()
    n = migrate(args.to)
    print(f"Done: {n} vectors migrated to {args.to}")


if __name__ == "__main__":
    main()
