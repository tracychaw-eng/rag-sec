"""EDGAR sync — fetch the latest N 10-Ks per ticker and ingest anything
not already in the corpus. Safe to run on a schedule: downloads skip
files on disk, ingestion skips filings already in the chunk store
(--force re-ingests).

Usage:
    python -m sec_rag.ingestion.sync MSFT NVDA JPM            # latest 1 each
    python -m sec_rag.ingestion.sync MSFT NVDA JPM --years 2  # latest 2 each
"""

import argparse

from openai import OpenAI
from qdrant_client import QdrantClient

from ..config import get_settings
from . import edgar
from .pipeline import ensure_collection, ingest_filing
from .store import ChunkStore

DEFAULT_TICKERS = ["MSFT", "NVDA", "JPM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=None)
    ap.add_argument("--years", type=int, default=1,
                    help="latest N 10-Ks per ticker")
    ap.add_argument("--force", action="store_true",
                    help="re-ingest filings already in the store")
    args = ap.parse_args()
    tickers = [t.upper() for t in (args.tickers or DEFAULT_TICKERS)]

    cfg = get_settings()
    store = ChunkStore(cfg.store_dir)
    qdrant = (QdrantClient(url=cfg.qdrant_url) if cfg.qdrant_url
              else QdrantClient(path=str(cfg.qdrant_path)))
    openai_client = OpenAI(api_key=cfg.openai_api_key)
    ensure_collection(qdrant, cfg)

    ingested, skipped = [], []
    for ticker in tickers:
        for meta in edgar.list_10k_filings(ticker, count=args.years):
            if store.has_filing(ticker, meta["date"]) and not args.force:
                skipped.append(f"{ticker} {meta['date']}")
                continue
            path = edgar.download_filing(meta, cfg.filings_dir)
            ingest_filing(path, cfg, store, qdrant, openai_client)
            ingested.append(f"{ticker} {meta['date']}")

    print(f"\nSYNC COMPLETE — ingested: {ingested or 'nothing new'}"
          f" | already present: {len(skipped)}")


if __name__ == "__main__":
    main()
