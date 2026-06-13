"""
ingest_langchain.py — Parallel ingestion using LangChain-native conventions.

Reads the same SEC 10-K filings on disk that ingest.py (LlamaIndex) consumes,
chunks them with the same token budget (512 / 50), embeds with the same
model (text-embedding-3-small), and writes to a SEPARATE Pinecone index so
each framework can be benchmarked on its own native storage layout.

Usage:
    python ingest_langchain.py NVDA MSFT JPM
    python ingest_langchain.py --all          # ingest every file in data/filings/

Env vars:
    PINECONE_API_KEY              required
    PINECONE_INDEX_LC             optional, defaults to "sec-filings-lc"
    OPENAI_API_KEY                required
"""

import os
import sys
import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from ingest import find_existing_filing, parse_filing_html

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
INDEX_NAME = os.environ.get("PINECONE_INDEX_LC", "sec-filings-lc")


def ensure_index(pc: Pinecone, name: str) -> None:
    existing = {i["name"] for i in pc.list_indexes()}
    if name in existing:
        print(f"  ✓ Pinecone index '{name}' already exists")
        return
    print(f"  🛠 Creating Pinecone index '{name}' (dim={EMBED_DIM}, cosine, serverless)...")
    pc.create_index(
        name=name,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(name).status.get("ready", False):
        time.sleep(1)
    print(f"  ✓ Index '{name}' ready")


def chunk_filing(text: str, ticker: str, filing_date: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    metadata = {
        "ticker": ticker.upper(),
        "doc_type": "10-K",
        "source": f"{ticker.upper()}_10K",
        "filing_date": filing_date,
        "file_name": f"{ticker.upper()}_10K_{filing_date}",
    }
    return [Document(page_content=c, metadata=metadata) for c in chunks]


def ingest_ticker(ticker: str, vectorstore: PineconeVectorStore, pc_index) -> None:
    ticker = ticker.upper()
    print(f"\n{'='*60}\n🚀 Ingesting (LangChain): {ticker}\n{'='*60}")

    filepath = find_existing_filing(ticker)
    if filepath is None:
        print(f"  ❌ No filing found on disk for {ticker}. Run ingest.py first to download.")
        return

    text = parse_filing_html(filepath)
    filing_date = filepath.stem.split("_")[-1]
    docs = chunk_filing(text, ticker, filing_date)
    print(f"  📦 Created {len(docs)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, tiktoken-counted)")

    print(f"  🗑 Deleting old vectors for {ticker}...")
    try:
        pc_index.delete(filter={"ticker": {"$eq": ticker}})
    except Exception as e:
        print(f"  ⚠ Could not delete old vectors: {e}")

    vectorstore.add_documents(docs)
    print(f"  ✅ Upserted {len(docs)} vectors to '{INDEX_NAME}'")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python ingest_langchain.py TICKER [TICKER ...]   |   --all")
        sys.exit(1)

    if "--all" in args:
        from pathlib import Path
        from edgar_downloader import DATA_DIR
        tickers = sorted({p.name.split("_")[0] for p in Path(DATA_DIR).glob("*_10K_*")})
        print(f"Found {len(tickers)} tickers on disk: {tickers}")
    else:
        tickers = [a for a in args if not a.startswith("--")]

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_index(pc, INDEX_NAME)
    pc_index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = PineconeVectorStore(index=pc_index, embedding=embeddings)

    for ticker in tickers:
        try:
            ingest_ticker(ticker, vectorstore, pc_index)
        except Exception as e:
            print(f"\n❌ Failed to ingest {ticker}: {e}")

    print(f"\n{'='*60}\nDone. LangChain index '{INDEX_NAME}' is ready.\n{'='*60}")


if __name__ == "__main__":
    main()
