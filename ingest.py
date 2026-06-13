"""
ingest.py — Auto-ingest SEC 10-K filings into the RAG pipeline.
Connects edgar_downloader.py → HTML parsing → LlamaIndex indexing → Pinecone.

Usage:
    python ingest.py NVDA                # ingest single ticker
    python ingest.py MSFT NVDA JPM       # ingest multiple
    python ingest.py NVDA --force        # re-download even if file exists
"""

import sys
import glob
from pathlib import Path
from bs4 import BeautifulSoup

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from edgar_downloader import download_10k, DATA_DIR

# ---------------------------------------------------------------------------
# CONFIG — adjust these to match your existing pipeline
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Set True if using Pinecone, False for ChromaDB
USE_PINECONE = True


# ---------------------------------------------------------------------------
# STEP 1: HTML → clean text
# ---------------------------------------------------------------------------
def parse_filing_html(filepath: Path) -> str:
    """Extract clean text from a 10-K HTML filing."""
    print(f"  🔍 Parsing HTML: {filepath.name}")
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "meta", "link", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 3:  # skip tiny noise lines
            lines.append(stripped)

    clean = "\n".join(lines)
    print(f"  ✓ Extracted {len(clean):,} chars from {filepath.name}")
    return clean


# ---------------------------------------------------------------------------
# STEP 2: Find existing filing on disk (skip re-download)
# ---------------------------------------------------------------------------
def find_existing_filing(ticker: str) -> Path | None:
    """Check if a 10-K for this ticker already exists in data/filings/."""
    pattern = str(DATA_DIR / f"{ticker.upper()}_10K_*")
    matches = glob.glob(pattern)
    if matches:
        # Return the most recent one (sorted by date in filename)
        return Path(sorted(matches)[-1])
    return None


# ---------------------------------------------------------------------------
# STEP 3: Build LlamaIndex documents with metadata
# ---------------------------------------------------------------------------
def create_documents(ticker: str, text: str, filing_date: str) -> list[Document]:
    """Create LlamaIndex Document with proper metadata for per-source routing."""
    doc = Document(
        text=text,
        metadata={
            "ticker": ticker.upper(),
            "doc_type": "10-K",
            "source": f"{ticker.upper()}_10K",
            "filing_date": filing_date,
            "file_name": f"{ticker.upper()}_10K_{filing_date}",  # display compat with main_pinecone.py
        },
        excluded_llm_metadata_keys=["doc_type", "file_name"],
        # Exclude ALL metadata from embeddings so vectors represent pure content,
        # matching the behavior of the original SimpleDirectoryReader-ingested vectors.
        excluded_embed_metadata_keys=["ticker", "doc_type", "source", "filing_date", "file_name"],
    )
    return [doc]


# ---------------------------------------------------------------------------
# STEP 4: Index into vector store
# ---------------------------------------------------------------------------
def index_documents(documents: list[Document], ticker: str):
    """Chunk, embed, and upsert documents into the vector store."""

    # Node parser (chunking)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  📦 Created {len(nodes)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if USE_PINECONE:
        _index_pinecone(nodes, ticker)
    else:
        _index_chromadb(nodes, ticker)


# Replace _index_pinecone in ingest.py with this:

def _index_pinecone(nodes, ticker: str):
    from pinecone import Pinecone
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core import StorageContext
    import os

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX", "sec-filings")
    pc_index = pc.Index(index_name)

    print(f"  🗑 Deleting old vectors for {ticker}...")
    try:
        pc_index.delete(filter={"ticker": {"$eq": ticker.upper()}})
    except Exception as e:
        print(f"  ⚠ Could not delete old vectors: {e}")

    vector_store = PineconeVectorStore(pinecone_index=pc_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    print(f"  ✅ Upserted {len(nodes)} vectors to Pinecone ({index_name})")


def _index_chromadb(nodes, ticker: str):
    """Upsert nodes into ChromaDB. Deletes old docs for this ticker first."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="sec_filings",
        metadata={"hnsw:space": "cosine"},
    )

    # Delete old docs for this ticker
    print(f"  🗑 Deleting old docs for {ticker}...")
    try:
        existing = collection.get(where={"ticker": {"$eq": ticker.upper()}})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"     Removed {len(existing['ids'])} old chunks")
    except Exception as e:
        print(f"  ⚠ Could not delete old docs: {e}")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    print(f"  ✅ Indexed {len(nodes)} vectors in ChromaDB")


# ---------------------------------------------------------------------------
# MAIN: Full ingest pipeline
# ---------------------------------------------------------------------------
def ingest_new_filing(ticker: str, force: bool = False) -> None:
    """End-to-end: download (if needed) → parse → chunk → embed → index."""
    ticker = ticker.upper()
    print(f"\n{'='*60}")
    print(f"🚀 Ingesting: {ticker}")
    print(f"{'='*60}")

    # Step 1: Download or find existing
    existing = find_existing_filing(ticker)
    if existing and not force:
        print(f"  📂 Found existing file: {existing.name} (use --force to re-download)")
        filepath = existing
    else:
        filepath = download_10k(ticker)

    # Step 2: Parse HTML → text
    clean_text = parse_filing_html(filepath)

    # Extract filing date from filename (e.g., NVDA_10K_2026-02-25.html)
    filing_date = filepath.stem.split("_")[-1]  # "2026-02-25"

    # Step 3: Create documents with metadata
    documents = create_documents(ticker, clean_text, filing_date)

    # Step 4: Index
    index_documents(documents, ticker)

    print(f"\n✅ {ticker} 10-K fully ingested and queryable!")


def main():
    force = "--force" in sys.argv
    tickers = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if not tickers:
        print("Usage: python ingest.py TICKER [TICKER ...] [--force]")
        print("Example: python ingest.py NVDA MSFT JPM")
        sys.exit(1)

    # Configure LlamaIndex defaults
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")

    for ticker in tickers:
        try:
            ingest_new_filing(ticker, force=force)
        except Exception as e:
            print(f"\n❌ Failed to ingest {ticker}: {e}")

    print(f"\n{'='*60}")
    print("Done! You can now query these filings via your RAG pipeline.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()