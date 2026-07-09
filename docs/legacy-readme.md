# LlamaIndex RAG Practice — SEC 10-K Filing Analysis

A hands-on project for Retrieval-Augmented Generation (RAG) using LlamaIndex, OpenAI embeddings, ChromaDB, and real SEC 10-K filings.

**Sessions covered:**

- **Mar 23** — LlamaIndex quickstart, in-memory RAG pipeline, chunk configuration
- **Mar 24** — ChromaDB persistent vector store, distance metrics, production patterns
- **Mar 26** — OpenAI embedding models, model comparison, cost logging, similarity experiments
- **Mar 28 AM** — Pinecone cloud vector store, migration from ChromaDB, local vs cloud tradeoffs
- **Mar 28 PM** — Evaluation dataset (20 Q&A pairs), 4-round debugging, reranking, per-source diversity, per-category engine routing
- **Mar 29–31** — HyDE, SubQuestionQueryEngine, cross-document hallucination fix (R005), RAGAS integration with full metrics
- **Apr 6** — SEC EDGAR API downloader, programmatic 10-K fetching with rate limiting
- **Apr 7 AM** — Auto-ingest pipeline connecting EDGAR downloader to Pinecone via `ingest.py`
- **Apr 7 PM** — Multi-turn conversation memory (`chat_engine.py`), Streamlit chat UI (`app.py`)
- **Apr 28** — LlamaIndex vs LangChain head-to-head: parallel native ingest (`ingest_langchain.py`), optimization parity (`compare_frameworks.py`), `text_key` mismatch debugging

---

## Project Structure

```
llamaindex-sec/
├── data/
│   ├── filings/                     # Raw 10-K HTML files downloaded by edgar_downloader.py
│   │   └── NVDA_10K_2026-02-25.html
│   ├── jpm_10k.txt                  # JPMorgan Chase 10-K Risk Factors (~114KB) — legacy
│   ├── msft_10k.txt                 # Microsoft 10-K Risk Factors (~70KB) — legacy
│   └── nvda_10k.txt                 # Nvidia 10-K Risk Factors (~115KB) — legacy
├── chroma_db/                       # ChromaDB persistent store (local)
│   └── chroma.sqlite3               # Contains sec_filings (ada-002) and sec_filings_v2 (3-small)
├── edgar_downloader.py              # Fetches 10-K filings from SEC EDGAR API
├── ingest.py                        # End-to-end LlamaIndex ingestion → sec-filings (Pinecone)
├── ingest_langchain.py              # Parallel LangChain-native ingestion → sec-filings-lc (Apr 28)
├── compare_frameworks.py            # LlamaIndex vs LangChain head-to-head (Apr 28)
├── framework_comparison.json        # Generated comparison results (rerank scores, sources, answers)
├── chat_engine.py                   # Multi-turn conversational RAG with Memory (Apr 7)
├── app.py                           # Streamlit chat UI for interactive filing Q&A (Apr 7)
├── main.py                          # Production query runner: Pinecone + reranker + soft abstention
├── main_pinecone.py                 # Simple query runner: Pinecone baseline (Mar 28 AM)
├── cleanup_old_vectors.py           # Remove legacy text-file vectors from Pinecone
├── evaluate.py                      # Full evaluation harness: 4 engines, RAGAS, HyDE, SubQuestion
├── query_with_diversity.py          # Per-source diversity for comparison questions
├── check_metadata.py                # Verify Pinecone metadata keys
├── inspect_chroma.py                # Inspect ChromaDB contents directly
├── inspect_pinecone.py              # Inspect Pinecone index directly
├── migrate_chroma_to_pinecone.py    # Zero-cost vector migration utility
├── explore_embeddings.py            # Embedding similarity experiments (Mar 26)
├── cost_logger.py                   # Cost tracking utility (Mar 26)
├── embedding_costs.json             # Generated cost log
├── evaluation_dataset.json          # 20 Q&A pairs across 4 categories
├── eval_results.json                # Generated evaluation results (retrieval + sources)
├── ragas_results.json               # Generated RAGAS scores (faithfulness, relevancy, recall, precision)
├── diversity_fix_results.json       # Per-source diversity fix results
├── comparison_log.md                # Manual model comparison notes
└── venv/
```

---

## Architecture

### Data Pipeline — run once per ticker to build the index

```text
edgar_downloader.py              ingest.py
───────────────────              ─────────
Fetches raw filings    ───────▶  Orchestrates end-to-end ingestion
from SEC EDGAR API               │
│                                ├─ calls edgar_downloader.download_10k()
│  ticker → CIK lookup           ├─ parses HTML → clean text (BeautifulSoup)
│  CIK → filing metadata         ├─ wraps text in LlamaIndex Document
│  accession → HTML file         │    metadata: ticker, source,
│  saves to data/filings/        │    filing_date, file_name
└─ returns Path                  └─ chunks → embeds → upserts to Pinecone
                                     (deletes old ticker vectors first)

Usage: python ingest.py NVDA MSFT JPM
```

### Query Pipeline — run anytime to query

```text
main_pinecone.py                 main.py
────────────────                 ───────
Simple query runner    vs.       Production query runner
│                                │
│  loads index from              │  loads index from
│  Pinecone (no re-embed)        │  Pinecone (no re-embed)
│                                │
│  similarity_top_k=3            │  similarity_top_k=20
│  no reranker                   │  + Cohere cross-encoder reranker
│  no custom prompt              │    → keeps top 5 after reranking
│                                │  + custom QA prompt (anti-hallucination)
│  runs 5 risk questions         │  runs same 5 risk questions
└─ prints scores/sources         └─ prints scores/sources
```

### Chat Pipeline — multi-turn conversational RAG (Apr 7)

```text
chat_engine.py                   app.py
──────────────                   ──────
CLI chat interface    vs.        Streamlit web UI
│                                │
│  both use:                     │  same engine, wrapped in
│  ┌──────────────────────┐      │  st.chat_message() UI
│  │ condense_plus_context│      │  + @st.cache_resource
│  │                      │      │    for session persistence
│  │ User: "How does      │      │
│  │  that compare to     │      │
│  │  Microsoft?"         │      │
│  │       │              │      │
│  │  Memory + history    │      │
│  │       │              │      │
│  │  Condensed query:    │      │
│  │  "How do NVIDIA's    │      │
│  │   risk factors       │      │
│  │   compare to         │      │
│  │   Microsoft's?"      │      │
│  │       │              │      │
│  │  Vector retrieval    │      │
│  │  (similarity_top_k=6)│      │
│  │       │              │      │
│  │  LLM generates       │      │
│  │  grounded answer     │      │
│  └──────────────────────┘      │
│                                │
│  Memory class (new API)        │  Session state for
│  token_limit=3000              │  chat history display
└─ type 'reset' to clear         └─ streamlit run app.py
```

**Why `condense_plus_context`?** Follow-up questions like "tell me more about that" are meaningless to a vector retriever. The condense step rewrites them into standalone queries using chat history, so retrieval actually works. Without it, embedding the word "that" returns garbage.

### Pinecone Index — "sec-filings" (1536-dim, cosine)

```text
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   vectors from ingest.py            vectors from main_pinecone  │
│   (current, structured)             (legacy, use cleanup script)│
│   ┌───────────────────────┐         ┌────────────────────────┐  │
│   │ ticker:      NVDA     │         │ file_name: nvda_10k.txt│  │
│   │ file_name:            │         │ file_name: msft_10k.txt│  │
│   │  NVDA_10K_2026-02-25  │         │ file_name: jpm_10k.txt │  │
│   │ source:   NVDA_10K    │         │ file_path: C:\...      │  │
│   │ filing_date: ...      │         │ file_type: text/plain  │  │
│   └───────────────────────┘         └────────────────────────┘  │
│                                                                 │
│   All metadata excluded from embeddings → pure content vectors  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup

```powershell
# Windows (PowerShell)
mkdir C:\AgentWorkspace\llamaindex-sec
cd C:\AgentWorkspace\llamaindex-sec
python -m venv venv
venv\Scripts\activate
pip install llama-index openai
pip install chromadb
pip install llama-index-vector-stores-chroma       # LlamaIndex ChromaDB integration
pip install llama-index-embeddings-openai          # Required for Settings.embed_model
pip install pinecone                               # Mar 28 AM
pip install llama-index-vector-stores-pinecone     # LlamaIndex Pinecone integration
pip install llama-index-postprocessor-cohere-rerank  # Reranker — Mar 28 PM
pip install beautifulsoup4                           # HTML parsing for ingest.py — Apr 6
pip install streamlit                                # Chat UI — Apr 7
pip install langchain langchain-classic langchain-openai langchain-pinecone \
            langchain-text-splitters langchain-cohere   # Framework comparison — Apr 28
```

Set your API keys — never hardcode them:

```powershell
setx OPENAI_API_KEY "sk-..."
setx PINECONE_API_KEY "pcsk_..."     # from app.pinecone.io → API Keys
setx COHERE_API_KEY "..."            # from dashboard.cohere.com → API Keys (free tier)
```

> After `setx`, always close and reopen PowerShell before running scripts.
> Verify with: `echo $env:COHERE_API_KEY`
> Or set inline for current session: `$env:COHERE_API_KEY = "..."`

---

## Data Collection

Downloaded the **Item 1A. Risk Factors** section from each company's most recent 10-K filing via SEC EDGAR:

- **Microsoft (MSFT):** <https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=MSFT&type=10-K>
- **Nvidia (NVDA):** <https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=NVDA&type=10-K>
- **JPMorgan (JPM):** <https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=JPM&type=10-K>

> Tip: Copy only Item 1A (Risk Factors to Item 1B) to keep files focused. Each section is 5,000–15,000 words, producing 20–50 chunks per file.

---

## Code

### v1 — In-Memory Pipeline (Mar 23)

Simple baseline. Fast to set up, but re-embeds everything on every run.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What are the main risk factors?")
print(response)

for i, node in enumerate(response.source_nodes):
    print(f"\nChunk {i+1}")
    print(f"Score:  {node.score:.3f}")
    print(f"Source: {node.metadata.get('file_name', 'unknown')}")
    print(f"Text preview:\n{node.text[:300]}")
```

---

### v2 — ChromaDB Persistent Pipeline (Mar 24)

Production-ready pattern. Embeds once, reloads from disk on every subsequent run.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# Connect to persistent ChromaDB on disk
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# CRITICAL: always set hnsw:space explicitly — ChromaDB defaults to L2,
# but LlamaIndex expects cosine. Mismatching these drops scores to ~0.65-0.69
chroma_collection = chroma_client.get_or_create_collection(
    "sec_filings",
    metadata={"hnsw:space": "cosine"}
)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() == 0:
    # First run — load, chunk, embed, store
    print("No data found. Embedding documents...")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"{chroma_collection.count()} chunks stored in ChromaDB.")
else:
    # Subsequent runs — load from disk, no API calls
    print(f"{chroma_collection.count()} chunks found. Loading from disk...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What are the main risk factors?")
print(response)

for i, node in enumerate(response.source_nodes):
    print(f"\nChunk {i+1}")
    print(f"Score:  {node.score:.3f}")
    print(f"Source: {node.metadata.get('file_name', 'unknown')}")
    print(f"Text preview:\n{node.text[:300]}")
```

---

### Inspecting ChromaDB Directly

```python
# inspect_chroma.py — run separately to poke at raw DB contents
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
print(client.list_collections())

collection = client.get_collection("sec_filings")
print(f"Total chunks: {collection.count()}")

results = collection.peek(limit=3)
for i, doc in enumerate(results['documents']):
    print(f"\nChunk {i+1}")
    print(f"ID:       {results['ids'][i]}")
    print(f"Metadata: {results['metadatas'][i]}")
    print(f"Text:     {doc[:200]}")
```

---

### v3 — Explicit Embedding Model + Cost Logging (Mar 26)

Adds explicit embedding model selection, cost tracking, and 5-question comparison.

```python
import os, time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from cost_logger import log_embedding_run, estimate_tokens

# Explicitly set embedding model — never rely on LlamaIndex defaults
# Default if unset: text-embedding-ada-002 (legacy, 5x more expensive)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# CRITICAL: one collection per embedding model — never mix models in one collection
chroma_collection = chroma_client.get_or_create_collection(
    "sec_filings_v2",                   # v2 = text-embedding-3-small
    metadata={"hnsw:space": "cosine"}
)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader("data").load_data()
    all_text = " ".join([doc.text for doc in documents])
    estimated_tokens = estimate_tokens(all_text)
    estimated_cost = estimated_tokens * (0.020 / 1_000_000)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    log_embedding_run("text-embedding-3-small", chroma_collection.count(),
                      estimated_tokens, estimated_cost)
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

query_engine = index.as_query_engine(similarity_top_k=3)
questions = [
    "What are the main risk factors?",
    "What cybersecurity risks does the company face?",
    "How does competition affect the business?",
    "What regulatory risks are mentioned?",
    "What macroeconomic risks could impact performance?"
]

results = []
for i, question in enumerate(questions, 1):
    response = query_engine.query(question)
    results.append({
        "question": question,
        "answer": str(response),
        "top_score": response.source_nodes[0].score if response.source_nodes else 0,
        "sources": [n.metadata.get('file_name', 'unknown') for n in response.source_nodes]
    })

# Summary table — always outside the query loop
for i, r in enumerate(results, 1):
    sources = ", ".join(set(r["sources"]))
    print(f"Q{i:<3} {r['top_score']:<12.3f} {sources}")
```

---

### Embedding Similarity Experiments

```python
# explore_embeddings.py
import os, math
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

def label_score(score, model="3-small"):
    # IMPORTANT: thresholds differ per model — never compare scores across models
    if model == "ada-002":
        return "SIMILAR ✅" if score > 0.85 else ("RELATED" if score > 0.75 else "UNRELATED ❌")
    else:  # 3-small uses a wider dynamic range
        return "SIMILAR ✅" if score > 0.45 else ("RELATED" if score > 0.20 else "UNRELATED ❌")

pairs = [
    ("What are the main risk factors?",       "What risks does the company face?"),
    ("What cybersecurity risks are mentioned?","How is the company exposed to hacking?"),
    ("What are the main risk factors?",        "What was the company's revenue last year?"),
    ("Cybersecurity threats to the business",  "The history of Renaissance painting"),
]

for a, b in pairs:
    score = cosine_similarity(get_embedding(a), get_embedding(b))
    print(f"Score: {score:.3f}  [{label_score(score)}]")
    print(f"  A: {a}")
    print(f"  B: {b}\n")
```

---

## How RAG Works — Plain English

| Step | What Happens |
|---|---|
| **1. Ingestion** | Each .txt file is read and split into overlapping chunks (~512 tokens each) |
| **2. Embedding** | Every chunk is converted to a vector (~1,500 numbers) by OpenAI's embedding model |
| **3. Indexing** | All vectors are stored in a searchable in-memory vector store |
| **4. Retrieval** | Your query is also embedded; top-k chunks with closest vectors are fetched |
| **5. Generation** | Retrieved chunks + your query are sent to GPT, which writes a grounded answer |

> **Key insight:** The LLM never memorizes your documents. It only sees the relevant chunks at query time. This is why RAG is preferred over fine-tuning for document Q&A — it's cheaper, faster to update, and cites sources.

---

## Chunk Configuration

```python
SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

| Parameter | Effect |
|---|---|
| `chunk_size` | Controls tokens per chunk. Smaller = precise retrieval, less context. Larger = more context, more noise. |
| `chunk_overlap` | Repeated tokens between adjacent chunks. Prevents losing context at boundaries. |

### Chunk Size Tradeoffs

| chunk_size | Approx chunks (this project) | Use case |
|---|---|---|
| 128 | ~500 | Precise fact lookup |
| 256 | ~250 | Short Q&A |
| **512** | **~128** | **General purpose (default)** |
| 1024 | ~65 | Complex reasoning, full risk items |

### Other Splitter Options

```python
from llama_index.core.node_parser import (
    SentenceSplitter,    # Respects sentence boundaries — general purpose
    TokenTextSplitter,   # Splits by token count only — fast
    SemanticSplitter,    # Groups by meaning — best quality, costs more
    MarkdownNodeParser,  # Splits on markdown headers
)
```

---

## Retrieval Scores

Scores are **cosine similarity** between the query vector and each chunk vector.

| Score | Meaning |
|---|---|
| 0.85+ | Very strong match |
| 0.75–0.85 | Good match |
| 0.60–0.75 | Weak match |
| below 0.60 | Likely noise |

Scores in this project: `0.813`, `0.808`, `0.781` — all solid retrieval.

**Why cosine similarity?** It measures the angle between vectors, ignoring magnitude. A short and long chunk about the same topic score similarly even if their vector lengths differ — Euclidean distance would wrongly penalize the length difference.

---

## OpenAI Embedding Models (Mar 26)

### Model Comparison

| Model | Dimensions | Cost per 1M tokens | Typical score range |
|---|---|---|---|
| `text-embedding-ada-002` | 1,536 | $0.100 | 0.83 – 0.87 |
| **`text-embedding-3-small`** | 1,536 | **$0.020** | 0.54 – 0.87 |
| `text-embedding-3-large` | 3,072 | $0.130 | — |

> Always set the model explicitly — LlamaIndex silently defaults to `ada-002` if `Settings.embed_model` is not set. Check with `print(Settings.embed_model)`.

### Actual Results on SEC 10-K Filings

| Q# | Question | ada-002 score | 3-small score |
|---|---|---|---|
| Q1 | Main risk factors | 0.830 | 0.539 |
| Q2 | Cybersecurity risks | 0.867 | 0.720 |
| Q3 | Competition | 0.838 | 0.627 |
| Q4 | Regulatory risks | 0.850 | 0.644 |
| Q5 | Macroeconomic risks | 0.871 | 0.649 |

### Why ada-002 Scores Higher But Is Not Better

ada-002 compresses all vectors into a narrow high range (0.75–0.95). 3-small uses a wider dynamic range including near-zero and negative values. This gives 3-small **more separation** between relevant and irrelevant results — which is what actually matters for retrieval quality. Answer content from 3-small was comparable or more detailed on Q2 and Q5 despite lower absolute scores.

**The rule: never compare absolute cosine scores across different embedding models. Only compare within the same model.**

### Indexing Cost Log (this project)

| Model | Chunks | Tokens | Cost |
|---|---|---|---|
| text-embedding-3-small | 128 | 74,348 | $0.001487 |
| text-embedding-ada-002 | 128 | 74,348 | ~$0.007435 (5x more) |

### The Critical Rule — One Collection Per Model

Different embedding models produce vectors in completely different coordinate spaces. Mixing them produces random nonsense similarity scores with no error message.

```python
# CORRECT — separate collection per model
chroma_client.get_or_create_collection("sec_filings")      # ada-002
chroma_client.get_or_create_collection("sec_filings_v2")   # 3-small

# WRONG — switching models without deleting collection
# Scores will silently degrade with no warning
```

If you switch models and forget to create a new collection:

```powershell
Remove-Item -Recurse -Force .\chroma_db
python main.py   # rebuild from scratch
```

### What an Embedding Actually Is

An embedding is a list of numbers (a vector) representing the meaning of text. For `text-embedding-3-small`, every piece of text maps to a list of 1,536 floats:

```
"cybersecurity risk" → [-0.0015, -0.0031, 0.1078, 0.0565, -0.0192, ...]
                        ← 1,536 numbers total →
```

The model was trained to place semantically similar text at nearby points in this 1,536-dimensional space. Two sentences that mean the same thing — even if they share no words — will have vectors that point in nearly the same direction (high cosine similarity). Two unrelated sentences will point in completely different directions (near-zero or negative cosine similarity).

### Similarity Score Calibration Per Model

```python
# ada-002: compressed range, high baseline
"SIMILAR" if score > 0.85 else "RELATED" if score > 0.75 else "UNRELATED"

# text-embedding-3-small: wider dynamic range
"SIMILAR" if score > 0.45 else "RELATED" if score > 0.20 else "UNRELATED"
```

Observed scores from `explore_embeddings.py`:

| Pair | Score | Label |
|---|---|---|
| "risk factors" ↔ "risks the company faces" | 0.414 | SIMILAR ✅ |
| "cybersecurity" ↔ "hacking/data breaches" | 0.588 | SIMILAR ✅ |
| "risk factors" ↔ "revenue last year" | 0.059 | UNRELATED ❌ |
| "cybersecurity" ↔ "Renaissance painting" | -0.007 | UNRELATED ❌ |

---

## ChromaDB — Persistent Vector Store

### What is a Vector Database?

A vector database stores high-dimensional numerical embeddings and lets you search them by semantic similarity rather than exact keyword match. It is the persistence and retrieval layer in a RAG pipeline — without it, embeddings have to be recomputed from scratch every run.

### Why ChromaDB Over In-Memory Storage

| | In-Memory (v1) | ChromaDB (v2) |
|---|---|---|
| Embed on every run | ✅ Yes — costs API money | ❌ No — embed once |
| Survives Python exit | ❌ No | ✅ Yes |
| Survives full reboot | ❌ No | ✅ Yes |
| Production-ready | ❌ No | ✅ Yes |
| Setup complexity | Low | Low |

### The Critical Gotcha — Distance Metrics

This is a real bug that trips up production teams:

| Store | Default Metric | Score range |
|---|---|---|
| LlamaIndex in-memory | Cosine similarity | 0.78–0.81 ✅ |
| ChromaDB (default) | L2 Euclidean distance | 0.65–0.69 ⚠️ |
| ChromaDB (configured) | Cosine similarity | 0.78–0.81 ✅ |

**Always set `hnsw:space` explicitly when creating a ChromaDB collection:**

```python
# Without this, scores will look ~15% lower even though retrieval quality is identical
chroma_client.get_or_create_collection(
    "sec_filings",
    metadata={"hnsw:space": "cosine"}   # cosine | l2 | ip (inner product)
)
```

**If scores drop unexpectedly after switching vector stores:**

```powershell
# Delete collection and rebuild with correct metric
Remove-Item -Recurse -Force .\chroma_db
python main.py
```

### What ChromaDB Stores Per Chunk

| Field | Contents |
|---|---|
| ID | Unique identifier for each chunk |
| Embedding | ~1,536 floats (the actual vector) |
| Document | Raw text of the chunk |
| Metadata | File name, page number, etc. |

### ChromaDB Collections

A collection is a named bucket for your vectors — like a table in SQL. One ChromaDB instance can hold multiple collections:

```python
chroma_client.get_or_create_collection("sec_filings")
chroma_client.get_or_create_collection("earnings_calls")
chroma_client.get_or_create_collection("analyst_reports")
```

### Vector Store Comparison

| Store | Type | Best For |
|---|---|---|
| **ChromaDB** | Open-source, local | Dev, prototyping, this project |
| **Pinecone** | Managed cloud | Production, no infra management |
| **pgvector** | Postgres extension | Teams already on Postgres |
| **Weaviate** | Open-source, self-host | Enterprise, hybrid search |
| **FAISS** | In-memory library | Research, maximum speed |

---

## Pinecone — Cloud Vector Store (Mar 28)

### Index Configuration

Created via app.pinecone.io → Indexes → Create Index → Custom settings:

| Field | Value | Why |
|---|---|---|
| Index name | `sec-filings` | Hyphens required, no underscores |
| Vector type | Dense | OpenAI embeddings are dense — all 1,536 values populated |
| Dimensions | `1536` | Must match text-embedding-3-small exactly |
| Metric | `cosine` | Matches ChromaDB configuration |
| Capacity mode | Serverless | Free tier, no pod management |

> Do NOT use Pinecone's hosted embedding options (Inference API). Always use Custom settings when bringing your own OpenAI embeddings — otherwise Pinecone embeds for you and you lose control of the pipeline.

### v4 — Pinecone Pipeline (Mar 28)

```python
import os, time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# Connect to Pinecone cloud index
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")

stats = pinecone_index.describe_index_stats()
vector_count = stats.get("total_vector_count", 0)
print(f"Vectors in Pinecone: {vector_count}")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if vector_count == 0:
    # First run — embed and upload to cloud
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
else:
    # Subsequent runs — load from Pinecone, no API calls
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What are the main risk factors?")
print(response)
```

### Score Comparison: ChromaDB vs Pinecone (same model — text-embedding-3-small)

| Q# | Question | ChromaDB | Pinecone | Δ |
|---|---|---|---|---|
| Q1 | Main risk factors | 0.539 | 0.382 | -0.157 |
| Q2 | Cybersecurity risks | 0.720 | 0.672 | -0.048 |
| Q3 | Competition | 0.627 | 0.534 | -0.093 |
| Q4 | Regulatory risks | 0.644 | 0.559 | -0.085 |
| Q5 | Macroeconomic risks | 0.649 | 0.567 | -0.082 |

**Why scores differ despite identical embeddings and cosine metric:**
Pinecone's serverless tier uses approximate nearest neighbor (ANN) search with HNSW indexing — it trades a small amount of precision for speed and scalability. ChromaDB on 128 chunks performs exact search (brute force), which is perfectly accurate but doesn't scale. At 128 vectors the difference is visible in scores but answer quality is effectively identical. At millions of vectors, exact search would be impossibly slow — ANN is the only viable option.

> **Rule:** Never compare absolute scores across vector stores, even with the same embedding model and metric. Compare answer quality, not numbers.

### ChromaDB vs Pinecone — When to Use Which

| | ChromaDB | Pinecone |
|---|---|---|
| Where data lives | Local file on disk | AWS/GCP/Azure cloud |
| Survives laptop death | ❌ No | ✅ Yes |
| Team access | ❌ One machine only | ✅ Anyone with API key |
| Search type | Exact (brute force) | Approximate (HNSW/ANN) |
| Scale | Millions (with limits) | Billions of vectors |
| Latency | Near-zero (local) | ~50–200ms (network) |
| Cost | Free forever | Free tier, then usage-based |
| Compliance (HIPAA/SOC2) | You manage | Available on paid plans |
| Setup | Zero config | Account + API key + index |

### Decision Framework

```
Prototyping / solo dev?       → ChromaDB
Team of 2+?                   → Pinecone
Need compliance?              → Pinecone paid or self-hosted Weaviate
Already on Postgres?          → pgvector
Vectors > 100M?               → Pinecone dedicated or Weaviate
```

### Migration: ChromaDB → Pinecone Without Re-Embedding

For large datasets where re-embedding is expensive, extract raw vectors from ChromaDB and upload directly:

```python
# migrate_chroma_to_pinecone.py
import os, chromadb
from pinecone import Pinecone

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_collection("sec_filings_v2")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")

results = chroma_collection.get(include=["embeddings", "documents", "metadatas"])

vectors = []
for id_, embedding, document, metadata in zip(
    results["ids"], results["embeddings"],
    results["documents"], results["metadatas"]
):
    metadata["text"] = document[:1000]   # Pinecone metadata size limit
    vectors.append({"id": id_, "values": embedding, "metadata": metadata})

# Upload in batches of 100
for i in range(0, len(vectors), 100):
    pinecone_index.upsert(vectors=vectors[i:i+100])
    print(f"Uploaded batch {i//100 + 1}")
```

Use this pattern in production when you have millions of vectors — re-embedding costs hundreds of dollars and hours of time.

---

## v4 — Production Pipeline: Pinecone + Reranker + Soft Abstention (Mar 28 PM)

Three improvements applied to `main.py` based on evaluation findings:

```python
import os, time
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core import PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from pinecone import Pinecone

# 1. Always set embedding model explicitly — never trust defaults
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 2. Soft abstention prompt — reduces hallucination without blocking inference
qa_prompt = PromptTemplate(
    "You are a financial document analyst. Answer using ONLY the context below.\n"
    "Rules:\n"
    "  1. If fully absent: say 'This information is not available in the provided documents.'\n"
    "  2. If partial: answer what you can, note what is missing.\n"
    "  3. If inference needed: reason explicitly, cite specific evidence.\n"
    "  4. Never use outside knowledge.\n\n"
    "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer: "
)

# 3. Reranker: top_k=20 candidates → Cohere cross-encoder → top_n=5 for LLM
reranker = CohereRerank(api_key=os.environ.get("COHERE_API_KEY"), top_n=5)

query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker],
    text_qa_template=qa_prompt
)
```

---

## Evaluation System

### Dataset Structure

`evaluation_dataset.json` — 20 questions across 4 categories, each testing a different RAG failure mode:

| Category | Count | What it tests | Failure mode caught |
|---|---|---|---|
| **Factual** | 5 | Direct lookup of specific facts | Retrieval misses |
| **Reasoning** | 5 | Inference across multiple chunks | LLM reasoning depth |
| **Comparison** | 5 | Cross-document synthesis | Source diversity failure |
| **Adversarial** | 5 | Questions with no answer in docs | Hallucination |

### Five Rounds of Debugging

#### Round 1 — Baseline (top_k=3, default prompt)

| Category | Result |
|---|---|
| Factual | ⚠️ 4/5 correct |
| Reasoning | ❌ 0/5 — all abstained (default prompt too strict) |
| Comparison | ❌ 1/5 — all chunks from one file |
| Adversarial | ❌ 1/5 — 4 hallucinated |

#### Round 2 — Hard abstention prompt + top_k=10

**Fix:** Explicit "say not available if absent" instruction.
**Result:** Adversarial 1/5 → 5/5 ✅. Comparison got worse — 3/5 falsely abstained because retrieval returned one source and the strict prompt blocked partial answers.
**Key lesson:** A fix for one category can break another. Prompt engineering has side effects.

#### Round 3 — Per-source diversity (ExactMatchFilter)

**Fix:** For comparison questions, query each file independently via Pinecone metadata filter, then synthesize.
**Result:** Comparison 1/5 → 5/5 ✅.
**Key lesson:** Cosine similarity always returns closest vectors regardless of source. Increasing top_k doesn't fix diversity — only metadata filtering guarantees it.

#### Round 4 — Per-category engine routing + reasoning prompt

**Fix:** Three separate engines. Dedicated reasoning prompt with inference instruction, no abstention rule. Reranker removed from reasoning path.
**Result:** Reasoning 0/5 → 5/5 ✅. All categories answered.
**Key lesson:** Rerankers hurt reasoning questions — they penalize indirect evidence needed for inference.

#### Round 5 — HyDE → SubQuestionQueryEngine for R002, per-source for R005

**R002 problem:** "Which company has greatest third-party operational dependency?" always retrieved Nvidia instead of JPMorgan. JPMorgan's CCP/clearing language is semantically distant from the query phrasing.

**HyDE attempt:** Generated a hypothetical answer and embedded that instead of the raw query. RAGAS confirmed it failed — `context_recall = 0.0`. The hypothetical answer still gravitated toward Nvidia vocabulary.

**SubQuestionQueryEngine fix:** Decomposed the question into one sub-question per company, routed each to a per-file engine, then synthesized. Guaranteed JPMorgan retrieval.
**Result:** R002 `context_recall` improved from 0.0 → 0.667 ✅

**R005 problem:** Cross-document hallucination — LLM attributed Xbox and Surface devices to Nvidia when Microsoft chunks were mixed into the same context.
**Fix:** Routed R005 through `comparison_query()` with per-source retrieval, keeping each company's context isolated.
**Result:** R005 faithful=0.800, recall=1.000, precision=1.000 ✅

### Final Engine Routing

| Question | Engine | Why |
|---|---|---|
| Factual | Reranking (top_k=20 → Cohere top_n=5) | Precise fact retrieval |
| Reasoning R001/R003/R004 | Plain retrieval (top_k=10) | Indirect evidence needed for inference |
| Reasoning R002 | SubQuestionQueryEngine | Semantic mismatch — HyDE failed (recall=0.0) |
| Reasoning R005 | Per-source diversity | Cross-doc hallucination prevention |
| Comparison C001–C005 | Per-source ExactMatchFilter + synthesis | Source diversity guarantee |
| Adversarial | Reranking + soft abstention prompt | Precision + hallucination prevention |

### RAGAS Results (Final Run)

RAGAS measures what retrieval score cannot — answer quality and retrieval usefulness.

**Overall scores:**

```
faithfulness:      0.589  — some LLM inference beyond retrieved chunks
answer_relevancy:  0.582  — answers partially miss question intent
context_recall:    0.708  — retrieval finds most needed chunks
context_precision: 0.660  — some noise in retrieved chunks
```

**Per-question breakdown:**

| ID | Faithful | Relevancy | Recall | Precision | Notes |
|---|---|---|---|---|---|
| F001 | 0.400 | 0.583 | 0.000 | 0.000 | Chunk truncation + source attribution problem |
| F002 | 1.000 | 0.000 | 1.000 | 1.000 | ✅ Perfect retrieval — relevancy fix needed |
| F003 | 0.333 | 0.775 | 1.000 | 1.000 | Faithful low — answer claims not in chunks |
| F004 | 0.000 | 0.523 | 0.000 | 0.000 | Chunk truncation cut off $28.9B figure |
| F005 | 1.000 | 0.872 | 1.000 | 1.000 | ✅ Perfect |
| R001 | 0.500 | 0.545 | 0.500 | 0.583 | Source attribution + plain engine misses chunks |
| R002 | 0.889 | 0.651 | 0.667 | 0.247 | ✅ SubQuestion fixed recall — precision noise remains |
| R003 | 0.471 | 0.635 | 0.667 | 0.959 | Inference adds ungrounded claims |
| R004 | 0.500 | 0.513 | 0.750 | 0.814 | Partial recall — CCAR/Basel not retrieved |
| R005 | 0.800 | 0.724 | 1.000 | 1.000 | ✅ Best reasoning result after per-source fix |

**Category averages:**

| Category | Faithful | Relevancy | Recall | Precision |
|---|---|---|---|---|
| Factual | 0.547 | 0.551 | 0.600 | 0.600 |
| Reasoning | 0.632 | 0.614 | 0.817 | 0.721 |

F001 and F004 recall=0.000 is caused by 1000-char chunk truncation in RAGAS contexts cutting off the key facts. Fix: increase truncation to 2000 chars.

Faithfulness < 1.0 on reasoning questions is expected — inference connects dots not explicitly stated in any chunk.

### Post-Run Debugging Findings (Opus 4.6 Analysis)

After the final RAGAS run, further analysis identified five specific issues prioritized by severity:

**HIGH — R001 and F001 Faithfulness (0.10 / 0.25)**

Root cause: 10-K chunks use "we/our" without naming the company explicitly. RAGAS judges most answer claims as ungrounded because it cannot trace "Microsoft faces ransomware attacks" back to a chunk that says "we face ransomware attacks" — the company name is missing from the chunk text.

Fix: Prepend `[Source: filename]` to each chunk before passing to RAGAS:

```python
"contexts": [
    [f"[Source: {r['unique_sources'][i % len(r['unique_sources'])]}] {text[:2000]}"
     for i, text in enumerate(r["chunk_texts"])]
    for r in ragas_items
],
```

For F001 specifically: also check whether the system answer adds terms not present in any retrieved chunk (e.g. "phishing" if no chunk mentions phishing). If so, tighten the QA prompt or increase top_k to retrieve the chunk containing that term.

**MEDIUM — R002 Precision (0.33)**

Root cause: SubQuestionQueryEngine retrieves too many useful + noise chunks alongside the relevant ones. Precision 0.33 means 2 out of 3 retrieved chunks were not useful for the answer.

Fix: Add Cohere reranker as postprocessor to the per-file engines inside SubQuestion:

```python
def make_filtered_engine(file_name: str):
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="file_name", value=file_name)]
    )
    return index.as_query_engine(
        similarity_top_k=8,
        filters=filters,
        node_postprocessors=[reranker],    # ← add reranker to sub-engines
        text_qa_template=per_source_prompt
    )
```

**MEDIUM — F002 Relevancy (0.48)**

Root cause: Answer is faithful (1.0) but doesn't directly address what was asked. The QA prompt produces comprehensive answers that cover adjacent topics rather than focusing tightly on the question.

Fix: Review `qa_prompt` to add an explicit instruction for concise, question-focused answers:

```python
"  4. Answer only what is directly asked. Do not add adjacent context "
"     unless it is essential to understanding the answer.\n"
```

**LOW — R001 Recall (0.50)**

Root cause: Plain retrieval engine misses chunks needed for multi-company reasoning. R001 asks which company is most exposed to geopolitical risk — the plain engine doesn't guarantee it retrieves relevant chunks from all three companies.

Fix: Route R001 through SubQuestionQueryEngine or per-source retrieval, same as R002 and R005:

```python
# Add R001 to COMPARISON_SOURCES for per-source routing
COMPARISON_SOURCES["R001"] = ["msft_10k.txt", "nvda_10k.txt", "jpm_10k.txt"]
```

### RAGAS Setup — Key API Details

```python
from ragas import evaluate as ragas_evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ragas_llm        = llm_factory("gpt-4o-mini", client=openai_client)
ragas_embeddings = RagasOpenAIEmbeddings(
    model="text-embedding-3-small",
    client=openai_client            # required positional arg in this version
)

# Must instantiate with () — bare class names cause "must be initialised" error
metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ContextRecall(llm=ragas_llm),
    ContextPrecision(llm=ragas_llm),
]
```

Common RAGAS errors encountered and fixes:

- `answer_relevancy=null` → use `RagasOpenAIEmbeddings` directly, not `embedding_factory`
- `unexpected keyword argument 'api_key'` → pass `client=openai_client` instead
- `missing 1 required positional argument: 'client'` → add `client=openai_client`
- `metrics must be initialised objects` → use `Faithfulness()` not `faithfulness` (capital + parentheses)
- `DeprecationWarning: LangchainLLMWrapper` → use `llm_factory` instead

### HyDE — When It Works and When It Doesn't

**What HyDE does:** Instead of embedding the raw question, the LLM generates a hypothetical answer first, then embeds that. Hypothetical answers live in "answer space" rather than "question space" — closer to real document chunks.

**When to use it:** Query language ≠ document language. Best for domain-specific terminology gaps.

**Why it failed for R002:** The hypothetical answer still generated Nvidia/Microsoft vocabulary rather than JPMorgan's CCP/clearing language. The semantic gap was too large even for HyDE.

**The escalation path:**

```
Standard retrieval → HyDE → SubQuestionQueryEngine → manual corpus expansion
```

SubQuestionQueryEngine is definitive because it bypasses semantic similarity entirely — it queries each company's file directly and guarantees retrieval regardless of vocabulary.

### Evaluation Metrics — Two Perspectives

| Metric | What it measures | Tool |
|---|---|---|
| `top_score` | Retrieval proximity — did we fetch relevant chunks? | Cosine similarity or Cohere reranker |
| `correctly_abstained` | Did system refuse out-of-scope questions? | String matching |
| `faithfulness` | Are answer claims grounded in retrieved chunks? | RAGAS (LLM-as-judge) |
| `answer_relevancy` | Does the answer address the question? | RAGAS (embeddings) |
| `context_recall` | Did retrieval find the chunks needed to answer? | RAGAS (LLM-as-judge) |
| `context_precision` | How much retrieved content was actually useful? | RAGAS (LLM-as-judge) |

**The key insight:** `top_score` and RAGAS metrics are independent. High retrieval score + low faithfulness = good retrieval, bad generation. Low retrieval score + high recall = retriever found the right chunks despite low similarity scores.

---

## Auto-Ingest Pipeline (Apr 6–7)

### SEC EDGAR Downloader (`edgar_downloader.py`)

Programmatic 10-K fetching from SEC EDGAR — no account or API key required. SEC only requires a `User-Agent` header with your name and email.

```python
# edgar_downloader.py — core flow
# 1. Ticker → CIK via SEC's company_tickers.json
# 2. CIK → filing metadata via submissions endpoint
# 3. Accession number → download raw HTML filing
# 4. Save to data/filings/{TICKER}_10K_{date}.html

python edgar_downloader.py              # defaults: MSFT, NVDA, JPM
python edgar_downloader.py AAPL GOOG    # custom tickers
```

Key implementation details: CIK must be zero-padded to 10 digits. Rate limiting at 0.12s between requests (SEC allows max 10 req/sec). Retry logic with exponential backoff on 429 responses.

### Auto-Ingest (`ingest.py`)

Connects EDGAR downloader to the RAG pipeline — one command to download, parse, chunk, embed, and index a new filing.

```python
# ingest.py — end-to-end pipeline
# 1. Download 10-K via edgar_downloader (skips if file exists, --force to override)
# 2. Parse HTML → clean text via BeautifulSoup
# 3. Create LlamaIndex Document with metadata (ticker, source, filing_date)
# 4. Chunk with SentenceSplitter (1024 tokens, 128 overlap)
# 5. Delete old vectors for this ticker (deduplication)
# 6. Upsert new vectors to Pinecone via StorageContext

python ingest.py NVDA                   # ingest single ticker
python ingest.py MSFT NVDA JPM          # ingest multiple
python ingest.py NVDA --force           # re-download even if file exists
```

**Lesson learned:** LlamaIndex's `VectorStoreIndex(nodes, vector_store=vector_store)` silently fails to upsert — you must wrap the vector store in `StorageContext.from_defaults(vector_store=vector_store)` and pass `storage_context=` to the index constructor.

---

## Conversational RAG — Multi-Turn Memory (Apr 7)

### The Problem

A standard query engine is **stateless** — each question is independent. When a user asks "How does that compare to Microsoft?", the retriever embeds the word "that" and returns garbage. There's no memory of what "that" refers to.

### The Solution — `condense_plus_context` Chat Engine

```python
# chat_engine.py — key components

from llama_index.core.memory import Memory

# Memory stores conversation history with a token budget
memory = Memory.from_defaults(
    session_id="sec-chat",
    token_limit=3000,
)

# condense_plus_context rewrites follow-ups into standalone queries
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    similarity_top_k=6,
)
```

**How it works step by step:**

1. User asks: "What are NVIDIA's main risk factors?" → retriever searches normally
2. User follows up: "How does that compare to Microsoft?"
3. Condense step reads chat history + follow-up → rewrites to: "How do NVIDIA's risk factors compare to those of Microsoft?"
4. Rewritten query hits vector retriever → retrieves chunks from both companies
5. LLM generates grounded answer from retrieved context

### Streamlit Chat UI (`app.py`)

```powershell
streamlit run app.py
```

Wraps the same chat engine in a web UI with `st.chat_message()` for display and `@st.cache_resource` to avoid reinitializing the Pinecone connection on every rerun.

### Debugging: `similarity_top_k` Matters for Cross-Company Queries

Default `similarity_top_k=2` caused the system to only retrieve chunks from one company on comparison questions. Increasing to `similarity_top_k=6` ensures both companies' chunks appear in retrieved context. This is the same source diversity problem from Mar 28, but at the retrieval level rather than the reranking level.

### Conversation Memory

| Question | Answer |
|---|---|
| *What is multi-turn memory in a RAG system?* | The system stores previous messages so follow-up questions like "tell me more about that" make sense. Without memory, every question is treated in isolation |
| *Explain this to a non-technical person* | "Imagine talking to a colleague about a company's financials. You ask about NVIDIA's risks, they answer, then you say 'How does that compare to Microsoft?' — they understand 'that' means the risks you just discussed. Our system does the same thing." |
| *What is `condense_plus_context`?* | It takes the follow-up + chat history, rewrites the follow-up into a standalone query, then retrieves context and generates an answer. The rewrite step is what makes retrieval work across turns |
| *What happens when conversation gets very long?* | Token limit in the memory buffer. Oldest messages get dropped when the limit is exceeded. You tune this based on your LLM's context window |
| *Does memory persist between sessions?* | No — it's in-memory per session. For persistence you'd store it in a database (LlamaIndex's Memory class supports SQLite and remote DBs) |
| *Why not stuff entire chat history into the prompt?* | Token costs and context window limits. Also, the retriever needs a clean query, not a wall of conversation history |
| *What's the difference between `Memory` and `ChatMemoryBuffer`?* | `ChatMemoryBuffer` is deprecated. `Memory` is the newer API — same FIFO short-term buffer, but adds support for long-term memory blocks (fact extraction, vector memory, static info) |

---

## Framework Comparison: LlamaIndex vs LangChain (Apr 28)

Built a head-to-head comparison of LlamaIndex and LangChain on the same SEC 10-K corpus, with the same OpenAI embeddings, reranker, and LLM. Started as a quick "point both at the same Pinecone index" experiment — turned into a four-stage debugging exercise.

### Stage 1 — The naive comparison fails silently

First attempt: point LangChain's `PineconeVectorStore` at the existing `sec-filings` index that LlamaIndex had populated. LangChain returned **zero documents** for every query, with this warning per record:

```text
Found document with no `text` key. Skipping.
```

**Root cause:** the two frameworks store chunk content differently in vector-DB metadata.

| Framework | Where the chunk text lives in Pinecone metadata |
|---|---|
| LlamaIndex | `_node_content` — a JSON blob containing the entire `TextNode` (text + node metadata) |
| LangChain | `text` — a flat string field, configurable via `text_key="text"` (the default) |

LangChain reads each record, can't find a `text` key, and silently skips it. The retriever returns `[]`, the LLM correctly says "no context provided," and the comparison shows a misleading 0.000 score for LangChain.

**Diagnostic snippet** to inspect what's actually in the metadata:

```python
from pinecone import Pinecone
import os, json

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
idx = pc.Index("sec-filings")
sample = idx.query(vector=[0.0]*1536, top_k=1, include_metadata=True)
print(json.dumps(sample["matches"][0]["metadata"], indent=2)[:800])
```

### Stage 2 — Three options for fixing it

| Option | Effort | Tradeoff |
| --- | --- | --- |
| Re-ingest using LangChain | High | Cleanest but duplicates the work |
| Custom retriever that decodes `_node_content` JSON | Medium | Keeps existing index, but you're benchmarking "LangChain on a LlamaIndex-shaped index" — not LangChain itself |
| Parallel index ingested by LangChain | Medium | Each framework uses its native conventions — most honest comparison |

Picked option 3. Each framework benchmarks against an index built by *its own* ingestion path, which is what you'd actually deploy.

### Stage 3 — Parallel native ingestion (`ingest_langchain.py`)

```python
# ingest_langchain.py — key bits
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Match LlamaIndex's 512-token / 50-overlap budget by counting tokens via tiktoken
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512,
    chunk_overlap=50,
)

# Reuse parse_filing_html() and find_existing_filing() from ingest.py
# so we read the SAME source HTML files

vectorstore = PineconeVectorStore(index=pc_index, embedding=embeddings)
vectorstore.add_documents(docs)   # writes 'text' as a flat metadata key
```

Index split:

```text
sec-filings        ←  ingest.py             (LlamaIndex-native, _node_content JSON)
sec-filings-lc     ←  ingest_langchain.py   (LangChain-native, flat text field)
```

Both indexes are 1536-dim cosine, populated from the same `data/filings/*.html` source files.

**Lesson:** the chunk *boundaries* won't be byte-identical between frameworks — LlamaIndex's `SentenceSplitter` is sentence-aware, while LangChain's `RecursiveCharacterTextSplitter.from_tiktoken_encoder` is recursive on separators with token counting. Both target 512/50, so chunks are similar in size but not identical. That's the realistic comparison: each framework uses its own defaults end-to-end.

### Stage 4 — Optimization parity

First end-to-end comparison was disappointing — answers on both sides were vaguer than the ones in `ragas_results.json` from earlier sessions. Reason: `compare_frameworks.py` was running stripped-down pipelines (raw `top_k=3`, basic prompt) while ragas had been run against the optimized [main.py](main.py) pipeline.

To make the comparison fair to both frameworks **and** representative of production quality, both pipelines now mirror main.py's optimizations:

| Optimization | LlamaIndex side | LangChain side |
| --- | --- | --- |
| Wide-net retrieval (`k=20`) | `index.as_query_engine(similarity_top_k=20, ...)` | `vectorstore.as_retriever(search_kwargs={"k": 20})` |
| Cohere `rerank-english-v3.0` → top 5 | `node_postprocessors=[CohereRerank(top_n=5)]` | `ContextualCompressionRetriever(base_compressor=CohereRerank(top_n=5), ...)` |
| Soft-abstention QA prompt | `text_qa_template=PromptTemplate(...)` | `ChatPromptTemplate.from_template(...)` |
| Cohere rate-limit pacing | `time.sleep(6)` between queries | `time.sleep(6)` between queries |

After this, the `top_score` field in both pipelines is the **Cohere rerank relevance score** (0–1, same scale on both sides) — so it is directly comparable across frameworks. Embedding-similarity scores from the unoptimized comparison are not.

```text
python compare_frameworks.py
# →  framework_comparison.json with config, per-question rerank scores, sources, answers
```

### LangChain v1.x reorganization gotcha

`ContextualCompressionRetriever` lives in **`langchain_classic`** in LangChain v1.x, not `langchain.retrievers`:

```python
# Used to be:  from langchain.retrievers.contextual_compression import ...
# In v1.x:
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
```

`langchain` and `langchain_classic` are both installed alongside each other in v1.x. Pylance still reports a resolved warning if `langchain-cohere` isn't installed — install it explicitly: `pip install langchain-cohere`.

### Framework Comparison

| Question | Answer |
|---|---|
| *Can you point both LlamaIndex and LangChain at the same Pinecone index?* | Technically yes, but the comparison is misleading. Each framework has its own metadata schema for storing chunk text — LlamaIndex serializes the whole node into `_node_content` as JSON, LangChain expects a flat `text` field. Cross-reading produces silent retrieval failures: zero docs returned, no error |
| *How would you do a fair head-to-head between two RAG frameworks?* | One index per framework, ingested by that framework's native path. Use the same source documents, the same embedding model, the same target chunk size/overlap, the same reranker, and the same LLM. The chunk *boundaries* will differ — that's part of what you're comparing |
| *Why didn't you just decode `_node_content` on the LangChain side?* | That benchmarks "LangChain reading a LlamaIndex-shaped index," not "LangChain." The point of the comparison is to evaluate each framework's defaults end-to-end, since that's what someone deploying it would actually run |
| *Why does the LangChain-Pinecone integration silently skip records?* | The default `text_key="text"` parameter looks for a flat metadata field. If that field isn't present, the record can't be reconstructed into a `Document` and gets skipped with a warning. No exception is raised because partial indexes are a legitimate use case |
| *Is the `top_score` you report comparable between frameworks?* | Only after both sides apply the Cohere reranker. Embedding cosine similarity is not directly comparable across stores or framework conventions, but Cohere's `rerank-english-v3.0` produces relevance scores on a fixed 0–1 scale regardless of where the chunk came from |
| *Why did your initial unoptimized comparison make LangChain look weaker than the ragas results suggested?* | Because the comparison harness was running raw top_k=3 with a basic prompt, while ragas had been run against main.py's optimized pipeline (k=20 → Cohere rerank → soft-abstention). Optimization parity is essential — otherwise you're comparing pipeline configurations, not frameworks |
| *What does LangChain v1.x's reorganization affect?* | Legacy chain/retriever code moves to `langchain_classic`. `ContextualCompressionRetriever` is one example. Both packages ship side-by-side, so old imports fail at runtime but new imports work |

---

## Industry RAG Stack

### Orchestration Frameworks

| Tool | Notes |
|---|---|
| **LlamaIndex** | Best for document-heavy RAG, rich connectors, ideal for 10-Ks and PDFs |
| **LangChain** | Most widely used, large ecosystem, better for agent workflows |
| **Haystack** | Enterprise/NLP-focused, strong eval tooling |

### Vector Databases

| Tool | Notes |
|---|---|
| **Pinecone** | Most popular managed option |
| **pgvector** | Postgres extension — common in enterprises already on Postgres |
| **Weaviate** | Open-source, strong hybrid search |
| **Chroma** | Lightweight, used by default in LlamaIndex for local dev |

### Embedding Models

| Tool | Notes |
|---|---|
| **OpenAI text-embedding-3-small** | Best cost/quality — used in this project |
| **OpenAI text-embedding-ada-002** | Legacy — 5x more expensive, no quality advantage |
| **sentence-transformers** | Free, runs locally — `all-MiniLM-L6-v2` is the workhorse |
| **Cohere Embed** | Strong multilingual support |

### Rerankers

| Tool | Notes |
|---|---|
| **Cohere Rerank** | Most commonly used managed reranker — used in this project |
| **BGE Reranker** | Strong open-source option via HuggingFace |
| **FlashRank** | Lightweight open-source, fast |

### Evaluation Tools

| Tool | Notes |
|---|---|
| **RAGAS** | Measures faithfulness, answer relevancy, context recall |
| **LangSmith** | Tracing and eval for LangChain pipelines |
| **TruLens** | Works natively with LlamaIndex |

---

## Important Concepts for This Project

### Core RAG Concepts

| Question | Answer |
|---|---|
| *What is RAG?* | Combining vector search (retrieval) with a generative model so answers are grounded in specific documents |
| *Why not fine-tune?* | Fine-tuning is expensive, slow to update, and doesn't cite sources. RAG lets you swap documents at runtime |
| *What is an embedding?* | A list of numbers representing the meaning of text. Semantically similar text maps to nearby points in vector space |
| *Why do similar questions have similar embeddings?* | The model was trained to place semantically equivalent sentences near each other in vector space regardless of exact wording |
| *What is a vector database?* | A store for embeddings that supports similarity search — the persistence layer in a RAG pipeline |
| *What are failure modes of RAG?* | Retrieval misses (wrong chunks fetched), context overflow, hallucination when chunks are ambiguous, source diversity failure on cross-document queries |

### Embedding Models

| Question | Answer |
|---|---|
| *Why use 3-small over ada-002?* | 5x cheaper at $0.02/1M tokens vs $0.10, with equal or better retrieval quality |
| *Why did ada-002 score higher than 3-small?* | Different vector space distributions — ada-002 compresses scores into a narrow high range, 3-small uses a wider dynamic range. Higher absolute score does not mean better retrieval. Never compare scores across models |
| *Can you mix embeddings from two models in one collection?* | Never — each model has its own vector space. Mixing produces nonsensical similarity scores with no error message |
| *What are embedding dimensions?* | The length of the vector. 3-small produces 1,536 dimensions. Must exactly match the Pinecone index dimension — upsert fails immediately if mismatched |

### Vector Stores

| Question | Answer |
|---|---|
| *Why did your scores drop when switching to ChromaDB?* | ChromaDB defaults to L2 Euclidean distance while LlamaIndex expects cosine. Always set `hnsw:space: cosine` explicitly on collection creation |
| *Why were Pinecone scores lower than ChromaDB with identical embeddings?* | Pinecone uses approximate nearest neighbor (ANN/HNSW) search for scalability. ChromaDB on 128 chunks uses exact brute-force search. ANN trades small precision loss for ability to scale to billions of vectors |
| *When would you choose ChromaDB over Pinecone?* | Solo development, prototyping, cost-sensitive projects, or when network latency is unacceptable. ChromaDB is just a local file — it disappears if your laptop dies |
| *How would you migrate from ChromaDB to Pinecone without re-embedding?* | Extract raw vectors via `chroma_collection.get(include=["embeddings"])`, reformat to Pinecone's upsert schema `{id, values, metadata}`, upload in batches of 100 |
| *What Pinecone index settings matter most?* | Dimensions must match embedding model exactly (1536 for 3-small). Metric must be cosine. Always use Custom settings — never Inference API — when bringing your own embeddings |

### Reranking

| Question | Answer |
|---|---|
| *What is a reranker?* | A second-pass cross-encoder that reads the query and each chunk together, producing more accurate relevance scores than embedding similarity alone |
| *Why is a cross-encoder more accurate than embedding similarity?* | Embedding similarity compares two vectors independently. A cross-encoder reads query + chunk jointly — it understands their interaction, not just their proximity |
| *When should you NOT use a reranker?* | Reasoning questions — the reranker penalizes chunks that don't directly answer the question, which kills multi-hop inference. We found this empirically: R003 had score 0.920 but abstained because the reranker filtered out all indirect evidence |
| *What does a reranker pipeline look like?* | Stage 1: embedding similarity retrieves top 20 candidates. Stage 2: Cohere cross-encoder reranks to top 5. Stage 3: LLM answers from top 5 |

### Evaluation

| Question | Answer |
|---|---|
| *How did you evaluate your RAG system?* | Built a 20-question dataset across 4 categories — factual, reasoning, comparison, adversarial — then ran 5 debugging rounds, each fixing a specific failure mode |
| *What are adversarial questions?* | Questions where the answer is intentionally not in the documents. They test whether the system hallucinates or correctly abstains |
| *Why does retrieval score not equal answer quality?* | Score measures vector proximity, not correctness. Comparison questions scored higher than factual on average but were actually failing — all 10 retrieved chunks came from one company |
| *What is source diversity failure?* | When all retrieved chunks come from one document despite a question requiring multiple. Increasing top_k doesn't fix this — cosine similarity always returns closest vectors regardless of source |
| *How did you fix hallucination?* | Added a soft abstention prompt — absent: say not available, partial: answer and flag gaps, inference needed: reason explicitly |
| *What is HyDE?* | Hypothetical Document Embeddings — embed a hypothetical answer instead of the raw question to bridge query/document vocabulary gaps |
| *When does HyDE fail?* | When the semantic gap is too large — our R002 hypothetical answer still generated Nvidia vocabulary rather than JPMorgan's CCP/clearing language. RAGAS confirmed recall=0.0 |
| *What is SubQuestionQueryEngine?* | Decomposes a multi-entity question into one sub-question per entity, routes each to a dedicated per-file engine, then synthesizes. Bypasses semantic similarity entirely — guarantees retrieval regardless of vocabulary |
| *What are RAGAS metrics?* | Faithfulness (are claims grounded in chunks?), answer relevancy (does answer address question?), context recall (did retrieval find needed chunks?), context precision (was retrieved content useful?) |
| *Why did F001 and F004 have recall=0.0 in RAGAS?* | 1000-char chunk truncation in RAGAS context cut off the key facts. Increasing to 2000 chars fixes it |
| *Why did you use different engines for different question types?* | One engine doesn't fit all. Reranking hurts reasoning (penalizes indirect evidence), plain retrieval hurts precision, neither guarantees source diversity |
| *What is a cross-document hallucination?* | LLM attributes facts from one company to another when their chunks are mixed in the same context. R005 had the LLM attributing Xbox/Surface to Nvidia. Fix: per-source retrieval keeps each company's context isolated |

---

## Debugging Lessons — Key Takeaways

These are the insights that came from real bugs encountered in this project.

**1. Retrieval score ≠ answer quality.** Comparison questions had higher average scores than factual questions but were actually failing. Score measures vector proximity, not correctness. Always evaluate answer quality, not just retrieval scores.

**2. One prompt does not fit all query types.** A hard abstention prompt that fixed hallucination (adversarial: 1/5 → 5/5) simultaneously caused false abstentions on reasoning questions. Prompt engineering has side effects across categories.

**3. Rerankers hurt reasoning questions.** The Cohere reranker gave R003 a score of 0.920 — excellent retrieval — but the system abstained because the reranker filtered out all chunks that didn't directly answer the question. Reasoning requires indirect evidence. Remove the reranker for inference-heavy queries.

**4. Increasing top_k doesn't fix source diversity.** Cosine similarity always returns the closest vectors regardless of source file. Even at top_k=10, all 10 chunks came from JPMorgan for questions asking about Microsoft and JPMorgan. The only fix is metadata-filtered per-source retrieval.

**5. Different vector stores have different score scales.** ChromaDB (exact search) scores ranged 0.78–0.87. Pinecone (ANN) scored 0.38–0.67. Reranker scores are on a completely different scale again. Never compare scores across retrieval systems.

**6. Embedding model defaults are silent.** LlamaIndex silently defaults to `text-embedding-ada-002` if `Settings.embed_model` is not set. Always set it explicitly. Changing it without re-indexing produces nonsensical similarity scores with no error.

**7. ChromaDB distance metric defaults to L2, not cosine.** Scores dropped from 0.78–0.81 to 0.65–0.69 after switching to ChromaDB — not because retrieval got worse, but because the metric changed silently. Always set `metadata={"hnsw:space": "cosine"}` on collection creation.

**8. setx doesn't update the current PowerShell session.** `setx COHERE_API_KEY "..."` saves for future sessions but not the current one. Either restart PowerShell or use `$env:COHERE_API_KEY = "..."` to set it immediately.

**9. HyDE can fail when the semantic gap is too large.** HyDE generates a hypothetical answer and embeds it, bringing the query into "answer space." But for R002, the hypothetical answer still generated Nvidia vocabulary rather than JPMorgan's CCP/clearing language. RAGAS confirmed context_recall=0.0. The escalation path is: standard retrieval → HyDE → SubQuestionQueryEngine.

**10. Cross-document hallucination is caused by mixed context.** When chunks from multiple companies appear in the same context window, the LLM can attribute facts from one company to another. R005 had the LLM attributing Xbox consoles and Surface devices to Nvidia. Fix: per-source retrieval keeps each company's context isolated before synthesis.

**11. RAGAS context truncation can silently zero out recall.** Truncating chunk texts to 1000 chars in RAGAS contexts caused F001 and F004 to show recall=0.0 — the key facts (attack type lists, $28.9B figure) appeared after the 1000-char cutoff. Increase to 2000 chars.

**12. RAGAS has brittle API versioning.** In the current version: import from `ragas.metrics.collections` not `ragas.metrics`, use capital class names with `()` not lowercase module objects, pass `client=openai_client` explicitly to `RagasOpenAIEmbeddings`, use `llm_factory` not `LangchainLLMWrapper`. These all changed in recent releases.

**13. RAGAS faithfulness is fooled by first-person chunks.** 10-K filings use "we/our" without naming the company. RAGAS judges answer claims like "Microsoft faces ransomware attacks" as ungrounded because no chunk says "Microsoft" — they say "we." Prepend `[Source: filename]` to each chunk in RAGAS contexts to anchor attribution.

**14. Reranker inside SubQuestion sub-engines reduces precision noise.** SubQuestionQueryEngine without a reranker retrieves too many noise chunks alongside useful ones (R002 precision=0.33). Adding the Cohere reranker as a postprocessor to each per-file sub-engine within SubQuestion cleans up the candidate set before generation.

**15. `VectorStoreIndex` silently fails to upsert without `StorageContext`.** Passing `vector_store=` directly to `VectorStoreIndex(nodes, vector_store=vs)` constructs the index object but doesn't persist vectors. You must use `StorageContext.from_defaults(vector_store=vs)` and pass `storage_context=` — otherwise 0 vectors are written with no error.

**16. Chat engine `similarity_top_k` must be tuned for cross-company queries.** Default `top_k=2-3` returns chunks from only one company. For comparison follow-ups like "How does that compare to Microsoft?", even after the condense step rewrites it correctly, the retriever needs `top_k=6+` to surface chunks from both companies. This is the retrieval-level analog of the source diversity problem from Mar 28.

**17. Two RAG frameworks cannot share a vector index without lying about it.** LangChain's `PineconeVectorStore` defaults to `text_key="text"` and silently skips records that don't have that flat metadata key. LlamaIndex stores chunk content in `_node_content` as a JSON-serialized `TextNode`. Pointing LangChain at a LlamaIndex-populated index returns **zero documents** with only a per-record warning — no exception, no error code, no failed query. Always either (a) ingest with each framework's native path into separate indexes, or (b) write a custom adapter that decodes the foreign metadata format. The "same index, two frameworks" comparison is not a real comparison.

**18. Optimization parity matters more than framework choice.** Initial unoptimized comparison made LangChain's answers look worse than LlamaIndex's, but the difference vanished once both pipelines used `k=20 → Cohere rerank top 5 → soft-abstention prompt`. The reranker is doing more work than either retrieval framework. When benchmarking frameworks, control for the optimization stack — otherwise you're benchmarking pipeline configurations, not frameworks.

**19. LangChain v1.x splits legacy retrievers into `langchain_classic`.** `ContextualCompressionRetriever`, `RetrievalQA`, and other v0-era constructs moved out of `langchain.retrievers` and into the `langchain_classic` namespace. Both packages ship side-by-side in v1.x. Old import paths fail at runtime; tutorials written for v0.x will break.

---

## Next Steps

- [x] ~~Rebuild the same pipeline in **LangChain** to compare frameworks~~ ✅ Done Apr 28 (`compare_frameworks.py` with optimization parity)
- [ ] Try **SemanticSplitter** for more coherent chunks
- [ ] Try `text-embedding-3-large` and measure quality vs cost tradeoff
- [ ] **HIGH** — Prepend `[Source: filename]` to RAGAS contexts to fix F001/R001 faithfulness (first-person chunks)
- [ ] **HIGH** — Increase RAGAS chunk truncation to 2000 chars to fix F001/F004 recall=0.0
- [ ] **MEDIUM** — Add Cohere reranker to SubQuestion sub-engines to fix R002 precision (0.33)
- [ ] **MEDIUM** — Tighten `qa_prompt` with concise-answer instruction to fix F002 relevancy (0.48)
- [ ] **LOW** — Route R001 through per-source or SubQuestion to fix recall (0.50)
- [ ] Add streaming responses to Streamlit chat UI
- [ ] Add long-term memory (FactExtractionMemoryBlock) for persistent user preferences
- [ ] Index full 10-K (not just Item 1A) to capture JPMorgan CCP/clearing sections
- [x] ~~Multi-turn conversation memory with condense_plus_context~~ ✅ Done Apr 7
- [x] ~~Streamlit chat UI~~ ✅ Done Apr 7
- [x] ~~Auto-ingest pipeline (EDGAR → parse → Pinecone)~~ ✅ Done Apr 7
- [x] ~~SEC EDGAR API downloader~~ ✅ Done Apr 6
- [x] ~~Persist to Chroma instead of in-memory storage~~ ✅ Done Mar 24
- [x] ~~Switch to text-embedding-3-small and compare vs ada-002~~ ✅ Done Mar 26
- [x] ~~Migrate to Pinecone and compare vs ChromaDB~~ ✅ Done Mar 28 AM
- [x] ~~Build evaluation dataset (20 Q&A pairs)~~ ✅ Done Mar 28 PM
- [x] ~~Add reranker~~ ✅ Done Mar 28 PM
- [x] ~~Fix comparison diversity with per-source metadata filtering~~ ✅ Done Mar 28 PM
- [x] ~~Fix reasoning abstention with dedicated prompt and engine~~ ✅ Done Mar 28 PM
- [x] ~~Apply HyDE to R002~~ ✅ Attempted — confirmed failed (recall=0.0)
- [x] ~~Apply SubQuestionQueryEngine to R002~~ ✅ Done — recall improved 0.0 → 0.667
- [x] ~~Fix R005 cross-document hallucination~~ ✅ Done — per-source routing
- [x] ~~Integrate RAGAS (faithfulness, relevancy, recall, precision)~~ ✅ Done

---

## Resources

- [LlamaIndex Starter Example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/en/stable/understanding/rag)
- [ChromaDB Quickstart](https://docs.trychroma.com/getting-started)
- [Pinecone Quickstart](https://docs.pinecone.io/guides/get-started/quickstart)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Rerank API](https://docs.cohere.com/docs/rerank-2)
- [RAGAS Evaluation Framework](https://docs.ragas.io)
- [RAGAS Metrics Reference](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index?q=%2210-K%22)
- [SEC EDGAR API — Company Tickers](https://www.sec.gov/files/company_tickers.json)
- [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat)
- [LlamaIndex Memory Docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/)
- [OpenAI Platform](https://platform.openai.com)
