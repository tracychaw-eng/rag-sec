import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from pinecone import Pinecone

# ─── Step 1: Embedding model — always set explicitly ──────────────────────────
# Never rely on LlamaIndex defaults. Default is text-embedding-ada-002
# (legacy, 5x more expensive than 3-small with no quality advantage).
# Must match the model used when the index was originally built.
# Changing this without re-indexing produces nonsensical similarity scores.
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)

Settings.text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# ─── Step 2: Connect to Pinecone ──────────────────────────────────────────────
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")

stats = pinecone_index.describe_index_stats()
vector_count = stats.get("total_vector_count", 0)
print(f"Vectors in Pinecone: {vector_count}")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ─── Step 3: Build or load index ──────────────────────────────────────────────
if vector_count == 0:
    print("No vectors found. Embedding and uploading to Pinecone...")
    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} documents")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"Indexed {pinecone_index.describe_index_stats()['total_vector_count']} chunks")

else:
    print("Loading existing index from Pinecone...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    print("Index loaded. No embedding API calls made.")

# ─── Step 4: Soft abstention prompt ───────────────────────────────────────────
# Reduces hallucination by instructing the LLM to acknowledge missing context
# rather than confabulating an answer from its pretrained knowledge.
#
# "Soft" vs "hard" abstention:
#   Hard: "If not in context, say not available" — caused false abstentions on
#         reasoning questions where the answer requires inference not lookup.
#   Soft: distinguishes fully absent (say so) from partially available
#         (answer what you can, flag what's missing). Better for general use
#         where query type is not known in advance.
qa_prompt = PromptTemplate(
    "You are a financial document analyst. Answer using ONLY the context below.\n"
    "Rules:\n"
    "  1. If the answer is fully absent from the context: respond with exactly "
    "'This information is not available in the provided documents.'\n"
    "  2. If only partial information is available: answer what you can, "
    "then explicitly note what is missing.\n"
    "  3. If the question requires inference or reasoning from the context: "
    "make the inference explicitly and cite the specific evidence you used.\n"
    "  4. Never use outside knowledge. Never guess beyond what the context states.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)

# ─── Step 5: Reranker ─────────────────────────────────────────────────────────
# Two-stage retrieval pipeline:
#   Stage 1 — embedding similarity (fast, approximate):
#              top_k=20 casts a wide net of candidates
#   Stage 2 — Cohere cross-encoder reranking (slower, accurate):
#              re-scores all 20 candidates by reading query + chunk together,
#              then keeps top_n=5 most relevant for the LLM
#
# Why reranking improves results:
#   Embedding similarity measures vector proximity — it finds chunks that are
#   semantically close to the query but can miss nuanced relevance.
#   A cross-encoder reads the full query and chunk together, producing much
#   more accurate relevance judgments at the cost of speed.
#
# Rate limit note: Cohere free tier = 10 calls/minute.
# Each query = 1 reranker call. Add time.sleep(6) between queries if
# running many in a loop to avoid TooManyRequestsError.
reranker = CohereRerank(
    api_key=os.environ.get("COHERE_API_KEY"),
    top_n=5
)

# ─── Step 6: Query engine ─────────────────────────────────────────────────────
query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker],
    text_qa_template=qa_prompt
)

# ─── Step 7: Query ────────────────────────────────────────────────────────────
questions = [
    "What are the main risk factors?",
    "What cybersecurity risks does the company face?",
    "How does competition affect the business?",
    "What regulatory risks are mentioned?",
    "What macroeconomic risks could impact performance?"
]

print("\n" + "=" * 60)
print("QUERY RESULTS — text-embedding-3-small + Cohere reranker")
print("=" * 60)

results = []

for i, question in enumerate(questions, 1):
    print(f"\nQ{i}: {question}")
    print("-" * 50)

    response = query_engine.query(question)
    print(f"Answer: {response}\n")

    print("Retrieved chunks:")
    for j, node in enumerate(response.source_nodes):
        print(f"  Chunk {j+1} | Score: {node.score:.3f} | "
              f"Source: {node.metadata.get('file_name', 'unknown')}")

    results.append({
        "question":   question,
        "answer":     str(response),
        "top_score":  round(response.source_nodes[0].score, 3)
                      if response.source_nodes else 0,
        "sources":    list(dict.fromkeys(
                          n.metadata.get("file_name", "unknown")
                          for n in response.source_nodes
                      ))
    })

    # Rate limit: Cohere free tier is 10 calls/minute
    # Sleep between queries to avoid TooManyRequestsError
    if i < len(questions):
        time.sleep(6)

# ─── Step 8: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCORE SUMMARY")
print("=" * 60)
print(f"{'Q#':<4} {'Top Score':<12} {'Sources'}")
print("-" * 60)
for i, r in enumerate(results, 1):
    sources = ", ".join(r["sources"])
    print(f"Q{i:<3} {r['top_score']:<12.3f} {sources}")