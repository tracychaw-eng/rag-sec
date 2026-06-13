import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

# ─── Step 1: Configure embedding model ────────────────────────────────────────
# Must match the dimension of your Pinecone index (1536)
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

# Connect to your existing index (created in dashboard)
pinecone_index = pc.Index("sec-filings")

# Check current state of the index
stats = pinecone_index.describe_index_stats()
vector_count = stats.get("total_vector_count", 0)
print(f"Vectors already in Pinecone: {vector_count}")

# ─── Step 3: Wrap in LlamaIndex interface ─────────────────────────────────────
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ─── Step 4: Build or load index ──────────────────────────────────────────────
if vector_count == 0:
    print("No vectors found. Embedding and uploading to Pinecone...")
    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} documents")

    start_time = time.time()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    elapsed = time.time() - start_time

    # Re-check count after upload
    updated_stats = pinecone_index.describe_index_stats()
    final_count = updated_stats.get("total_vector_count", 0)
    print(f"\nUploaded {final_count} vectors in {elapsed:.1f}s")

else:
    print(f"{vector_count} vectors found in Pinecone. Loading index...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    print("Index loaded. No embedding API calls made.")

# ─── Step 5: Run the same 5 queries ───────────────────────────────────────────
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What are the main risk factors?",
    "What cybersecurity risks does the company face?",
    "How does competition affect the business?",
    "What regulatory risks are mentioned?",
    "What macroeconomic risks could impact performance?"
]

print("\n" + "="*60)
print("QUERY RESULTS — Pinecone + text-embedding-3-small")
print("="*60)

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
        "question": question,
        "top_score": response.source_nodes[0].score if response.source_nodes else 0,
        "sources": [n.metadata.get('file_name', 'unknown')
                    for n in response.source_nodes]
    })

# ─── Step 6: Summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCORE SUMMARY — Pinecone")
print("="*60)
print(f"{'Q#':<4} {'Top Score':<12} {'Sources'}")
print("-" * 60)
for i, r in enumerate(results, 1):
    sources = ", ".join(set(r["sources"]))
    print(f"Q{i:<3} {r['top_score']:<12.3f} {sources}")