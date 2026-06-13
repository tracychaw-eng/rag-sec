# This is the production-grade migration pattern — 
# extract vectors from ChromaDB and upload directly to 
# Pinecone without calling OpenAI at all

import os
import chromadb
from pinecone import Pinecone

# Source: ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_collection("sec_filings_v2")

# Destination: Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")

# Pull everything from ChromaDB (raw vectors included)
print(f"Fetching {chroma_collection.count()} vectors from ChromaDB...")
results = chroma_collection.get(
    include=["embeddings", "documents", "metadatas"]
)

# Reformat for Pinecone's upsert format
vectors = []
for i, (id_, embedding, document, metadata) in enumerate(zip(
    results["ids"],
    results["embeddings"],
    results["documents"],
    results["metadatas"]
)):
    # Pinecone stores metadata as a flat dict — add text here for retrieval
    metadata["text"] = document[:1000]   # Pinecone metadata has size limits
    vectors.append({
        "id": id_,
        "values": embedding,
        "metadata": metadata
    })

# Upload in batches of 100 (Pinecone recommends ≤100 per upsert)
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    pinecone_index.upsert(vectors=batch)
    print(f"Uploaded batch {i//batch_size + 1}: {len(batch)} vectors")

print(f"\nMigration complete. Pinecone now has "
      f"{pinecone_index.describe_index_stats()['total_vector_count']} vectors.")
