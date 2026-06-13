# This confirms the data is actually in Pinecone, not just cached locally
import os
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("sec-filings")

# Index statistics
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Dimension:     {stats['dimension']}")

# Embed a query and search directly — bypassing LlamaIndex entirely
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
query_vector = embed_model.get_query_embedding("What are cybersecurity risks?")

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print("\n--- DIRECT PINECONE QUERY ---")
for match in results["matches"]:
    print(f"\nScore: {match['score']:.3f}")
    print(f"ID:    {match['id']}")
    print(f"File:  {match['metadata'].get('file_name', 'unknown')}")
    text = match["metadata"].get("text", "")
    print(f"Text:  {text[:200]}")

