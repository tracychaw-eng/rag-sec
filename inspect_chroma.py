import chromadb

# Connect to the same persistent store
client = chromadb.PersistentClient(path="./chroma_db")

# List all collections
print(client.list_collections())

# Open your collection
collection = client.get_collection("sec_filings")

# How many chunks?
print(f"Total chunks: {collection.count()}")

# Peek at the first 3 chunks (raw data, no query)
results = collection.peek(limit=3)
print("\n--- RAW CHUNK SAMPLE ---")
for i, doc in enumerate(results['documents']):
    print(f"\nChunk {i+1}:")
    print(f"ID:       {results['ids'][i]}")
    print(f"Metadata: {results['metadatas'][i]}")
    print(f"Text:     {doc[:200]}")