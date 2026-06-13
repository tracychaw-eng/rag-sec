# check_metadata.py
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")

# IDs come back as nested lists — flatten them
id_pages = list(pinecone_index.list(limit=3))
ids = [item for sublist in id_pages for item in sublist]
print(f"Sample IDs: {ids[:3]}")

# FetchResponse uses .vectors attribute, not dict access
fetch_result = pinecone_index.fetch(ids=[ids[0]])
vector_data = fetch_result.vectors[ids[0]]

print(f"\n--- Metadata for '{ids[0]}' ---")
print(vector_data.metadata)