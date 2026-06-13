import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
idx = pc.Index("sec-filings")

for fname in ["nvda_10k.txt", "msft_10k.txt", "jpm_10k.txt"]:
    idx.delete(filter={"file_name": {"$eq": fname}})
    print(f"Deleted vectors for {fname}")

print("Done. Remaining vectors:", idx.describe_index_stats()["total_vector_count"])
