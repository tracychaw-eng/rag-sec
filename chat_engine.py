"""
chat_engine.py — Multi-turn conversational RAG over SEC 10-K filings.
Uses condense_plus_context mode: rewrites follow-up questions into
standalone queries so vector retrieval works across conversation turns.

Usage:
    python chat_engine.py

Test sequence:
    You: What are NVIDIA's main risk factors?
    You: How does that compare to Microsoft?
    You: Which of those two has more regulatory risk?
    You: Summarize that in 3 bullet points.
"""

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import Memory
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# ---------------------------------------------------------------------------
# CONNECT TO PINECONE
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index(os.environ.get("PINECONE_INDEX", "sec-filings"))
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

print(f"✓ Connected to Pinecone index: {os.environ.get('PINECONE_INDEX', 'sec-filings')}")

# ---------------------------------------------------------------------------
# MEMORY + CHAT ENGINE
# ---------------------------------------------------------------------------
memory = Memory.from_defaults(
    session_id="sec-chat",
    token_limit=3000,
)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    verbose=True,  # shows the condensed standalone query
    similarity_top_k=6,
    system_prompt=(
        "You are a financial analyst assistant with access to SEC 10-K filings "
        "for Microsoft (MSFT), NVIDIA (NVDA), and JPMorgan Chase (JPM). "
        "Answer questions based on the filing content. "
        "If the context doesn't contain enough information, say so clearly. "
        "Be concise and cite which company's filing you're referencing."
    ),
)

# ---------------------------------------------------------------------------
# INTERACTIVE CHAT LOOP
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("📄 SEC 10-K Filing Chat (multi-turn)")
    print("   Type 'quit' to exit, 'reset' to clear memory")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            chat_engine.reset()
            print("🔄 Memory cleared. Starting fresh conversation.")
            continue

        response = chat_engine.chat(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()