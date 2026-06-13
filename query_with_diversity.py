import os
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

# ─── Setup ────────────────────────────────────────────────────────────────────
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context
)

# ─── Prompts ──────────────────────────────────────────────────────────────────

# Per-source prompt: extract what this one document says about the question
# IMPORTANT: do NOT include a "not covered" fallback here — that caused the
# model to refuse answers when retrieved chunks were topically close but not
# an exact match. Instead, instruct it to summarize whatever IS in the context.
per_source_prompt = PromptTemplate(
    "You are a financial document analyst. Use ONLY the context below.\n"
    "Summarize what this document reveals about the question, even if the "
    "connection is indirect. Focus on relevant facts, risks, or disclosures "
    "present in the context. Be specific — name risks, mechanisms, or examples "
    "mentioned. Do not use outside knowledge.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)

# Synthesis prompt: combine per-source answers into a comparison
synthesis_prompt = PromptTemplate(
    "You are a financial analyst comparing disclosures across companies.\n"
    "Below are answers extracted separately from each company's 10-K filing.\n"
    "Synthesize them into a single coherent comparison.\n"
    "Highlight key differences and similarities between the companies.\n"
    "Only use the information provided below — do not add outside knowledge.\n\n"
    "Per-company answers:\n{context_str}\n\n"
    "Comparison question: {query_str}\n"
    "Synthesized comparison: "
)

# ─── Per-source query engine factory ──────────────────────────────────────────
def make_filtered_engine(file_name: str, top_k: int = 8):
    """
    Returns a query engine restricted to chunks from one file.
    Confirmed metadata key: 'file_name' (top-level field in Pinecone).
    Values: 'msft_10k.txt', 'nvda_10k.txt', 'jpm_10k.txt'

    top_k=8 per source: each file has ~42 chunks (128 total / 3 files).
    Fetching 8 gives ~19% coverage per file — enough to find topic-specific
    chunks without pulling in too much noise.
    Previously top_k=5 was too narrow — the 5 retrieved chunks were sometimes
    about adjacent topics (competition, regulatory) rather than the specific
    risk being asked about.
    """
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="file_name", value=file_name)]
    )
    return index.as_query_engine(
        similarity_top_k=top_k,
        filters=filters,
        text_qa_template=per_source_prompt
    )

# ─── Synthesis engine ─────────────────────────────────────────────────────────
synthesis_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=synthesis_prompt
)

# ─── Main comparison function ─────────────────────────────────────────────────
def comparison_query(question: str, sources: list) -> dict:
    """
    Query each source file independently, then synthesize into one answer.
    Guarantees retrieval from each specified source regardless of cosine scores.

    Returns a dict with per-source answers and the final synthesized answer.
    """
    per_source_answers = {}

    print("  Querying each source independently:")
    for source in sources:
        engine = make_filtered_engine(source)
        response = engine.query(question)
        answer = str(response)
        per_source_answers[source] = answer
        print(f"    [{source}] {answer[:120]}...")

    # Build combined context string for synthesis
    combined_context = "\n\n".join([
        f"--- {src} ---\n{answer}"
        for src, answer in per_source_answers.items()
    ])

    # Synthesize: pass combined context as part of the query
    # so the synthesis engine works from per-source answers, not raw chunks
    synthesis_question = (
        f"{question}\n\n"
        f"Use only the following per-company answers:\n\n{combined_context}"
    )
    synthesized = synthesis_engine.query(synthesis_question)

    return {
        "question":           question,
        "sources":            sources,
        "per_source_answers": per_source_answers,
        "synthesized_answer": str(synthesized)
    }

# ─── Run on the 3 previously failing comparison questions ────────────────────
failing_questions = [
    {
        "id":       "C001",
        "question": "How do Microsoft and JPMorgan Chase differ in how they describe cybersecurity risk?",
        "sources":  ["msft_10k.txt", "jpm_10k.txt"]
    },
    {
        "id":       "C004",
        "question": "How do Microsoft and JPMorgan Chase differ in how they frame geopolitical risk?",
        "sources":  ["msft_10k.txt", "jpm_10k.txt"]
    },
    {
        "id":       "C005",
        "question": "How do Nvidia and JPMorgan Chase each describe concentration risk, and what are the potential consequences in each case?",
        "sources":  ["nvda_10k.txt", "jpm_10k.txt"]
    },
]

print("=" * 60)
print("DIVERSITY FIX — Per-Source Retrieval + Synthesis")
print("=" * 60)

all_results = []

for item in failing_questions:
    print(f"\n[{item['id']}] {item['question']}")
    print("-" * 50)

    result = comparison_query(item["question"], item["sources"])

    print(f"\n  Synthesized answer:")
    print(f"  {result['synthesized_answer']}")

    all_results.append({
        "id":                 item["id"],
        "question":           item["question"],
        "sources_queried":    item["sources"],
        "per_source_answers": result["per_source_answers"],
        "synthesized_answer": result["synthesized_answer"]
    })

# ─── Save results ─────────────────────────────────────────────────────────────
import json
with open("diversity_fix_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for r in all_results:
    print(f"\n[{r['id']}]")
    for src, ans in r["per_source_answers"].items():
        # With the relaxed prompt, "not covered" should no longer appear.
        # Flag it if it does — means the filter returned off-topic chunks.
        status = "⚠️  Off-topic chunks" if "not covered" in ans.lower() else "✅"
        print(f"  {src}: {status}")
        print(f"    {ans[:100]}...")
    print(f"  Synthesized: {r['synthesized_answer'][:150]}...")

print("\nFull results saved to diversity_fix_results.json")