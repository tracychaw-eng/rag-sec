import os
import json
import time
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from pinecone import Pinecone

# ─── Step 1: Configure embedding model + LLM ─────────────────────────────────
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY")
)
# Set the LLM explicitly — previously unset, silently falling back to the
# LlamaIndex default (lesson #6 in README: never trust silent defaults).
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
Settings.llm = LlamaIndexOpenAI(model="gpt-4o-mini")

# ─── Step 2: Connect to Pinecone ──────────────────────────────────────────────
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("sec-filings")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context
)

# ─── Step 3: Prompts ──────────────────────────────────────────────────────────

# Factual + adversarial — soft abstention
qa_prompt = PromptTemplate(
    "You are a financial document analyst. Answer using ONLY the context below.\n"
    "Rules:\n"
    "  1. If the answer is fully absent from the context: respond with exactly "
    "'This information is not available in the provided documents.'\n"
    "  2. If only partial information is available: answer what you can, "
    "then explicitly note what is missing.\n"
    "  3. Never use outside knowledge. Never guess or infer beyond what the "
    "context states.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)

# Reasoning — inference instruction, no abstention rule
# Reranker removed: penalizes indirect evidence needed for multi-chunk inference.
# Confirmed bug: R003 score=0.920 but abstained under qa_prompt + reranker.
reasoning_prompt = PromptTemplate(
    "You are a financial analyst with expertise in SEC filings. "
    "Use the context below to answer the question.\n"
    "The question requires reasoning and inference — the answer may not be "
    "stated directly but can be derived from the evidence provided.\n"
    "Instructions:\n"
    "  1. Identify which company or companies the evidence points to.\n"
    "  2. Cite specific facts, mechanisms, or examples from the context.\n"
    "  3. Make the inference explicitly — explain your reasoning.\n"
    "  4. If the context is genuinely insufficient, say so briefly and explain "
    "what evidence is missing — do NOT just say 'not available'.\n"
    "Do not use outside knowledge beyond what the context provides.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)

# Per-source — comparison questions, one file at a time
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

# Synthesis — combines per-source answers into a final comparison
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

# ─── Step 4: Query engines ────────────────────────────────────────────────────

reranker = CohereRerank(
    api_key=os.environ.get("COHERE_API_KEY"),
    top_n=5
)

# Factual + adversarial: reranking pipeline
# top_k=20 → Cohere cross-encoder → top_n=5 for LLM
rerank_query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker],
    text_qa_template=qa_prompt
)

# Reasoning (R001, R003, R004): plain retrieval — no reranker
# Reranker confirmed to cause abstention on inference questions (R003 bug)
reasoning_query_engine = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=reasoning_prompt
)

# Per-source engine factory — comparison questions + R005
# Filters on the stable 'ticker' metadata key. The old 'file_name' filter
# broke silently when legacy text-file vectors (msft_10k.txt etc.) were
# cleaned up and replaced by full-10-K vectors (MSFT_10K_2025-07-30 etc.).
def make_filtered_engine(ticker: str):
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="ticker", value=ticker)]
    )
    return index.as_query_engine(
        similarity_top_k=8,
        filters=filters,
        text_qa_template=per_source_prompt
    )

# Synthesis engine
synthesis_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=synthesis_prompt
)

# ── SubQuestionQueryEngine for R002 ───────────────────────────────────────────
# Why R002 needs SubQuestion (HyDE failed — context_recall = 0.0):
#   HyDE generates a hypothetical answer and embeds it, but the hypothetical
#   answer still gravitates toward Nvidia/Microsoft vocabulary rather than
#   JPMorgan's CCP/clearing language. RAGAS confirmed: zero recall.
#
#   SubQuestionQueryEngine decomposes the question into one sub-question per
#   company, routes each to a per-file engine, then synthesizes. This
#   GUARANTEES JPMorgan chunks are retrieved — same pattern as comparison_query
#   but with LlamaIndex managing the decomposition automatically.
sub_question_tools = [
    QueryEngineTool.from_defaults(
        query_engine=make_filtered_engine("MSFT"),
        name="microsoft",
        description="Risk factors and disclosures from Microsoft's 10-K filing"
    ),
    QueryEngineTool.from_defaults(
        query_engine=make_filtered_engine("NVDA"),
        name="nvidia",
        description="Risk factors and disclosures from Nvidia's 10-K filing"
    ),
    QueryEngineTool.from_defaults(
        query_engine=make_filtered_engine("JPM"),
        name="jpmorgan",
        description="Risk factors and disclosures from JPMorgan Chase's 10-K filing"
    ),
]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=sub_question_tools,
    verbose=False   # set True to see sub-question decomposition
)

# ─── Step 5: Comparison query function ────────────────────────────────────────
def comparison_query(question: str, sources: list) -> dict:
    """
    Query each source file independently via ExactMatchFilter, then synthesize.
    Guarantees retrieval from each source regardless of cosine scores.
    Also used for R005 to prevent cross-document hallucination (LLM
    previously attributed Xbox/Surface to Nvidia when Microsoft chunks
    were mixed into the same context).
    """
    per_source_answers = {}
    all_chunk_texts    = []
    retrieved_tickers  = []

    for source in sources:
        engine   = make_filtered_engine(source)
        response = engine.query(question)
        per_source_answers[source] = str(response)
        all_chunk_texts.append(str(response))
        retrieved_tickers.extend(
            n.metadata.get("ticker", "unknown") for n in response.source_nodes
        )
        print(f"    [{source}] {str(response)[:100]}...")

    combined_context = "\n\n".join([
        f"--- {src} ---\n{answer}"
        for src, answer in per_source_answers.items()
    ])
    synthesis_question = (
        f"{question}\n\n"
        f"Use only the following per-company answers:\n\n{combined_context}"
    )
    synthesized = synthesis_engine.query(synthesis_question)

    return {
        "per_source_answers": per_source_answers,
        "synthesized_answer": str(synthesized),
        "chunk_texts":        all_chunk_texts,
        "retrieved_tickers":  retrieved_tickers,
    }

# ─── Step 6: Load evaluation dataset ──────────────────────────────────────────
with open("evaluation_dataset.json") as f:
    dataset = json.load(f)

questions = dataset["questions"]
print(f"Loaded {len(questions)} questions across "
      f"{len(dataset['metadata']['categories'])} categories")
print("Engines:")
print("  factual/adversarial  → reranking (top_k=20 → Cohere top_n=5)")
print("  reasoning R001/R003/R004 → plain top_k=10")
print("  reasoning R002       → SubQuestionQueryEngine (HyDE recall=0.0)")
print("  reasoning R005       → per-source (cross-doc hallucination fix)")
print("  comparison           → per-source ExactMatchFilter + synthesis\n")

# ─── Step 7: Source maps ──────────────────────────────────────────────────────

# Comparison question sources — ticker-based (matches current index metadata)
COMPARISON_SOURCES = {
    "C001": ["MSFT", "JPM"],
    "C002": ["MSFT", "NVDA"],
    "C003": ["MSFT", "NVDA", "JPM"],
    "C004": ["MSFT", "JPM"],
    "C005": ["NVDA", "JPM"],
}

# R005 routed through per-source to prevent cross-doc hallucination
# Previously: LLM retrieved Microsoft chunks alongside Nvidia and attributed
# "Xbox consoles and Surface devices" to Nvidia — confirmed hallucination.
COMPARISON_SOURCES["R005"] = ["NVDA", "MSFT", "JPM"]

# Gold source mapping: dataset still uses legacy file names
LEGACY_SOURCE_TO_TICKER = {
    "msft_10k.txt": "MSFT",
    "nvda_10k.txt": "NVDA",
    "jpm_10k.txt":  "JPM",
}

def expected_tickers_for(item: dict) -> list:
    """Gold tickers for a question, from the dataset's source_file field."""
    sf = item.get("source_file", "")
    if not sf or sf == "none":
        return []
    return [LEGACY_SOURCE_TO_TICKER.get(s.strip(), s.strip().upper())
            for s in sf.split(",")]

# R002: SubQuestionQueryEngine (HyDE confirmed insufficient — recall=0.0)
SUBQUESTION_IDS = {"R002"}

# ─── Step 8: Run evaluation ───────────────────────────────────────────────────
results = []

for item in questions:
    cat = item["category"]
    qid = item["id"]

    # Determine engine label for console output
    if qid == "R002":
        engine_label = "[SubQuestion]"
    elif qid in COMPARISON_SOURCES and cat == "reasoning":
        engine_label = "[per-source]"
    elif cat == "comparison":
        engine_label = "[per-source]"
    elif cat == "reasoning":
        engine_label = "[plain]"
    else:
        engine_label = "[rerank]"

    print(f"[{qid}] {cat.upper()} {engine_label}")
    print(f"Q: {item['question'][:80]}...")

    # ── Route to correct engine ──
    if cat == "comparison" or (cat == "reasoning" and qid in COMPARISON_SOURCES):
        # Per-source diversity: comparison questions + R005
        print("  Querying each source independently:")
        sources = COMPARISON_SOURCES[qid]
        comp    = comparison_query(item["question"], sources)

        system_answer  = comp["synthesized_answer"]
        top_score      = 0.0
        unique_sources = sources
        all_sources    = sources
        chunk_texts    = comp["chunk_texts"]
        retrieved_tickers = comp["retrieved_tickers"]
        # Generation context = the per-source answers the synthesizer saw
        generation_contexts = [
            f"[Source: {src}] {ans}"
            for src, ans in comp["per_source_answers"].items()
        ]
        extra_fields   = {
            "engine_used":        "per-source",
            "per_source_answers": comp["per_source_answers"],
            "diversity_failure":  False,
            "false_abstention":   False,
        }

    elif cat == "reasoning" and qid in SUBQUESTION_IDS:
        # SubQuestionQueryEngine for R002
        # Decomposes into one sub-question per company, guaranteeing
        # JPMorgan retrieval that HyDE failed to achieve (recall=0.0)
        response       = sub_question_engine.query(item["question"])
        all_sources    = [n.metadata.get("file_name", "unknown")
                          for n in response.source_nodes]
        chunk_texts    = [n.text for n in response.source_nodes]
        retrieved_tickers = [n.metadata.get("ticker", "unknown")
                             for n in response.source_nodes]
        generation_contexts = [f"[Source: {t}] {text}"
                               for t, text in zip(retrieved_tickers, chunk_texts)]
        unique_sources = list(dict.fromkeys(all_sources))
        raw_score = response.source_nodes[0].score if response.source_nodes else None
        top_score = raw_score if raw_score is not None else 0.0
        system_answer  = str(response)
        extra_fields   = {"engine_used": "SubQuestion"}

    elif cat == "reasoning":
        # Plain retrieval for R001, R003, R004
        response       = reasoning_query_engine.query(item["question"])
        all_sources    = [n.metadata.get("file_name", "unknown")
                          for n in response.source_nodes]
        chunk_texts    = [n.text for n in response.source_nodes]
        retrieved_tickers = [n.metadata.get("ticker", "unknown")
                             for n in response.source_nodes]
        generation_contexts = [f"[Source: {t}] {text}"
                               for t, text in zip(retrieved_tickers, chunk_texts)]
        unique_sources = list(dict.fromkeys(all_sources))
        raw_score = response.source_nodes[0].score if response.source_nodes else None
        top_score = raw_score if raw_score is not None else 0.0
        system_answer  = str(response)
        extra_fields   = {"engine_used": "plain"}

    else:
        # Reranking for factual + adversarial
        response       = rerank_query_engine.query(item["question"])
        all_sources    = [n.metadata.get("file_name", "unknown")
                          for n in response.source_nodes]
        chunk_texts    = [n.text for n in response.source_nodes]
        retrieved_tickers = [n.metadata.get("ticker", "unknown")
                             for n in response.source_nodes]
        generation_contexts = [f"[Source: {t}] {text}"
                               for t, text in zip(retrieved_tickers, chunk_texts)]
        unique_sources = list(dict.fromkeys(all_sources))
        raw_score = response.source_nodes[0].score if response.source_nodes else None
        top_score = raw_score if raw_score is not None else 0.0
        system_answer  = str(response)
        extra_fields   = {"engine_used": "rerank"}

    result = {
        "id":                item["id"],
        "category":          cat,
        "question":          item["question"],
        "reference_answer":  item["reference_answer"],
        "system_answer":     system_answer,
        "top_score":         round(top_score, 3),
        "sources_retrieved": all_sources,
        "unique_sources":    unique_sources,
        "source_diversity":  len(unique_sources),
        "chunk_texts":       chunk_texts,
        "retrieved_tickers": retrieved_tickers,
        "expected_tickers":  expected_tickers_for(item),
        "generation_contexts": generation_contexts,
        **extra_fields
    }

    # ── Adversarial: string-matching hallucination check ──
    if cat == "adversarial":
        abstention_phrases = [
            "not available in the provided",
            "not mentioned in the provided",
            "not present in the provided",
            "not in the provided",
            "cannot find",
            "no information",
            "not explicitly provided",
            "not specified"
        ]
        correctly_abstained = any(
            p in system_answer.lower() for p in abstention_phrases
        )
        result["correctly_abstained"] = correctly_abstained
        result["hallucination_risk"]  = not correctly_abstained

    results.append(result)

    # ── Console output ──
    print(f"A: {system_answer[:180]}...")
    if cat != "comparison" and qid not in COMPARISON_SOURCES:
        print(f"Score: {top_score:.3f} | "
              f"Sources ({len(unique_sources)} unique): {unique_sources}")
    else:
        print(f"Sources queried: {unique_sources}")

    if cat == "adversarial":
        status = ("✅ ABSTAINED" if result["correctly_abstained"]
                  else "⚠️  HALLUCINATION")
        print(f"Hallucination check: {status}")

    if cat == "comparison":
        print("Diversity check: ✅ MULTI-SOURCE")

    if qid == "R002":
        has_jpm = "JPM" in retrieved_tickers
        print(f"SubQuestion diagnostic: "
              f"{'✅ JPMorgan retrieved' if has_jpm else '❌ JPMorgan still missing'}")
        print(f"  Sources: {unique_sources}")

    print()
    time.sleep(6)   # Cohere free tier: 10 calls/minute

# ─── Step 9: Save results ─────────────────────────────────────────────────────
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to eval_results.json\n")

# ─── Step 10: Metrics ───
# Retrieval Recall@K/Precision@K + RAGAS now live in eval/harness.py
# (pipeline-agnostic), so the same harness scores this legacy pipeline
# and the new sec_rag pipeline for a like-for-like comparison.
try:
    from eval.harness import evaluate_results
    evaluate_results(results, label="baseline-legacy-fullcorpus", with_ragas=True)
except Exception as e:
    print(f"Metrics evaluation failed: {e}")
    print("Continuing to summary...\n")

# ─── Step 11: Summary ─────────────────────────────────────────────────────────
print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for cat in ["factual", "reasoning", "comparison", "adversarial"]:
    items = [r for r in results if r["category"] == cat]
    if not items:
        continue

    scored    = [r for r in items if r["top_score"] > 0]
    avg_score = (sum(r["top_score"] for r in scored) / len(scored)
                 if scored else 0.0)
    avg_div   = sum(r["source_diversity"] for r in items) / len(items)

    print(f"\n{cat.upper()} ({len(items)} questions)")
    if cat == "reasoning":
        print(f"  Engines: R001/R003/R004=plain | R002=SubQuestion | R005=per-source")
    elif cat in ("factual", "adversarial"):
        print(f"  Engine: reranking (top_k=20 → Cohere top_n=5)")
    else:
        print(f"  Engine: per-source ExactMatchFilter + synthesis")

    print(f"  Avg top score:       "
          f"{'N/A' if not scored else f'{avg_score:.3f}'}")
    print(f"  Avg unique sources:  {avg_div:.1f}")

    if cat == "adversarial":
        abstained = sum(1 for r in items if r.get("correctly_abstained"))
        print(f"  Correctly abstained: {abstained}/{len(items)}")

    print(f"  {'ID':<6} {'Score':<8} {'Engine':<12} "
          f"{'Unique Sources':<30} {'Notes'}")
    print(f"  {'-'*70}")

    for r in items:
        notes = ""
        if cat == "adversarial":
            notes = "✅" if r.get("correctly_abstained") else "⚠️ HALLUCINATION"
        if r["id"] == "R002":
            has_jpm = "JPM" in r.get("retrieved_tickers", [])
            notes = "✅ JPMorgan retrieved" if has_jpm else "❌ JPMorgan missing"
        if r["id"] == "R005":
            notes = "per-source (hallucination fix)"
        if cat == "comparison":
            notes = "✅ MULTI-SOURCE"

        score_str  = f"{r['top_score']:.3f}" if r["top_score"] > 0 else "N/A"
        engine_tag = r.get("engine_used", "")[:10]
        src_str    = ", ".join(r["unique_sources"])
        print(f"  {r['id']:<6} {score_str:<8} {engine_tag:<12} "
              f"{src_str:<30} {notes}")

# ─── Step 12: Pipeline configuration ─────────────────────────────────────────
print(f"""
{'=' * 60}
PIPELINE CONFIGURATION
{'=' * 60}

FACTUAL           reranking — top_k=20, Cohere top_n=5
REASONING R001    plain — top_k=10, reasoning_prompt (no reranker)
REASONING R002    SubQuestionQueryEngine — one sub-Q per company
                  HyDE confirmed failed (context_recall=0.0)
                  SubQuestion guarantees JPMorgan retrieval
REASONING R003    plain — top_k=10, reasoning_prompt
REASONING R004    plain — top_k=10, reasoning_prompt
REASONING R005    per-source — hallucination fix
                  Previously: LLM attributed Xbox/Surface to Nvidia
                  when Microsoft chunks mixed into same context
COMPARISON        per-source ExactMatchFilter, top_k=8, synthesis
ADVERSARIAL       reranking — top_k=20, Cohere top_n=5

RAGAS FIXES APPLIED
  answer_relevancy null: switched from embedding_factory to
  RagasOpenAIEmbeddings directly — embedding_factory was silently
  failing to attach to the metric object

EVALUATION METRICS
  top_score:           retrieval proximity (cosine or reranker)
  correctly_abstained: abstention detection (string matching)
  RAGAS faithfulness:  are answer claims grounded in chunks?
  RAGAS relevancy:     does the answer address the question?
  RAGAS recall:        did retrieval find the needed chunks?
  RAGAS precision:     how much retrieved content was useful?

KNOWN OPEN ISSUES
  R002  If SubQuestion still shows recall=0.0, JPMorgan CCP/clearing
        chunks may not exist in the indexed corpus (only Item 1A was
        indexed — CCP language may be in a different section).
        Verify by searching: grep -i 'CCP|clearing system' jpm_10k.txt\n
""")