"""
compare_frameworks.py — Optimized LlamaIndex vs LangChain comparison.

Both pipelines use:
  - Wide-net retrieval (k=20)
  - Cohere cross-encoder reranking (top_n=5)
  - Soft-abstention QA prompt

Each framework reads its own native Pinecone index:
  - LlamaIndex  → sec-filings        (ingested by ingest.py)
  - LangChain   → sec-filings-lc     (ingested by ingest_langchain.py)

Usage:
    python compare_frameworks.py

Outputs:
    - Comparison table to console
    - framework_comparison.json with full results
"""

import os
import json
import time

# ---------------------------------------------------------------------------
# SHARED CONFIG
# ---------------------------------------------------------------------------
QUESTIONS = [
    "What are the main risk factors?",
    "What cybersecurity risks does the company face?",
    "How does competition affect the business?",
    "What regulatory risks are mentioned?",
    "What macroeconomic risks could impact performance?",
]

RETRIEVE_K = 20      # wide net for the cross-encoder reranker
RERANK_TOP_N = 5     # final chunks fed to the LLM
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# Cohere free tier = 10 reranker calls/minute. Pace queries to stay under it.
COHERE_SLEEP_SEC = 6

LLAMAINDEX_INDEX = os.environ.get("PINECONE_INDEX", "sec-filings")
LANGCHAIN_INDEX = os.environ.get("PINECONE_INDEX_LC", "sec-filings-lc")

# Soft-abstention prompt body (shared rules; placeholders differ by framework)
QA_RULES = (
    "You are a financial document analyst. Answer using ONLY the context below.\n"
    "Rules:\n"
    "  1. If the answer is fully absent from the context: respond with exactly "
    "'This information is not available in the provided documents.'\n"
    "  2. If only partial information is available: answer what you can, "
    "then explicitly note what is missing.\n"
    "  3. If the question requires inference or reasoning from the context: "
    "make the inference explicitly and cite the specific evidence you used.\n"
    "  4. Never use outside knowledge. Never guess beyond what the context states.\n"
)
QA_PROMPT_LI = QA_RULES + "\nContext:\n{context_str}\n\nQuestion: {query_str}\nAnswer: "
QA_PROMPT_LC = QA_RULES + "\nContext:\n{context}\n\nQuestion: {question}\nAnswer: "


# ---------------------------------------------------------------------------
# LLAMAINDEX PIPELINE — k=20 retrieve → Cohere rerank to 5 → abstention prompt
# ---------------------------------------------------------------------------
def run_llamaindex():
    print("🔵 Running LlamaIndex pipeline (k=20 → rerank top 5 → abstention prompt)...")
    from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.postprocessor.cohere_rerank import CohereRerank
    from pinecone import Pinecone

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(LLAMAINDEX_INDEX)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    reranker = CohereRerank(
        api_key=os.environ["COHERE_API_KEY"],
        model=COHERE_RERANK_MODEL,
        top_n=RERANK_TOP_N,
    )
    qa_prompt = PromptTemplate(QA_PROMPT_LI)

    query_engine = index.as_query_engine(
        similarity_top_k=RETRIEVE_K,
        node_postprocessors=[reranker],
        text_qa_template=qa_prompt,
    )

    results = []
    for i, q in enumerate(QUESTIONS):
        response = query_engine.query(q)
        nodes = response.source_nodes
        top_score = nodes[0].score if nodes else 0
        sources = sorted(set(
            n.metadata.get("source", n.metadata.get("file_name", "unknown"))
            for n in nodes
        ))
        answer = str(response).strip()

        results.append({
            "question": q,
            "top_score": round(top_score, 3),
            "sources": sources,
            "answer": answer,
        })
        print(f"  ✓ Q: {q[:50]}... Rerank top score: {top_score:.3f}")

        if i < len(QUESTIONS) - 1:
            time.sleep(COHERE_SLEEP_SEC)

    return results


# ---------------------------------------------------------------------------
# LANGCHAIN PIPELINE — k=20 retrieve → Cohere rerank to 5 → abstention prompt
# ---------------------------------------------------------------------------
def run_langchain():
    print("\n🟢 Running LangChain pipeline (k=20 → rerank top 5 → abstention prompt)...")
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_cohere import CohereRerank
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=LANGCHAIN_INDEX,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_K},
    )
    reranker = CohereRerank(
        cohere_api_key=os.environ["COHERE_API_KEY"],
        model=COHERE_RERANK_MODEL,
        top_n=RERANK_TOP_N,
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    prompt = ChatPromptTemplate.from_template(QA_PROMPT_LC)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer_chain = prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', doc.metadata.get('file_name', 'unknown'))}]\n{doc.page_content}"
            for doc in docs
        )

    results = []
    for i, q in enumerate(QUESTIONS):
        docs = compression_retriever.invoke(q)
        top_score = docs[0].metadata.get("relevance_score", 0) if docs else 0
        sources = sorted(set(
            doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
            for doc in docs
        ))

        answer = answer_chain.invoke({"context": format_docs(docs), "question": q})

        results.append({
            "question": q,
            "top_score": round(top_score, 3),
            "sources": sources,
            "answer": answer.strip(),
        })
        print(f"  ✓ Q: {q[:50]}... Rerank top score: {top_score:.3f}")

        if i < len(QUESTIONS) - 1:
            time.sleep(COHERE_SLEEP_SEC)

    return results


# ---------------------------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------------------------
def print_comparison(li_results, lc_results):
    print("\n" + "=" * 120)
    print("FRAMEWORK COMPARISON — k=20 retrieve → Cohere rerank top 5 → abstention prompt")
    print("Scores below are Cohere rerank relevance (0-1). Comparable across frameworks.")
    print("=" * 120)

    header = f"| {'Q#':<3} | {'LlamaIndex Rerank':<17} | {'LangChain Rerank':<16} | {'Δ':<8} | {'LlamaIndex Sources':<30} | {'LangChain Sources':<30} | {'Same?':<5} |"
    separator = "|" + "-" * 5 + "|" + "-" * 19 + "|" + "-" * 18 + "|" + "-" * 10 + "|" + "-" * 32 + "|" + "-" * 32 + "|" + "-" * 7 + "|"

    print(separator)
    print(header)
    print(separator)

    for i, (li, lc) in enumerate(zip(li_results, lc_results), 1):
        li_src = ", ".join(li["sources"])
        lc_src = ", ".join(lc["sources"])
        delta = li["top_score"] - lc["top_score"]
        same = "✅" if set(li["sources"]) == set(lc["sources"]) else "❌"

        row = f"| Q{i:<2} | {li['top_score']:<17.3f} | {lc['top_score']:<16.3f} | {delta:<+8.3f} | {li_src:<30} | {lc_src:<30} | {same:<5} |"
        print(row)

    print(separator)

    # Answer comparison
    print("\n" + "=" * 120)
    print("ANSWER COMPARISON")
    print("=" * 120)

    for i, (li, lc) in enumerate(zip(li_results, lc_results), 1):
        print(f"\nQ{i}: {li['question']}")
        print("-" * 80)
        print(f"  LlamaIndex ({len(li['answer'])} chars): {li['answer'][:200]}...")
        print(f"  LangChain  ({len(lc['answer'])} chars): {lc['answer'][:200]}...")

    # Summary stats
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    li_avg = sum(r["top_score"] for r in li_results) / len(li_results)
    lc_avg = sum(r["top_score"] for r in lc_results) / len(lc_results)
    li_avg_len = sum(len(r["answer"]) for r in li_results) / len(li_results)
    lc_avg_len = sum(len(r["answer"]) for r in lc_results) / len(lc_results)
    source_match = sum(
        1 for li, lc in zip(li_results, lc_results)
        if set(li["sources"]) == set(lc["sources"])
    )

    print(f"  Avg rerank score:    LlamaIndex {li_avg:.3f}  |  LangChain {lc_avg:.3f}  |  Δ {li_avg - lc_avg:+.3f}")
    print(f"  Avg answer length:   LlamaIndex {li_avg_len:.0f} chars  |  LangChain {lc_avg_len:.0f} chars")
    print(f"  Source agreement:    {source_match}/{len(li_results)} questions retrieved same sources")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Framework Comparison: LlamaIndex vs LangChain (optimized)")
    print(f"  LlamaIndex index: {LLAMAINDEX_INDEX}   |   LangChain index: {LANGCHAIN_INDEX}")
    print(f"  retrieve k={RETRIEVE_K}, rerank top_n={RERANK_TOP_N}, model=gpt-4o-mini")
    print(f"  reranker={COHERE_RERANK_MODEL}, embedding=text-embedding-3-small")
    print("=" * 70)

    li_results = run_llamaindex()
    lc_results = run_langchain()

    print_comparison(li_results, lc_results)

    output = {
        "config": {
            "llamaindex_index": LLAMAINDEX_INDEX,
            "langchain_index": LANGCHAIN_INDEX,
            "embedding": "text-embedding-3-small",
            "llm": "gpt-4o-mini",
            "reranker": COHERE_RERANK_MODEL,
            "retrieve_k": RETRIEVE_K,
            "rerank_top_n": RERANK_TOP_N,
            "score_meaning": "Cohere rerank relevance (comparable across frameworks)",
        },
        "llamaindex": li_results,
        "langchain": lc_results,
    }

    with open("framework_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n💾 Full results saved to framework_comparison.json")


if __name__ == "__main__":
    main()
