"""
eval/harness.py — Pipeline-agnostic evaluation harness.

Takes a results JSON produced by any pipeline (legacy evaluate.py or the new
sec_rag pipeline) and computes:

  Retrieval metrics (no LLM needed, deterministic):
    - source_recall@K:    fraction of expected tickers present in retrieved chunks
    - source_precision@K: fraction of retrieved chunks from expected tickers

  RAGAS metrics (LLM-as-judge, gpt-4o-mini):
    - faithfulness, answer_relevancy, context_recall, context_precision
    - run on factual + reasoning questions (comparison contexts are two-stage
      per-source answers; adversarial has no valid ground truth)

Expected record schema (per question):
    id, category, question, reference_answer, system_answer,
    retrieved_tickers: [str, ...]      # one per retrieved chunk, in rank order
    expected_tickers:  [str, ...]      # gold, empty list for adversarial
    generation_contexts: [str, ...]    # texts actually given to the LLM,
                                       # already source-prefixed
    engine_used: str

Usage:
    python -m eval.harness --results eval_results.json --label baseline
    python -m eval.harness --results eval_results.json --label baseline --no-ragas
"""

import argparse
import json
import os
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Retrieval metrics — deterministic, no LLM
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(results: list[dict]) -> dict:
    """Source-level Recall@K and Precision@K against expected tickers."""
    per_question = {}
    for r in results:
        expected = set(r.get("expected_tickers") or [])
        retrieved = r.get("retrieved_tickers") or []
        if not expected:
            # adversarial ("none") — recall/precision undefined
            per_question[r["id"]] = {"recall": None, "precision": None,
                                     "k": len(retrieved)}
            continue
        hit = expected & set(retrieved)
        recall = len(hit) / len(expected)
        precision = (sum(1 for t in retrieved if t in expected) / len(retrieved)
                     if retrieved else 0.0)
        per_question[r["id"]] = {
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "k": len(retrieved),
            "missing_tickers": sorted(expected - hit),
        }

    by_category: dict[str, dict] = {}
    for r in results:
        m = per_question[r["id"]]
        if m["recall"] is None:
            continue
        cat = by_category.setdefault(r["category"], {"recall": [], "precision": []})
        cat["recall"].append(m["recall"])
        cat["precision"].append(m["precision"])

    summary = {
        cat: {
            "source_recall@K": round(sum(v["recall"]) / len(v["recall"]), 3),
            "source_precision@K": round(sum(v["precision"]) / len(v["precision"]), 3),
            "n": len(v["recall"]),
        }
        for cat, v in by_category.items()
    }
    all_r = [m["recall"] for m in per_question.values() if m["recall"] is not None]
    all_p = [m["precision"] for m in per_question.values() if m["precision"] is not None]
    summary["overall"] = {
        "source_recall@K": round(sum(all_r) / len(all_r), 3) if all_r else None,
        "source_precision@K": round(sum(all_p) / len(all_p), 3) if all_p else None,
        "n": len(all_r),
    }
    return {"per_question": per_question, "summary": summary}


# ---------------------------------------------------------------------------
# Abstention check — adversarial questions
# ---------------------------------------------------------------------------
ABSTENTION_PHRASES = [
    "not available in the provided",
    "not mentioned in the provided",
    "not present in the provided",
    "not in the provided",
    "cannot find",
    "no information",
    "not explicitly provided",
    "not specified",
]


def compute_abstention(results: list[dict]) -> dict:
    out = {}
    for r in results:
        if r["category"] != "adversarial":
            continue
        answer = r["system_answer"].lower()
        out[r["id"]] = any(p in answer for p in ABSTENTION_PHRASES)
    return out


# ---------------------------------------------------------------------------
# RAGAS — LLM-as-judge
# ---------------------------------------------------------------------------
def run_ragas(results: list[dict], judge_model: str = "gpt-4o-mini"):
    """Returns (per_question_scores: dict, aggregates: dict)."""
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (Faithfulness, AnswerRelevancy,
                               ContextRecall, ContextPrecision)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    api_key = os.environ["OPENAI_API_KEY"]
    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, api_key=api_key))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key))

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextRecall(llm=llm),
        ContextPrecision(llm=llm),
    ]

    items = [
        r for r in results
        if r["category"] in ("factual", "reasoning")
        and r["reference_answer"] != "NOT IN DOCUMENTS"
        and r.get("generation_contexts")
    ]
    if not items:
        return {}, {}

    ds = Dataset.from_dict({
        "question": [r["question"] for r in items],
        "answer": [r["system_answer"] for r in items],
        "contexts": [r["generation_contexts"] for r in items],
        "ground_truth": [r["reference_answer"] for r in items],
    })
    df = ragas_evaluate(ds, metrics=metrics).to_pandas()

    cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    per_question = {}
    for i, r in enumerate(items):
        row = df.iloc[i]
        per_question[r["id"]] = {
            c: (round(float(row[c]), 3) if row.get(c) is not None
                and row[c] == row[c] else None)  # NaN check
            for c in cols
        }

    def agg(ids):
        out = {}
        for c in cols:
            vals = [per_question[i][c] for i in ids if per_question[i][c] is not None]
            out[c] = round(sum(vals) / len(vals), 3) if vals else None
        return out

    aggregates = {"overall": agg(list(per_question))}
    for cat in ("factual", "reasoning"):
        ids = [r["id"] for r in items if r["category"] == cat]
        if ids:
            aggregates[cat] = agg(ids)
    return per_question, aggregates


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def evaluate_results(results: list[dict], label: str, with_ragas: bool = True,
                     out_dir: Path = Path("eval")) -> dict:
    retrieval = compute_retrieval_metrics(results)
    abstention = compute_abstention(results)

    report = {
        "label": label,
        "date": date.today().isoformat(),
        "n_questions": len(results),
        "retrieval": retrieval["summary"],
        "retrieval_per_question": retrieval["per_question"],
        "abstention": abstention,
        "abstention_note": (
            "Corpus is now full 10-Ks: several adversarial answers (revenue, "
            "headcount, CEO) ARE in the documents. Abstention gold labels are "
            "stale — dataset revision needed before trusting this category."
        ),
    }

    if with_ragas:
        ragas_pq, ragas_agg = run_ragas(results)
        report["ragas"] = ragas_agg
        report["ragas_per_question"] = ragas_pq

    out_path = out_dir / f"metrics_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_report(report, results)
    print(f"\nSaved: {out_path}")
    return report


def _print_report(report: dict, results: list[dict]) -> None:
    print("=" * 68)
    print(f"EVALUATION REPORT — {report['label']}  ({report['date']})")
    print("=" * 68)

    print("\nRETRIEVAL (source-level, vs expected tickers)")
    print(f"  {'category':<14} {'recall@K':<10} {'precision@K':<12} {'n'}")
    for cat, m in report["retrieval"].items():
        print(f"  {cat:<14} {m['source_recall@K']!s:<10} "
              f"{m['source_precision@K']!s:<12} {m['n']}")

    misses = {qid: m for qid, m in report["retrieval_per_question"].items()
              if m.get("missing_tickers")}
    if misses:
        print("\n  Questions with missed sources:")
        for qid, m in misses.items():
            print(f"    {qid}: missing {m['missing_tickers']} "
                  f"(recall={m['recall']})")

    if report.get("ragas"):
        print("\nRAGAS (factual + reasoning)")
        cols = ["faithfulness", "answer_relevancy", "context_recall",
                "context_precision"]
        print(f"  {'scope':<12} " + " ".join(f"{c[:12]:<14}" for c in cols))
        for scope, m in report["ragas"].items():
            print(f"  {scope:<12} " + " ".join(f"{m[c]!s:<14}" for c in cols))

        print("\n  Per-question:")
        cat_by_id = {r["id"]: r["category"] for r in results}
        for qid, m in sorted(report["ragas_per_question"].items()):
            print(f"    {qid} ({cat_by_id.get(qid, '?')[:4]}): "
                  + "  ".join(f"{c[:9]}={m[c]}" for c in cols))

    if report.get("abstention"):
        n_abstained = sum(report["abstention"].values())
        print(f"\nADVERSARIAL abstention: {n_abstained}/{len(report['abstention'])}"
              f"  ⚠ {report['abstention_note'][:60]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-ragas", action="store_true")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    evaluate_results(results, label=args.label, with_ragas=not args.no_ragas)


if __name__ == "__main__":
    main()
