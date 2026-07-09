"""Run the evaluation dataset through the new sec_rag pipeline and score it
with the same harness used for the legacy baseline.

Usage:
    python -m eval.run_secrag                 # full run + RAGAS
    python -m eval.run_secrag --no-ragas      # retrieval metrics only
    python -m eval.run_secrag --only F001 R002
"""

import argparse
import json
import time
from pathlib import Path

from sec_rag.pipeline import RAGPipeline

from .harness import evaluate_results

LEGACY_SOURCE_TO_TICKER = {
    "msft_10k.txt": "MSFT",
    "nvda_10k.txt": "NVDA",
    "jpm_10k.txt": "JPM",
}


def expected_tickers_for(item: dict) -> list:
    sf = item.get("source_file", "")
    if not sf or sf == "none":
        return []
    return [LEGACY_SOURCE_TO_TICKER.get(s.strip(), s.strip().upper())
            for s in sf.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--label", default="secrag-v1")
    ap.add_argument("--split", choices=["dev", "holdout", "all"],
                    default="all",
                    help="holdout is for release evaluation only — never "
                         "tune against it")
    args = ap.parse_args()

    dataset = json.loads(Path("evaluation_dataset.json").read_text(encoding="utf-8"))
    questions = dataset["questions"]
    if args.split != "all":
        questions = [q for q in questions
                     if q.get("split", "dev") == args.split]
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]

    pipeline = RAGPipeline()
    results = []

    for i, item in enumerate(questions, 1):
        print(f"[{item['id']}] ({i}/{len(questions)}) {item['question'][:70]}...")
        t0 = time.time()
        ans = pipeline.answer_sync(item["question"])
        elapsed = time.time() - t0

        print(f"  intent={ans.plan.intent} tickers={ans.plan.tickers} "
              f"engine={ans.engine_used} contexts={len(ans.contexts)} "
              f"({elapsed:.1f}s)")
        print(f"  A: {ans.text[:140]}...")

        results.append({
            "id": item["id"],
            "category": item["category"],
            "split": item.get("split", "dev"),
            "tags": item.get("tags", []),
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "system_answer": ans.text,
            "engine_used": ans.engine_used,
            "plan": ans.plan.model_dump(),
            "retrieved_tickers": ans.retrieved_tickers,
            "expected_tickers": expected_tickers_for(item),
            "generation_contexts": ans.generation_contexts,
            "retrieved_items": [c.parent.item for c in ans.contexts],
            "latency_s": round(elapsed, 2),
        })
        # Cohere free tier: 10 rerank calls/minute; comparison questions use
        # one call per ticker. Pace conservatively.
        time.sleep(6)

    out = Path(f"eval/results_{args.label}.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw results saved to {out}")

    evaluate_results(results, label=args.label, with_ragas=not args.no_ragas)


if __name__ == "__main__":
    main()
