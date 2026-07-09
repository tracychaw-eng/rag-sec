"""Compare two metrics reports produced by eval.harness.

Usage:
    python -m eval.compare baseline-legacy-fullcorpus secrag-v1
"""

import json
import sys
from pathlib import Path


def load(label: str) -> dict:
    return json.loads(Path(f"eval/metrics_{label}.json").read_text(encoding="utf-8"))


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  —  "


def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.3f}"
    return ""


def main():
    label_a, label_b = sys.argv[1], sys.argv[2]
    a, b = load(label_a), load(label_b)

    print(f"{'':<34} {label_a[:18]:>18} {label_b[:18]:>18} {'delta':>8}")
    print("-" * 82)

    print("RETRIEVAL (source-level)")
    cats = sorted(set(a["retrieval"]) | set(b["retrieval"]))
    for cat in cats:
        for metric in ("source_recall@K", "source_precision@K"):
            va = a["retrieval"].get(cat, {}).get(metric)
            vb = b["retrieval"].get(cat, {}).get(metric)
            print(f"  {cat + ' ' + metric.split('_')[1]:<32} "
                  f"{fmt(va):>18} {fmt(vb):>18} {delta(va, vb):>8}")

    if a.get("ragas") and b.get("ragas"):
        print("\nRAGAS")
        scopes = [s for s in ("overall", "factual", "reasoning")
                  if s in a["ragas"] and s in b["ragas"]]
        for scope in scopes:
            for metric in ("faithfulness", "answer_relevancy",
                           "context_recall", "context_precision"):
                va = a["ragas"][scope].get(metric)
                vb = b["ragas"][scope].get(metric)
                print(f"  {scope + ' ' + metric:<32} "
                      f"{fmt(va):>18} {fmt(vb):>18} {delta(va, vb):>8}")

    for label, rep in ((label_a, a), (label_b, b)):
        if rep.get("abstention"):
            n = sum(rep["abstention"].values())
            print(f"\n{label} adversarial abstention: {n}/{len(rep['abstention'])} "
                  f"(gold labels stale — see report note)")


if __name__ == "__main__":
    main()
