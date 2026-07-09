"""Eval smoke gate — fast, deterministic quality checks for CI.

Runs a small fixed set of questions through the pipeline and asserts
behavior that must never regress (no RAGAS/LLM-judge — those run nightly):

  - source-level retrieval recall = 1.0 on questions with known sources
  - factual answers carry inline citations
  - the historically-hard cases retrieve their target company
  - adversarial questions abstain

Exit code 0 = pass, 1 = fail (gate the deploy on it).

Usage:
    python -m eval.smoke_gate
"""

import json
import re
import sys
from pathlib import Path

from sec_rag.pipeline import RAGPipeline

CITATION_RE = re.compile(r"\[[A-Z]{2,5},?\s*Item\s*\S+\]", re.IGNORECASE)
ABSTAIN_MARKER = "not available in the provided documents"

# (question id, checks) — questions come from evaluation_dataset.json
SMOKE_IDS = {
    "F003": {"expect_tickers": {"JPM"}, "citations": True},
    "F004": {"expect_tickers": {"MSFT"}, "citations": True,
             "must_contain": "28.9"},          # the $28.9B IRS figure
    "R002": {"expect_tickers": {"JPM"}},        # historic lexical-gap failure
    "C001": {"expect_tickers": {"MSFT", "JPM"}},  # source diversity
    "A004": {"abstain": True},                  # out-of-corpus company
}


def main() -> int:
    dataset = json.loads(
        Path("evaluation_dataset.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in dataset["questions"]}

    pipeline = RAGPipeline()
    failures: list[str] = []

    for qid, checks in SMOKE_IDS.items():
        item = questions[qid]
        ans = pipeline.answer_sync(item["question"])
        retrieved = set(ans.retrieved_tickers)
        print(f"[{qid}] engine={ans.engine_used} retrieved={sorted(retrieved)}")

        expect = checks.get("expect_tickers")
        if expect and not expect.issubset(retrieved):
            failures.append(
                f"{qid}: missing sources {sorted(expect - retrieved)}")

        if checks.get("citations") and not CITATION_RE.search(ans.text):
            failures.append(f"{qid}: no inline citation in answer")

        must = checks.get("must_contain")
        if must and must not in ans.text:
            failures.append(f"{qid}: answer lacks required content {must!r}")

        if checks.get("abstain") and ABSTAIN_MARKER not in ans.text.lower():
            failures.append(f"{qid}: expected abstention, got an answer")

    print()
    if failures:
        print("SMOKE GATE FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print(f"SMOKE GATE PASSED ({len(SMOKE_IDS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
