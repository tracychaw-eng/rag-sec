"""Threshold gate for eval reports — fails (exit 1) when any metric drops
below the floor in eval/thresholds.json.

Used by the nightly workflow so a quality regression turns the run red
(and pages whoever watches CI) instead of drifting silently.

Usage:
    python -m eval.check_thresholds --label nightly
    python -m eval.check_thresholds --metrics eval/metrics_secrag-v3.json
"""

import argparse
import json
import sys
from pathlib import Path


def check(metrics: dict, thresholds: dict) -> list[str]:
    """Returns a list of violations (empty = pass)."""
    violations = []

    # Floors are calibrated per dataset version; gating a report from a
    # different exam is meaningless and must fail loudly.
    report_ds = metrics.get("dataset_version")
    floor_ds = thresholds.get("dataset_version")
    if floor_ds is not None and report_ds is not None and report_ds != floor_ds:
        violations.append(
            f"dataset_version mismatch: report graded on dataset v{report_ds}, "
            f"thresholds calibrated for v{floor_ds} — recalibrate "
            f"eval/thresholds.json before gating")

    ragas_overall = (metrics.get("ragas") or {}).get("overall") or {}
    for name, floor in thresholds.get("ragas", {}).items():
        value = ragas_overall.get(name)
        if value is None:
            violations.append(f"ragas.{name}: missing from report")
        elif value < floor:
            violations.append(f"ragas.{name}: {value} < {floor}")

    retrieval_overall = (metrics.get("retrieval") or {}).get("overall") or {}
    for name, floor in thresholds.get("retrieval", {}).items():
        value = retrieval_overall.get(name)
        if value is None:
            violations.append(f"retrieval.{name}: missing from report")
        elif value < floor:
            violations.append(f"retrieval.{name}: {value} < {floor}")

    floor = thresholds.get("abstention_rate")
    abstention = metrics.get("abstention") or {}
    if floor is not None and abstention:
        rate = sum(abstention.values()) / len(abstention)
        if rate < floor:
            violations.append(f"abstention_rate: {rate:.2f} < {floor}")

    return violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--metrics")
    ap.add_argument("--thresholds", default="eval/thresholds.json")
    args = ap.parse_args()

    if not args.metrics and not args.label:
        ap.error("provide --label or --metrics")
    path = Path(args.metrics or f"eval/metrics_{args.label}.json")

    metrics = json.loads(path.read_text(encoding="utf-8"))
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))

    violations = check(metrics, thresholds)
    if violations:
        print(f"THRESHOLD GATE FAILED ({path}):")
        for v in violations:
            print(f"  ✗ {v}")
        return 1
    print(f"THRESHOLD GATE PASSED ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
