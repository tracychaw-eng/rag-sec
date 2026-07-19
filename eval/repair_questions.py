"""Dataset v5 QA pass — audit generated questions for the two defect
classes found during the quality push, regenerate the defective ones in
place, and write dataset v5.

Defect classes:
  1. Ambiguous-scope numeric questions: multiple legitimate rows answer
     them (reported vs managed basis, carrying value vs contractual
     amount). Detected by an LLM ambiguity audit + known ids.
  2. Speculative reasoning framings ("how might X affect Y") whose
     reference answers speculate beyond the filing — a faithfulness
     rubric penalizes any engagement. Detected by regex.

Repaired questions keep their id, split, category, and tags — the
dev/holdout membership is unchanged, and regeneration is fully automated
(new prompts + validation gates), so holdout questions remain
never-tuned-on.

Usage:
    python -m eval.repair_questions            # audit only (report)
    python -m eval.repair_questions --write    # audit + repair + write v5
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from sec_rag.config import get_settings
from sec_rag.ingestion.store import ChunkStore
from sec_rag.retrieval.retriever import HybridRetriever

from .generate_questions import SPECULATIVE_Q_RE, _author, _validate

# Confirmed defective during the quality push (docs/phase3.md)
KNOWN_DEFECTIVE = {"N105", "N107", "R102", "R103", "R104"}

AMBIGUITY_AUDIT_PROMPT = """You audit evaluation questions for a SEC-filings QA system.

Given a question about a financial figure and rows from the SAME filing
that resemble its subject, decide whether MORE THAN ONE distinct figure is
a defensible answer to the question exactly as worded (e.g. the filing
shows the line item on both a reported and a managed basis, or as both a
carrying value and a contractual amount, and the question does not say
which).

Return JSON: {"ambiguous": true/false, "reason": "one short sentence"}"""

MAX_ATTEMPTS = 4


def _related_rows(question: str, parents, max_rows: int = 40) -> list[str]:
    """Table rows from the question's filing sharing vocabulary with it."""
    words = {w for w in question.casefold().split() if len(w) > 4}
    rows = []
    for p in parents:
        for ln in p.text.splitlines():
            if ln.lstrip().startswith("|") and any(c.isdigit() for c in ln):
                if sum(1 for w in words if w in ln.casefold()) >= 2:
                    rows.append(ln.strip()[:160])
    # dedupe, cap
    seen, out = set(), []
    for r in rows:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out[:max_rows]


async def audit(questions, store, client) -> dict[str, str]:
    """qid -> reason for every flagged question."""
    flagged: dict[str, str] = {}
    parents_by_filing: dict = {}
    for p in store.load_parents():
        parents_by_filing.setdefault((p.ticker, p.filing_date), []).append(p)

    for q in questions:
        qid, tags = q["id"], q.get("tags") or []
        if qid in KNOWN_DEFECTIVE:
            flagged[qid] = "known defective (quality-push diagnosis)"
            continue
        if not tags:
            continue        # hand-written legacy/adversarial: out of scope

        if SPECULATIVE_Q_RE.search(q["question"]) and tags[0] in (
                "reasoning", "multihop"):
            flagged[qid] = "speculative framing (how might/could)"
            continue

        if tags[0] in ("numeric", "multiyear"):
            src_chunks = q.get("source_chunks") or []
            parents = []
            for cid in src_chunks:
                ticker, _, rest = cid.partition("_")
                date = rest.split("_it")[0]
                parents.extend(parents_by_filing.get((ticker, date), []))
            rows = _related_rows(q["question"], parents)
            if len(rows) < 2:
                continue
            resp = await client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": AMBIGUITY_AUDIT_PROMPT},
                    {"role": "user", "content":
                        f"Question: {q['question']}\n"
                        f"Reference answer: {q['reference_answer'][:200]}\n\n"
                        f"Rows from the same filing:\n" + "\n".join(rows)},
                ])
            try:
                verdict = json.loads(resp.choices[0].message.content)
                if verdict.get("ambiguous") is True:
                    flagged[qid] = f"ambiguous: {verdict.get('reason', '')[:90]}"
            except (json.JSONDecodeError, AttributeError):
                pass
    return flagged


async def regenerate(q: dict, reason: str, store, retriever, client) -> bool:
    """Repair one question in place. Returns True on success."""
    qtype = (q.get("tags") or ["factual"])[0]
    parents_by_id = store.parents_by_id()
    chunks = [parents_by_id[cid] for cid in q.get("source_chunks", [])
              if cid in parents_by_id]
    if not chunks:
        return False

    for _ in range(MAX_ATTEMPTS):
        data = await _author(client, qtype, chunks)
        if data is None:
            continue
        if await _validate(data, chunks, qtype, retriever):
            continue        # rejection reason returned -> try again
        q["question"] = data["question"]
        q["reference_answer"] = data["reference_answer"]
        q["notes"] = (f"Repaired in dataset v5 ({reason}); regenerated with "
                      f"hardened prompts, key facts verified verbatim.")
        return True
    return False


async def main_async(write: bool) -> None:
    cfg = get_settings()
    store = ChunkStore(cfg.store_dir)
    client = AsyncOpenAI(api_key=cfg.openai_api_key)

    path = Path("evaluation_dataset.json")
    ds = json.loads(path.read_text(encoding="utf-8"))
    questions = ds["questions"]

    flagged = await audit(questions, store, client)
    print(f"AUDIT: {len(flagged)} flagged of "
          f"{sum(1 for q in questions if q.get('tags'))} generated questions")
    for qid, reason in sorted(flagged.items()):
        q = next(x for x in questions if x["id"] == qid)
        print(f"  {qid} [{(q.get('tags') or ['?'])[0]}] {reason}")
        print(f"     {q['question'][:100]}")

    if not write:
        print("\n(audit only — rerun with --write to repair)")
        return

    retriever = HybridRetriever(cfg)
    repaired, dropped = [], []
    for qid, reason in sorted(flagged.items()):
        q = next(x for x in questions if x["id"] == qid)
        ok = await regenerate(q, reason, store, retriever, client)
        (repaired if ok else dropped).append(qid)
        status = "✓ repaired" if ok else "✗ could not repair"
        print(f"  {qid}: {status}")
        if ok:
            print(f"     new: {q['question'][:100]}")

    ds["metadata"]["version"] = 5
    ds["metadata"]["revision_note"] = (
        ds["metadata"]["revision_note"]
        + f" v5 (2026-07-18): QA pass repaired {len(repaired)} defective "
          f"generated questions in place ({', '.join(repaired)}) — "
          "ambiguous-scope numerics and speculative reasoning framings; "
          "ids/splits unchanged.")
    path.write_text(json.dumps(ds, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\nDataset v5 written: {len(repaired)} repaired, "
          f"{len(dropped)} unrepairable (left as-is: {dropped or 'none'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args.write))


if __name__ == "__main__":
    main()
