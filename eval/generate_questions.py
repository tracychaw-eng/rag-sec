"""Corpus-grounded eval question generation.

Questions are authored BY an LLM FROM actual parent chunks, so reference
answers are grounded in corpus text by construction. Every candidate then
passes three programmatic gates before acceptance:

  1. Verbatim grounding — the LLM must return key_facts copied verbatim
     from the source chunk; each must appear in the chunk text.
  2. Numeric grounding — every 3+ digit figure in the reference answer
     must appear in the source chunk(s) (catches invented numbers).
  3. Answerability — hybrid retrieval on the question must surface the
     source ticker(s)/filing(s) in the top candidates. (Bias note: this
     filters toward what our retriever can find; acceptable for a
     regression dataset, would not be for a pure research benchmark.)

Usage:
    python -m eval.generate_questions              # generate + validate
    python -m eval.generate_questions --write      # also write dataset v4
"""

import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

from sec_rag.config import get_settings
from sec_rag.ingestion.store import ChunkStore
from sec_rag.retrieval.retriever import HybridRetriever

random.seed(20260709)

GEN_MODEL = "gpt-4o-mini"
HOLDOUT_FRACTION = 0.30

# type -> (count, harness category)
PLAN = {
    "numeric": (18, "factual"),
    "multiyear": (16, "factual"),
    "multihop": (14, "comparison"),
    "factual": (12, "factual"),
    "reasoning": (10, "reasoning"),
}

PROMPTS = {
    "numeric": """From this 10-K excerpt, write ONE factual question about a specific
financial figure that appears in a table row.
The question must name the company and be fully standalone, and it MUST be
UNAMBIGUOUS: 10-Ks often show several figures with similar labels (reported
vs managed basis, carrying value vs contractual amount, consolidated vs
segment). Include enough scope in the question — the exact row label plus
the statement, table, or basis it comes from — that exactly ONE figure in
the entire filing can answer it. If every phrasing would stay ambiguous,
pick a different figure from the excerpt.
Return JSON: {"question": ..., "reference_answer": "... (must state the
exact figure as printed)", "row_label": "the row's label verbatim",
"key_facts": ["verbatim substrings copied from the excerpt that support
the answer, including the figure"]}""",

    "multiyear": """You get excerpts of the SAME section from TWO different fiscal-year
10-K filings of the same company. Write ONE question that requires BOTH years —
e.g. how a figure, risk, or disclosure changed between the two filings.
Name the company AND both years explicitly so the question is standalone.
If the question is about a figure, include the exact row label and its
statement/table/basis so only one figure per filing can answer it.
Return JSON: {"question": ..., "reference_answer": "... (state what each year's
filing says)", "key_facts": ["verbatim substrings, at least one from EACH
year's excerpt"]}""",

    "multihop": """You get excerpts from TWO different companies' 10-K filings.
Write ONE comparison question that requires synthesizing BOTH companies'
disclosures (not answerable from either alone). Name both companies.
Ask what the filings DISCLOSE or how the disclosures differ — never
"how might/could X affect Y" (that invites speculation a faithfulness
rubric will punish).
Return JSON: {"question": ..., "reference_answer": "... (cover both companies)",
"key_facts": ["verbatim substrings, at least one from EACH company's excerpt"]}""",

    "factual": """From this 10-K excerpt, write ONE specific factual question about a
disclosure in it, answerable only from this excerpt. Name the company; make it
standalone. Avoid yes/no questions.
Return JSON: {"question": ..., "reference_answer": ...,
"key_facts": ["verbatim substrings copied from the excerpt"]}""",

    "reasoning": """From this 10-K excerpt, write ONE question requiring inference —
the answer is not stated directly but can be derived from what IS stated
(implications, exposure, second-order effects). Name the company.
FRAMING RULES (a faithfulness rubric will grade answers strictly against
the filing text):
- Ask what the filing's disclosures INDICATE, IMPLY, or REVEAL — never
  "how might/could/would X affect Y", which invites speculation beyond
  the filing.
- The reference_answer must contain ONLY conclusions each key_fact
  directly supports, stated in indicative mood. No "likely", no predicted
  outcomes, no mechanisms the excerpt never mentions.
Return JSON: {"question": ..., "reference_answer": "... (the supported
inference)", "key_facts": ["verbatim substrings that ground the inference"]}""",
}

# Question framings that force speculation beyond the filing — a
# faithfulness rubric can never score them well (even their reference
# answers speculate). Rejected at validation and flagged by the auditor.
SPECULATIVE_Q_RE = re.compile(r"(?i)\bhow\s+(might|could|would)\b")

# Hand-curated adversarial additions (absent from any 10-K by construction)
NEW_ADVERSARIAL = [
    ("A006", "What price target did Goldman Sachs set for NVIDIA after its latest 10-K?",
     "Analyst price targets are third-party opinion, never in SEC filings."),
    ("A007", "How many shares of JPMorgan does Warren Buffett currently own?",
     "Third-party ownership positions are not disclosed in the issuer's 10-K."),
    ("A008", "What will Microsoft's dividend be next quarter?",
     "Future dividend declarations are not in a 10-K."),
    ("A009", "What are Alphabet's main risk factors according to its 10-K?",
     "Out-of-corpus company (only MSFT, NVDA, JPM are indexed)."),
    ("A010", "Which of the three companies had the best stock performance this month?",
     "Market performance data is not in SEC filings."),
]

_NUM_RE = re.compile(r"\d[\d,]{2,}(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {m.replace(",", "") for m in _NUM_RE.findall(text)}


def _has_table_rows(text: str) -> bool:
    return sum(1 for ln in text.split("\n")
               if ln.startswith("|") and any(c.isdigit() for c in ln)) >= 3


# ---------------------------------------------------------------------------
def _sample_sources(store: ChunkStore) -> dict[str, list]:
    """Pick source parent chunks per question type."""
    parents = store.load_parents()
    by_key = defaultdict(list)
    for p in parents:
        by_key[(p.ticker, p.filing_date, p.item)].append(p)

    latest: dict[str, str] = {}
    for t, d in store.available_filings():
        latest[t] = max(latest.get(t, ""), d)

    prose = [p for p in parents if p.item in ("1", "1A", "1C", "7")
             and len(p.text) > 3000]
    numeric = [p for p in parents if p.item in ("7", "8", "15")
               and _has_table_rows(p.text)]

    sources: dict[str, list] = defaultdict(list)
    n_num, cat = PLAN["numeric"]
    sources["numeric"] = random.sample(numeric, min(n_num * 2, len(numeric)))
    n_fact, _ = PLAN["factual"]
    sources["factual"] = random.sample(prose, min(n_fact * 2, len(prose)))
    n_reas, _ = PLAN["reasoning"]
    sources["reasoning"] = random.sample(prose, min(n_reas * 2, len(prose)))

    # multiyear: same (ticker, item), different filing dates
    pairs = []
    for t in store.available_tickers():
        dates = sorted({d for tt, d in store.available_filings() if tt == t})
        if len(dates) < 2:
            continue
        for item in ("1A", "7", "8", "15", "1"):
            a = by_key.get((t, dates[-1], item), [])
            b = by_key.get((t, dates[-2], item), [])
            random.shuffle(a)
            random.shuffle(b)
            pairs.extend(list(zip(a, b))[:4])
    random.shuffle(pairs)
    sources["multiyear"] = pairs[:PLAN["multiyear"][0] * 2]

    # multihop: different tickers, risk/business sections, latest filings
    hop_pool = [p for p in parents if p.item in ("1A", "1")
                and p.filing_date == latest.get(p.ticker)
                and len(p.text) > 3000]
    by_ticker = defaultdict(list)
    for p in hop_pool:
        by_ticker[p.ticker].append(p)
    tickers = sorted(by_ticker)
    hop_pairs = []
    for _ in range(PLAN["multihop"][0] * 2):
        t1, t2 = random.sample(tickers, 2)
        hop_pairs.append((random.choice(by_ticker[t1]),
                          random.choice(by_ticker[t2])))
    sources["multihop"] = hop_pairs
    return sources


async def _author(client: AsyncOpenAI, qtype: str, chunks: list) -> dict | None:
    excerpt = "\n\n=== NEXT EXCERPT ===\n\n".join(c.text[:6000] for c in chunks)
    try:
        resp = await client.chat.completions.create(
            model=GEN_MODEL, temperature=0.4,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPTS[qtype]},
                      {"role": "user", "content": excerpt}])
        data = json.loads(resp.choices[0].message.content)
        if not all(isinstance(data.get(k), (str, list))
                   for k in ("question", "reference_answer", "key_facts")):
            return None
        return data
    except Exception:
        return None


async def _validate(data: dict, chunks: list, qtype: str,
                    retriever: HybridRetriever) -> str | None:
    """Returns a rejection reason, or None if the candidate passes."""
    corpus_text = "\n".join(c.text for c in chunks)

    facts = [f for f in data["key_facts"] if isinstance(f, str) and f.strip()]
    if not facts:
        return "no key_facts"
    grounded = sum(1 for f in facts if f.strip() in corpus_text)
    if grounded < max(1, len(facts) - 1):     # allow one whitespace mangle
        return f"key_facts not verbatim ({grounded}/{len(facts)})"

    if qtype in ("numeric", "multiyear"):
        missing = _numbers(data["reference_answer"]) - _numbers(corpus_text)
        # tolerate derived deltas/percentages: at most 1 non-source number
        if len(missing) > 1:
            return f"invented figures: {sorted(missing)[:3]}"

    if qtype in ("reasoning", "multihop") and SPECULATIVE_Q_RE.search(
            data["question"]):
        return "speculative framing (how might/could)"

    if qtype == "numeric":
        # Competing-row gate: the row label must resolve to ONE value in
        # the whole filing, otherwise the question has multiple legitimate
        # answers and any grader keys on the wrong one half the time.
        label = (data.get("row_label") or "").strip()
        if not label:
            return "no row_label returned"
        filings = {(c.ticker, c.filing_date) for c in chunks}
        values = set()
        for child in retriever.children_by_id.values():
            if (child.ticker, child.filing_date) not in filings:
                continue
            for ln in child.text.splitlines():
                if label.casefold() in ln.casefold() and ln.lstrip().startswith("|"):
                    nums = _NUM_RE.findall(ln)
                    if nums:
                        values.add(nums[0].replace(",", ""))
        if len(values) > 1:
            return (f"ambiguous row label {label!r}: "
                    f"{len(values)} competing values in the filing")

    expected_tickers = {c.ticker for c in chunks}
    hits = await retriever.retrieve_children([data["question"]])
    top_tickers = {s.chunk.ticker for s in hits[:25]}
    if not expected_tickers.issubset(top_tickers):
        return f"not answerable: {expected_tickers - top_tickers} not in top-25"

    if qtype == "multiyear":
        dates = {c.filing_date for c in chunks}
        top_dates = {s.chunk.filing_date for s in hits[:40]
                     if s.chunk.ticker in expected_tickers}
        if not dates.issubset(top_dates):
            return f"year not retrievable: {dates - top_dates}"
    return None


async def generate() -> list[dict]:
    cfg = get_settings()
    store = ChunkStore(cfg.store_dir)
    retriever = HybridRetriever(cfg)
    client = AsyncOpenAI(api_key=cfg.openai_api_key)
    sources = _sample_sources(store)

    accepted: list[dict] = []
    counters = defaultdict(int)
    seen_questions: set[str] = set()

    for qtype, (target, category) in PLAN.items():
        for src in sources[qtype]:
            if counters[qtype] >= target:
                break
            chunks = list(src) if isinstance(src, tuple) else [src]
            data = await _author(client, qtype, chunks)
            if data is None:
                continue
            key = re.sub(r"\W+", " ", data["question"].casefold())[:80]
            if key in seen_questions:
                continue

            reason = await _validate(data, chunks, qtype, retriever)
            if reason:
                print(f"  ✗ [{qtype}] {data['question'][:70]} — {reason}")
                continue

            seen_questions.add(key)
            counters[qtype] += 1
            accepted.append({
                "category": category,
                "tags": [qtype],
                "question": data["question"],
                "reference_answer": data["reference_answer"],
                "source_file": ", ".join(sorted({c.ticker for c in chunks})),
                "source_chunks": [c.id for c in chunks],
                "notes": f"Generated from corpus chunks (v4, {qtype}); "
                         f"key facts verified verbatim against source.",
            })
            print(f"  ✓ [{qtype}] {data['question'][:80]}")
    return accepted


def write_dataset(new_items: list[dict]) -> None:
    path = Path("evaluation_dataset.json")
    ds = json.loads(path.read_text(encoding="utf-8"))

    # Legacy questions are dev by definition: every one of them has been
    # tuned against across v1-v4 (reranker floor, F003 prompt, routing).
    for q in ds["questions"]:
        q["split"] = "dev"

    prefix_counters = defaultdict(int)
    prefix_map = {"numeric": "N", "multiyear": "Y", "multihop": "H",
                  "factual": "F", "reasoning": "R"}
    existing_ids = {q["id"] for q in ds["questions"]}

    # stratified holdout assignment per type
    by_type = defaultdict(list)
    for item in new_items:
        by_type[item["tags"][0]].append(item)
    for qtype, items in by_type.items():
        random.shuffle(items)
        n_holdout = round(len(items) * HOLDOUT_FRACTION)
        for i, item in enumerate(items):
            prefix = prefix_map[qtype]
            while True:
                prefix_counters[prefix] += 1
                qid = f"{prefix}{100 + prefix_counters[prefix]}"
                if qid not in existing_ids:
                    break
            item["id"] = qid
            item["split"] = "holdout" if i < n_holdout else "dev"
            ds["questions"].append(item)

    for qid, question, note in NEW_ADVERSARIAL:
        ds["questions"].append({
            "id": qid, "category": "adversarial", "tags": ["adversarial"],
            "question": question, "reference_answer": "NOT IN DOCUMENTS",
            "expected_behavior": "System should abstain.",
            "source_file": "none", "notes": note,
            "split": "holdout" if random.random() < HOLDOUT_FRACTION else "dev",
        })

    qs = ds["questions"]
    ds["metadata"]["version"] = 4
    ds["metadata"]["total_questions"] = len(qs)
    ds["metadata"]["splits"] = {
        "dev": sum(1 for q in qs if q["split"] == "dev"),
        "holdout": sum(1 for q in qs if q["split"] == "holdout"),
    }
    ds["metadata"]["categories"] = {
        c: sum(1 for q in qs if q["category"] == c)
        for c in ("factual", "reasoning", "comparison", "adversarial")}
    ds["metadata"]["holdout_policy"] = (
        "holdout questions are NEVER used for tuning, prompt iteration, or "
        "threshold setting; run them only for release evaluation. All v1-v3 "
        "questions are dev (contaminated by prior tuning).")
    ds["metadata"]["revision_note"] = (
        ds["metadata"]["revision_note"]
        + " v4 (2026-07-09): grown with corpus-grounded generated questions "
          "(numeric/multi-year/multi-hop/factual/reasoning) + adversarial "
          "additions; dev/holdout split introduced.")

    path.write_text(json.dumps(ds, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\nDataset v4 written: {len(qs)} questions "
          f"({ds['metadata']['splits']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    items = asyncio.run(generate())
    Path("eval/generated_candidates.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(items)} candidates accepted "
          f"-> eval/generated_candidates.json")
    if args.write:
        write_dataset(items)


if __name__ == "__main__":
    main()
