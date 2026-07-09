# Phase 3 — Advanced enterprise features (2026-07-09)

Builds on Phases 1–2 (`docs/phase1.md`, `docs/phase2.md`).

## 1. Scale-out retrieval: Qdrant server + replicas > 1

- `SECRAG_QDRANT_URL` switches both ingestion and retrieval from embedded
  Qdrant (file-locked, single process) to a Qdrant server.
- `python -m sec_rag.ingestion.migrate_qdrant --to http://host:6333`
  migrates existing vectors without re-embedding (scroll → upsert).
- Compose now runs Qdrant as a service; k8s runs the API at `replicas: 2`
  with the chunk store mounted read-only (`ReadWriteMany`), Qdrant and
  Redis as their own Deployments. Every piece of per-process state
  (caches, sessions, rate limits, vectors) now has a shared backend.

## 2. Multi-year corpus + scheduled EDGAR sync

- EDGAR client ported into the package (`ingestion/edgar.py`) with
  `list_10k_filings(ticker, count=N)` — latest N annual filings.
- Chunk store re-keyed to `(ticker, filing_date)`: years coexist,
  re-ingesting one filing never clobbers another; Qdrant idempotency
  filter is now (ticker AND filing_date), payload carries `filing_year`.
- `python -m sec_rag.ingestion.sync --years 2` downloads + ingests only
  what's missing; `.github/workflows/edgar-sync.yml` runs it weekly
  (needs `SECRAG_QDRANT_URL` secret to hit a real index).
- The query planner extracts `years` when a question restricts to
  specific filing years ("in the 2025 10-K", "year over year") and the
  filter flows through dense (payload `filing_year`) and BM25 alike.
  Chunk context headers already carry the filing date, so mixed-year
  contexts stay attributable.

## 3. Table-aware extraction + numeric QA

- The parser renders each HTML `<table>` as pipe-delimited rows
  ("| Revenue | 130,497 | 60,922 |") before text extraction — previously
  `get_text()` scattered each cell onto its own line, which is why
  numeric questions failed. Empty iXBRL spacer cells are dropped.
- The chunker keeps table rows on their own lines inside chunks (prose
  joins with spaces, rows with newlines).
- Dataset v3 adds numeric factual questions (F006–F007) whose reference
  answers were verified against the ingested corpus text.

## 4. Semantic/episodic user memory with governance

- `memory.py`: per-user semantic facts (role, preferences, focus areas)
  extracted from chat turns by a background LLM pass, plus an episodic
  event log. In-process or Redis (same `SECRAG_REDIS_URL` switch).
- Governance enforced in code: extraction stores facts about the USER
  only (never document content — the pollution rule from the
  architecture review); every fact carries provenance (session,
  timestamp) and confidence; writes below `SECRAG_MEMORY_CONFIDENCE_FLOOR`
  (0.7) are dropped; case-insensitive dedupe; capped at 50 facts.
- Fully user-visible and erasable: `GET /memory/{user_id}` and
  `DELETE /memory/{user_id}`.
- Wire-in: send `X-User-Id` on `/chat` — facts become a preamble for the
  planner and generator ("prefers tables" actually changes output).
  Personalized answers are never cached (answer cache bypassed whenever
  user context is present).

## 5. Online judge sampling

- `observability/judge.py`: with `SECRAG_JUDGE_SAMPLE_RATE` (default 0),
  a sampled fraction of production answers is scored for faithfulness by
  an LLM pass (claims → supported claims) as a fire-and-forget task —
  zero added latency to the user request.
- Results append to `logs/judge.jsonl`; a rolling window (last 200)
  surfaces on `GET /health` as `judge.faithfulness_avg`; scores < 0.5
  emit a structured warning log. This catches hallucination drift
  between nightly evals.

## Verification (2026-07-09)

- **Corpus**: 6 filings (2 fiscal years × MSFT/NVDA/JPM), 645 parents /
  4,272 children — ~2.1x the Phase 2 corpus. JPM's prior-year 10-K
  required paging EDGAR's older submission batches (high-volume filers
  push old 10-Ks out of the "recent" window).
- **Migration**: 4,272 vectors embedded→server with zero re-embedding;
  smoke gate passed against the server (`SECRAG_QDRANT_URL`).
- **Numeric QA e2e**: "NVIDIA total revenue?" → "$215,938M, up 65% from
  $130,497M" straight from income-statement table rows.
- **Year filter e2e**: "According to the fiscal 2024 10-K..." →
  plan.years=[2024], all contexts from the 2024-07-30 filing only.
- **Memory e2e**: facts extracted with confidence + provenance; bullet
  preference applied in a NEW session without restating it;
  DELETE /memory erased everything.
- **Judge e2e**: sampled answer scored (faithfulness_avg on /health).
- **Eval v4 vs v3** (note: corpus doubled, dataset grew to 22 questions):
  faithfulness .930→.962, relevancy .852→.878, context_recall .833→.945
  (table extraction recovered the reasoning-recall soft spot .667→.867),
  context_precision .933→.863 (mixed-year contexts include rows the
  single-year gold labels don't credit — dataset refinement, not a
  retrieval bug). Source-level recall stayed 1.0 across the board
  despite 2x distractors. F006/F007 numeric: faithfulness 1.0,
  recall 1.0. Threshold gate: PASSED. Abstention: 5/5.
- 57 unit tests, ruff clean.
