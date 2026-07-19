# rag-sec — Production RAG over SEC 10-K Filings

Ask questions about SEC 10-K filings and get cited, evaluated answers.
Custom hybrid-retrieval pipeline (no RAG framework), FastAPI service,
evaluation-gated development.

```text
"What was NVIDIA's total revenue in its most recent fiscal year?"
→ "NVIDIA's total revenue in fiscal year 2026 was $215,938 million,
   up 65% from $130,497 million [NVDA, Item 7]."     ~6s · ~$0.02
```

(Generation uses gpt-4o by default — measurably better grounding; set
`SECRAG_LLM_MODEL=gpt-4o-mini` for ~$0.004/answer at lower faithfulness.)

**Corpus:** full 10-K filings (all Items, tables included), latest 2
fiscal years for MSFT, NVDA, JPM — extendable to any ticker via
`python -m sec_rag.ingestion.sync <TICKERS> --years N`.

## Quality (measured, not claimed)

Scored with RAGAS (LLM-as-judge) on a 90-question dataset with a held-out
slice that is never tuned against (runs `secrag-v9` / `release-v9`,
dataset v4):

| Metric | dev (n=71) | holdout (n=19, never tuned on) |
| --- | --- | --- |
| faithfulness | 0.901 | 0.821¹ |
| answer relevancy | 0.825 | 0.950 |
| context recall | 0.873 | 0.867 |
| source recall@K | **1.000** | **1.000** |
| source precision@K | 0.891 | 0.821 |

¹ On the questions scored in both the before/after runs, holdout
faithfulness is statistically flat (0.86–0.89 at n=12); the point drop is
a composition effect — previously-failed retrievals abstained and were
excluded from the average, and now get answered and graded. Details in
`docs/phase3.md` (“Production quality push”).

Reports: `eval/metrics_*.json` (each stamped with its `dataset_version`);
regression floors: `eval/thresholds.json`, enforced nightly in CI.

## Architecture

```text
INGESTION (offline, run when the corpus should change)
  EDGAR sync ─▶ Item-aware parse (tables → pipe rows) ─▶ parent/child chunks
  (paged API)   ─▶ chunk store (JSONL per filing) + embeddings ─▶ Qdrant

QUERY (per request, fully async)
  question ─▶ planner (intent / tickers / years / rewrite+paraphrases)
           ─▶ hybrid retrieval: dense (Qdrant) ∥ BM25 ─▶ RRF fusion
           ─▶ class-aware rerank (Cohere; none for reasoning;
              per-ticker fan-out for comparisons — in parallel)
           ─▶ child→parent expansion ─▶ cited generation (or SSE stream)

CROSS-CUTTING
  answer/embedding caches · per-key rate limits · chat sessions ·
  user memory  ── in-process, or Redis via SECRAG_REDIS_URL
  tracing + $-cost per request ─▶ JSONL / Langfuse / OpenTelemetry
  online judge: sampled production answers scored for faithfulness
```

Retrieval policy encodes measured findings: rerankers hurt multi-hop
reasoning (skip them there); per-ticker filtered retrieval is the only
source-diversity guarantee for comparisons; reranker score floors need a
`min_keep` guard. Full history and rationale: `docs/phase1.md` →
`docs/phase3.md`.

## Quickstart

Prereqs: Python 3.12+, Docker, `OPENAI_API_KEY`, `COHERE_API_KEY` in env.

```powershell
python -m venv venv; venv\Scripts\activate
pip install -e ".[dev]"

# 1. Build the corpus (offline; embeds ~$0.05; skips filings it has)
python -m sec_rag.ingestion.sync MSFT NVDA JPM --years 2

# 2. Run the stack (API + Redis + Qdrant server)
docker compose up --build -d
#    one-time: copy local vectors into the server (no re-embedding)
python -m sec_rag.ingestion.migrate_qdrant --to http://localhost:6333

# 3. Ask — open the chat UI in your browser:
#    http://localhost:8000        (API key: dev-key)
#    …or hit the API directly:
curl -s -X POST localhost:8000/query -H "X-API-Key: dev-key" `
  -H "Content-Type: application/json" `
  -d '{"question": "How do MSFT and JPM differ on cybersecurity risk?"}'
```

No Docker? `uvicorn sec_rag.api.app:app --port 8000` serves from the
embedded Qdrant (`qdrant_db/`) directly — same API and UI, single process.

## Chat UI

`http://localhost:8000/` serves a built-in chat page (single
self-contained HTML file, no separate frontend to run):

- **streaming answers** — tokens render as they generate, with a
  retrieval indicator while the search runs
- **source chips** per answer (ticker + Item, e.g. `NVDA Item 7`) plus
  latency and dollar cost
- **real conversations** — follow-ups like "how does that compare to
  Microsoft?" resolve against the previous turn; *New chat* resets
- API-key field (persisted in the browser), light/dark follows your
  system theme

It talks to the same authenticated endpoints as any other client — no
special server-side state. The interactive API reference (Swagger) stays
at `http://localhost:8000/docs`.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | built-in chat UI (see above) |
| `GET /health` | readiness, corpus/filings, cache hit rates, judge stats |
| `POST /query` | single-shot answer + citations, provenance, latency, cost |
| `POST /chat/{session}` | multi-turn (follow-ups condensed to standalone queries) |
| `POST /chat/{session}/stream` | same, SSE token streaming (`meta → delta* → done`) |
| `DELETE /chat/{session}` | clear session |
| `GET/DELETE /memory/{user}` | view / erase per-user memory (governance) |

Auth: `X-API-Key` when `SECRAG_API_KEYS` is set (compose default:
`dev-key`). Send `X-User-Id` on `/chat` to enable per-user memory —
durable preferences ("prefers bullet points") extracted with confidence
gating and applied in later sessions; fully visible and erasable.

## Configuration

Everything lives in [src/sec_rag/config.py](src/sec_rag/config.py)
(`SECRAG_*` env vars). The ones that matter first:

| Variable | Default | Effect |
| --- | --- | --- |
| `SECRAG_LLM_MODEL` | gpt-4o | generation model (`gpt-4o-mini` = ~6x cheaper, lower faithfulness) |
| `SECRAG_REDIS_URL` | unset | Redis for caches/sessions/rate limits (needed for >1 replica) |
| `SECRAG_QDRANT_URL` | unset | Qdrant server instead of embedded local mode |
| `SECRAG_API_KEYS` | unset | comma-separated keys; unset = auth off (dev only) |
| `SECRAG_RATE_LIMIT_PER_MINUTE` | 60 | per-key sliding/fixed window |
| `SECRAG_JUDGE_SAMPLE_RATE` | 0 | fraction of answers faithfulness-scored in production |
| `LANGFUSE_PUBLIC_KEY`/`_SECRET_KEY` | unset | trace export to Langfuse |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | trace export as OpenTelemetry spans |

Without Langfuse/OTel, traces append to `logs/traces.jsonl` (spans,
tokens, dollar cost per request).

## Evaluation

The dataset (`evaluation_dataset.json`, v4) has 90 questions across
factual / numeric / multi-year / cross-company multi-hop / reasoning /
adversarial, with a **held-out split reserved for release evaluation**
(never tune against it — see `holdout_policy` in the dataset metadata).

```powershell
python -m pytest tests/unit -q                     # 57 tests, no keys needed
python -m eval.smoke_gate                          # deterministic CI gate
python -m eval.run_secrag --label myrun --split dev    # full RAGAS run
python -m eval.compare secrag-v4 myrun             # diff two runs
python -m eval.check_thresholds --label myrun      # regression floors
python -m eval.generate_questions                  # grow the dataset
                                                   #   (corpus-grounded)
```

CI (`.github/workflows/`): lint + tests + docker build per push; image
publish to GHCR on master; nightly RAGAS run gated by thresholds with
Slack alerting; weekly EDGAR sync. Workflows need repo secrets
(`OPENAI_API_KEY`, `COHERE_API_KEY`) once pushed to GitHub.

## Deployment

- `docker-compose.yml` — API + Redis + Qdrant, single host.
- `deploy/k8s.yaml` — 2 API replicas + Redis + Qdrant server + PVCs
  (replica-safe: all shared state is in Redis/Qdrant).
- Image is **code-only**: corpus mounts at runtime, so re-ingesting never
  requires a rebuild and rebuilding never requires re-embedding.

## Layout

```text
src/sec_rag/
├── config.py          all tunables (pydantic-settings)
├── pipeline.py        plan → retrieve → generate (+cache/trace/judge)
├── ingestion/         EDGAR client, parser, chunker, store, sync, migration
├── retrieval/         planner, BM25, RRF fusion, reranker, hybrid retriever
├── generation/        cited, intent-specific prompts (PROMPT_VERSION)
├── api/               FastAPI app (auth, rate limits, SSE, memory)
│                        + ui.html (built-in chat page served at /)
├── memory.py          per-user semantic/episodic memory + governance
└── observability/     tracing+cost, JSON logs, online judge
eval/                  harness, runners, thresholds, question generation
docs/                  phase1–3 design docs · legacy-readme.md (original
                       LlamaIndex implementation journal this project grew from)
```

## History

This project began as a LlamaIndex implementation journal (preserved in
[docs/legacy-readme.md](docs/legacy-readme.md)) and was redesigned into a
framework-free production system in three phases — quality first
(hybrid retrieval, eval harness), then hardening (async, caching, auth,
Docker/CI, observability), then scale-out (Qdrant server, multi-year
corpus, tables, memory, online judging). The phase docs record what was
measured at each step, including the fixes that mattered most: the
original "low" scores were largely measurement artifacts, reranker score
floors starve retrieval without a min-keep guard, and RAGAS zeroes
conditional prose unless answers open affirmatively.
