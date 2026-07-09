# Phase 2 — Production hardening (started 2026-07-08)

Builds on the Phase 1 pipeline (`docs/phase1.md`) with the service layer
and cross-cutting production concerns.

## What was added

```
src/sec_rag/
├── api/app.py               # FastAPI: /health, /query, /chat/{session_id}
├── cache.py                  # TTL+LRU caches (CacheBackend = Redis seam)
└── observability/tracing.py  # request traces, token/cost metering,
                              #   Langfuse or local-JSONL export
```

### Retries

- OpenAI calls use the SDK's built-in retry (exponential backoff on
  429/5xx/connection errors), configured via `SECRAG_OPENAI_MAX_RETRIES`
  (default 4) and `SECRAG_OPENAI_TIMEOUT_S` (default 60).
- Cohere reranking retries via tenacity (jittered exponential backoff,
  4 attempts) and then **degrades gracefully to RRF fusion order** —
  a degraded answer beats a failed request.

### Caching

- **Answer cache**: single-shot questions, keyed on
  `(collection, PROMPT_VERSION, normalized question)` — a re-ingest or
  prompt edit can never serve stale answers. TTL 1h, LRU 2048.
- **Query-embedding cache**: keyed on `(model, text)`; multi-query
  retrieval and eval re-runs hit it constantly.
- Both are in-process. `CacheBackend` protocol in `cache.py` is the seam
  for Redis when running multiple replicas.
- Hit rates are exposed on `GET /health`.

### Tracing / cost metering

Every `RAGPipeline.answer()` produces one trace: spans for plan → retrieve
→ generate with wall-clock ms, plus per-LLM-call token counts and dollar
cost (price table in `observability/tracing.py` — update it when models
change) and Cohere rerank search counts.

- **Langfuse**: set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`
  (+ `LANGFUSE_HOST` for self-hosted) and traces export automatically.
- **Fallback**: without keys, traces append to `logs/traces.jsonl` —
  same fields, greppable.
- The trace summary rides on every API response (`latency_ms`, `cost_usd`,
  `trace_id`).

### API

```powershell
uvicorn sec_rag.api.app:app --port 8000
```

- `GET  /health` — readiness, corpus stats, cache hit rates
- `POST /query {"question": ...}` — single-shot; returns answer, intent,
  engine, per-context provenance (ticker/item/score), latency, cost
- `POST /chat/{session_id}` — multi-turn; history feeds the planner's
  rewrite so follow-ups ("how does that compare to Microsoft?") become
  standalone comparison queries
- `DELETE /chat/{session_id}` — clear session

Session history is in-process; move to Redis before scaling past one
replica. Endpoints are sync `def` (threadpool) because the pipeline uses
sync clients — async clients are a Phase 2 follow-up.

## Also in this change set

- **F003 answer style fix**: RAGAS's noncommittal classifier scores
  conditional risk-factor prose as relevancy 0.0 (even the human reference
  answer scored 0.0). The factual prompt now requires an affirmative
  enumerating opening sentence → F003 relevancy 0.0 → 0.975. Prompt
  changes bump `PROMPT_VERSION` (cache invalidation).
- **Evaluation dataset v2**: adversarial questions rewritten for the
  full-10-K corpus (v1's revenue/headcount/CEO answers ARE in full 10-Ks).
  New set is absent-by-construction (market data, forecasts, analyst
  opinion, out-of-corpus company). Verified 5/5 correct abstention.

## Smoke test results (2026-07-08)

- `/query` cold: correct NVDA supply-chain answer, 7.1s, $0.0036,
  6 contexts all NVDA Item 1A; identical repeat returned `cached: true`.
- `/chat` follow-up with pronoun resolved to comparison(MSFT, NVDA) with
  both companies in retrieved contexts.
- Trace log: spans plan 1.3s / retrieve 1.0s / generate 10.2s, token and
  cost accounting per request.

## Phase 2 completion (2026-07-08, second change set)

### Async core + parallel retrieval

The whole request path is async (`AsyncOpenAI`, `AsyncQdrantClient`,
`cohere.AsyncClient`). Multi-query dense searches run concurrently, and
comparison/reasoning per-ticker fan-outs run in parallel with
`asyncio.gather` — comparison retrieval for two tickers now completes in
~1.0s total (previously sequential per ticker). `answer_sync()` is the
facade for CLI/eval callers; it reuses one private event loop because
`asyncio.run()` per call orphans the async clients' pooled connections
(observed: intermittent Cohere RuntimeErrors on 2nd+ call).

### SSE streaming

`POST /chat/{session_id}/stream` emits `meta` (intent/engine as soon as
retrieval finishes) → `delta`* (tokens) → `done` (latency, cost, trace id,
context provenance). First token arrives after plan+retrieve (~2.5s)
instead of the full ~7-10s. Streamed answers still hit the answer cache
and cost metering (usage comes from the final stream chunk).

### Redis backends

`SECRAG_REDIS_URL` switches the answer cache, embedding cache
(`cache.RedisCache`), and chat sessions (`sessions.RedisSessionStore`)
from in-process to Redis. Values are JSON so backends are interchangeable;
unit-tested against fakeredis. Unset = in-process (single replica dev).

### Auth + rate limiting

`SECRAG_API_KEYS` (comma-separated) enables `X-API-Key` auth: 401 missing,
403 wrong, per-key sliding-window rate limit (`SECRAG_RATE_LIMIT_PER_MINUTE`,
default 60) returning 429 + Retry-After. Empty keys = auth disabled with a
loud startup warning (local dev only). `/health` stays unauthenticated for
load balancers.

### Docker

Code-only image (`docker build -t sec-rag .`); corpus mounted at runtime
(`./data/store`, `./qdrant_db`), keys via env. Non-root user, healthcheck
on `/health`. Qdrant local mode takes an exclusive file lock — one
container per volume; move to a Qdrant server for replicas.

### CI (`.github/workflows/ci.yml`)

- `lint-and-test`: ruff + 34 unit tests on every push/PR
- `docker-build`: image builds on every push/PR
- `eval-smoke-gate` (manual dispatch, needs `OPENAI_API_KEY`/`COHERE_API_KEY`
  secrets): downloads filings from EDGAR, ingests, runs `eval/smoke_gate.py`
  — deterministic assertions (no LLM judge): source recall on known-source
  questions, citations present, the $28.9B figure retrievable, R002
  retrieves JPM, adversarial abstains. Exit 1 fails the run.

## Phase 2+ (2026-07-08, third change set)

### Redis-backed rate limiter

`ratelimit.make_limiter()` selects the backend: in-process sliding window
without Redis, Redis fixed-window (INCR + EXPIRE per minute-bucket) with
`SECRAG_REDIS_URL` — the limit holds across replicas instead of
multiplying by replica count. Verified live under compose: 60/min budget
returned exactly 429 beyond it.

### Deploy targets

- `docker-compose.yml`: API + Redis, corpus volumes, one command up.
- `deploy/k8s.yaml`: Deployment/Service/PVC + Redis; replicas pinned to 1
  with the Qdrant-local-lock caveat documented (move to a Qdrant server to
  scale out — Redis already makes everything else replica-safe).
- CI `publish-image` job pushes `ghcr.io/<repo>:latest` + `:<sha>` on
  every push to master (GITHUB_TOKEN, no extra secrets).

### Nightly full-RAGAS workflow

`.github/workflows/nightly-eval.yml` (07:00 UTC + manual): EDGAR download
→ ingest → full 20-question RAGAS run → `eval/check_thresholds.py`
against `eval/thresholds.json` (floors set ~0.05–0.10 under measured
values to absorb judge noise). A breach fails the run — GitHub notifies —
and posts to Slack if the `SLACK_WEBHOOK_URL` secret exists. Reports
upload as artifacts either way.

### Structured logging + OTel

- `observability/logs.py`: one JSON object per line (ts/level/logger/
  message + extras), uvicorn loggers routed through the same formatter;
  enabled in the API lifespan (`SECRAG_LOG_JSON=false` for plain text).
  Serving-path prints replaced with loggers.
- `observability/tracing.py` grew an `OtelSink`: request traces re-emit as
  real OTel spans with reconstructed timestamps (spans now record start
  offsets), exported over OTLP-HTTP when `OTEL_EXPORTER_OTLP_ENDPOINT` is
  set. Sinks compose: JSONL always, Langfuse and OTel when configured;
  one sink failing never blocks another (`MultiSink`).

## Remaining / next (Phase 3)

- Scale-out retrieval: Qdrant server + replicas > 1
- Multi-year/multi-company corpus + scheduled EDGAR sync
- Table-aware extraction and numeric QA
- Semantic/episodic user memory with governance
- Online judge sampling on production traffic
