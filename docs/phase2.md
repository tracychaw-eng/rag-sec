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

## Remaining Phase 2 work

- Docker image + CI (lint, unit tests, eval smoke gate)
- Async clients + streaming responses (SSE) on /chat
- Redis backends for caches + session store
- API auth + rate limiting
- Latency: generation dominates (~10s on comparison questions) — consider
  streaming and/or smaller context budgets
