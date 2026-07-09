# Phase 1 — Redesigned RAG pipeline (`sec_rag`)

Implemented 2026-07-08. This phase replaces the script-based LlamaIndex
pipeline with a custom, framework-light package targeting the highest-ROI
quality fixes identified in the architectural review.

## What was built

```
src/sec_rag/
├── config.py                # pydantic-settings: every tunable in one place
├── models.py                # typed domain models (chunks, plans, answers)
├── pipeline.py              # RAGPipeline.answer(): plan → retrieve → generate
├── ingestion/
│   ├── parser.py            # 10-K HTML → Item sections (TOC + running-header safe)
│   ├── chunker.py           # parent (~1500 tok) / child (~300 tok) chunks,
│   │                        #   context headers fix "we/our" attribution
│   ├── store.py             # parents+children as JSONL per ticker
│   └── pipeline.py          # idempotent ingest → embeddings → Qdrant (local mode)
├── retrieval/
│   ├── planner.py           # LLM query planner: intent, tickers, rewrite,
│   │                        #   paraphrases (replaces question-ID routing)
│   ├── bm25.py              # in-process BM25 over children
│   ├── fusion.py            # Reciprocal Rank Fusion
│   ├── rerank.py            # Cohere cross-encoder (retry + graceful fallback)
│   └── retriever.py         # hybrid dense+BM25 → RRF → class-aware rerank
│                            #   → child→parent expansion
└── generation/generator.py  # citation-required, intent-specific prompts

eval/
├── harness.py               # pipeline-agnostic: Recall@K/Precision@K + RAGAS
├── run_secrag.py            # run dataset through the new pipeline
├── compare.py               # diff two metric reports
└── metrics_*.json           # versioned eval reports
```

## Retrieval policy (encodes the project's empirical lessons)

| Intent | Retrieval | Rationale |
| --- | --- | --- |
| factual | hybrid → Cohere rerank, score floor 0.30 | precision |
| reasoning | hybrid, **no reranker**, per-ticker fan-out when multi-company | rerankers penalize indirect evidence (legacy R003 bug) |
| comparison | per-ticker hybrid+rerank loops | metadata filtering is the only source-diversity guarantee (legacy lesson #4) |

## How to run

```powershell
# Ingest (idempotent per ticker; needs OPENAI_API_KEY)
python -m sec_rag.ingestion.pipeline            # all filings in data/filings
python -m sec_rag.ingestion.pipeline NVDA       # one ticker

# Query interactively
python -c "from sec_rag.pipeline import RAGPipeline; print(RAGPipeline().answer('What are NVIDIA supply chain risks?').text)"

# Evaluate + compare
python -m eval.run_secrag --label secrag-v1
python -m eval.compare baseline-legacy-fullcorpus secrag-v1

# Unit tests
python -m pytest tests/unit -q
```

## Key findings during Phase 1

1. **The legacy eval was broken twice over.** The per-source filters targeted
   `file_name = "msft_10k.txt"` — vectors deleted when the index moved to
   full 10-Ks. And the historic RAGAS numbers (faithfulness 0.589) were
   dominated by measurement artifacts (1000-char context truncation,
   missing source attribution), not real quality: the re-measured baseline
   is faithfulness 0.891.
2. **Adversarial gold labels are stale.** Revenue, headcount, and CEO *are*
   in the full 10-K corpus now, so "correct abstention" on A001/A004 is the
   wrong expected behavior. The dataset needs revision (Phase 1 follow-up).
3. **MSFT 10-K HTML repeats Item headings as running page headers** — any
   naive Item splitter silently loses ~90% of section text. The parser
   merges consecutive same-item segments before filtering.
4. **Absolute reranker score floors are unreliable.** In secrag-v1, F005's
   correct chunk ranked #1 in dense retrieval and #1 after RRF, but Cohere
   scored it 0.122 — below the 0.30 floor — so the pipeline abstained on a
   question it had retrieved perfectly. Cross-encoder score distributions
   vary per query phrasing. Fix (v2): the floor only trims results beyond
   `rerank_min_keep` (default 4); it can never starve the context.

## Results (secrag-v2 vs re-measured legacy baseline, 2026-07-08)

RAGAS (gpt-4o-mini judge), factual + reasoning questions:

| Metric | legacy baseline | secrag-v2 | delta |
| --- | --- | --- | --- |
| faithfulness | 0.891 | **0.915** | +0.024 |
| answer_relevancy | 0.574 | **0.791** | +0.217 |
| context_recall | 0.863 | **0.900** | +0.037 |
| context_precision | 0.728 | **0.845** | +0.117 |

Factual category is now perfect on retrieval (RAGAS recall/precision 1.0/1.0,
source-level precision@K 0.88 → 1.0) with relevancy 0.333 → 0.735. All five
factual questions score faithfulness 1.0. The historic hard cases (F004
$28.9B truncation, R002 JPMorgan lexical gap, R005 cross-doc attribution)
all resolve through generic mechanisms — no per-question routing remains.

Remaining soft spots: F003 relevancy=0.0 (answer style judged noncommittal —
investigate the JPM resolution-planning answer), reasoning recall 0.800 vs
0.867 baseline (R002/R003/R004 at 0.667: reference answers cite evidence
across more sections than the 3-parents-per-ticker fan-out returns), and
reasoning context precision 0.690 (inherent: comparative reasoning contexts
include companies the gold labels don't credit).

Full reports: `eval/metrics_baseline-legacy-fullcorpus.json`,
`eval/metrics_secrag-v2.json`. Compare any two labels with
`python -m eval.compare <a> <b>`.

## Known limitations / next steps

- BM25 is in-process (rebuilt from JSONL at startup, ~1s). Fine at 2k
  children; move to Qdrant sparse vectors or OpenSearch when the corpus grows.
- Qdrant runs in embedded local mode (`qdrant_db/`); swap
  `QdrantClient(path=...)` for a server URL in production (config change).
- No caching, retries are reranker-only, no tracing yet — Phase 2.
- Evaluation dataset: grow past 20 questions, fix adversarial labels,
  add numeric/table questions, hold out a tuning-free slice.
