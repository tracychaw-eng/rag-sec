"""Central configuration — single source of truth for every tunable.

All values can be overridden via environment variables with the SECRAG_
prefix (e.g. SECRAG_RERANK_TOP_N=8) or a local .env file. API keys use
their conventional unprefixed names.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SECRAG_",
        env_file=".env",
        extra="ignore",
    )

    # --- API keys (conventional env names, no prefix) ---
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    cohere_api_key: str = Field(validation_alias="COHERE_API_KEY")

    # --- Models ---
    llm_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    rerank_model: str = "rerank-english-v3.0"

    # --- Paths / storage ---
    filings_dir: Path = Path("data/filings")
    store_dir: Path = Path("data/store")          # parent/child chunk JSONL
    qdrant_path: Path = Path("qdrant_db")          # embedded Qdrant (local mode)
    # Set SECRAG_QDRANT_URL (e.g. http://qdrant:6333) to use a Qdrant server
    # instead of embedded local mode. Required for replicas > 1 — local mode
    # takes an exclusive file lock.
    qdrant_url: str | None = None
    # Collection name carries the embedding model + schema version so an
    # embedding-model change can never silently mix vector spaces.
    collection: str = "sec10k_children_3small_v3"

    # --- Chunking ---
    child_tokens: int = 300
    child_overlap_tokens: int = 60
    parent_tokens: int = 1500
    min_section_chars: int = 1500   # segments shorter than this = TOC noise

    # --- Retrieval ---
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    rerank_top_n: int = 8           # children kept after rerank (precise mode)
    rerank_score_floor: float = 0.30
    rerank_min_keep: int = 4        # floor never cuts below this many
    reasoning_top_children: int = 12  # reasoning mode: no reranker (see README
                                      # lesson: rerankers penalize indirect
                                      # evidence needed for inference)
    reasoning_parents_per_ticker: int = 3  # multi-company reasoning fan-out
    per_ticker_top_n: int = 5       # comparison mode: children per ticker
    max_parents: int = 6            # context budget: parents passed to the LLM

    # --- Generation ---
    temperature: float = 0.0
    max_answer_tokens: int = 1024

    # --- Resilience (OpenAI SDK built-in retry: exponential backoff on
    #     429/5xx/connection errors; Cohere retried via tenacity) ---
    openai_timeout_s: float = 60.0
    openai_max_retries: int = 4

    # --- Caching / sessions (in-process by default; set SECRAG_REDIS_URL
    #     to back caches and chat sessions with Redis — required for
    #     multi-replica deployments) ---
    cache_max_items: int = 2048
    cache_ttl_s: float = 3600.0
    enable_answer_cache: bool = True
    redis_url: str | None = None

    # --- API security ---
    # Comma-separated API keys. Empty = auth disabled (local dev only);
    # the server logs a warning at startup in that case.
    api_keys: str = ""
    rate_limit_per_minute: int = 60

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    # --- Online judge (production hallucination sampling) ---
    judge_sample_rate: float = 0.0     # fraction of answers scored; 0 = off
    judge_log_path: Path = Path("logs/judge.jsonl")

    # --- User memory ---
    memory_confidence_floor: float = 0.7

    # --- Observability ---
    langfuse_public_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY")
    otel_endpoint: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    traces_path: Path = Path("logs/traces.jsonl")
    log_json: bool = True   # structured JSON logs in the API server

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def get_settings() -> Settings:
    return Settings()
