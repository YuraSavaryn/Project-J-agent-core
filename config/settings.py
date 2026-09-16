"""Application settings loaded from environment variables via pydantic-settings.

Priority (highest → lowest):
  1. Real environment variables (e.g. set in docker-compose `environment:`)
  2. Values from the `.env` file (loaded automatically by pydantic-settings)

All fields are required and must be provided either in the `.env` file 
or as environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for agent-core.

    All settings are required and are populated via environment variables or a `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str
    app_host: str
    app_port: int
    log_level: str

    # ── LLM Providers ────────────────────────────────────────────────────────
    groq_api_key: str
    groq_base_url: str
    groq_default_model: str

    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_default_model: str

    llm_primary_provider: str
    llm_request_timeout_seconds: float
    llm_retry_attempts: int
    llm_retry_backoff_seconds: float

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str
    redis_session_ttl_seconds: int

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_dsn: str

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_host: str
    chroma_port: int
    chroma_collection: str
    chroma_vector_dim: int
    chroma_distance_fn: str

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: str
    embedding_worker_url: str
    embedding_request_timeout_seconds: float
    gemini_api_key: str
    gemini_embedding_model: str

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_top_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int


# Singleton settings instance — import this everywhere
settings = Settings()