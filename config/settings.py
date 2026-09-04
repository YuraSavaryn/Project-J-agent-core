"""Application settings loaded from environment variables via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for agent-core.

    All settings can be overridden via environment variables or a `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── LLM Providers ────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_default_model: str = "llama-3.1-70b-versatile"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "anthropic/claude-3.5-sonnet"

    llm_primary_provider: str = "groq"  # groq | openrouter
    llm_request_timeout_seconds: float = 30.0

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_session_ttl_seconds: int = 86400  # 24 hours

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_dsn: str = "postgresql+asyncpg://agent:secret@localhost:5432/agentcore"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    # One shared collection for all knowledge (per step 9 convention)
    chroma_collection: str = "ngo_knowledge_base"
    # bge-m3 / e5 embedding dimension; cosine distance for similarity
    chroma_vector_dim: int = 1024
    chroma_distance_fn: str = "cosine"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: str = "local"  # local | gemini
    embedding_worker_url: str = "http://localhost:8002"
    gemini_api_key: str = ""
    gemini_embedding_model: str = "text-embedding-004"

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64


# Singleton settings instance — import this everywhere
settings = Settings()

