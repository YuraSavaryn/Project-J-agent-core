"""FastAPI dependency providers for shared resources.

Import the dependency functions here and use them in route handlers via ``Depends()``.
"""

import secrets
from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from redis.asyncio import Redis, from_url
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.db.base import get_db
from core.llm.router import LLMRouter
from core.memory.session_store import SessionStore
from core.orchestrator import Orchestrator
from knowledge.rag.retriever import Retriever

# ── API Key auth ──────────────────────────────────────────────────────────────

async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate the X-API-Key header when API_KEY is configured.

    If ``settings.api_key`` is empty, authentication is disabled and every
    request passes through. When enabled:
    - Missing key  → 401 Unauthorized
    - Wrong key    → 403 Forbidden  (prevents timing-based enumeration via
                                     ``secrets.compare_digest``)
    """
    if not settings.api_key:
        return  # auth disabled
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_api_key", "message": "X-API-Key header is required."},
        )
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "invalid_api_key", "message": "Invalid API key."},
        )


ApiKeyDep = Annotated[None, Depends(require_api_key)]

# ── Database session ──────────────────────────────────────────────────────────
DbDep = Annotated[AsyncSession, Depends(get_db)]

# ── Redis ─────────────────────────────────────────────────────────────────────
_redis_client: Redis | None = None  # type: ignore[type-arg]


async def get_redis() -> AsyncGenerator[Redis, None]:  # type: ignore[type-arg]
    """Yield the shared async Redis client (connection pool singleton)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = await from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,
        )
    yield _redis_client


RedisDep = Annotated[Redis, Depends(get_redis)]  # type: ignore[type-arg]


# ── SessionStore ──────────────────────────────────────────────────────────────

async def get_session_store(
    redis: RedisDep,
    db: AsyncSession = Depends(get_db),
) -> SessionStore:
    """Return a ``SessionStore`` bound to the request-scoped Redis client and DB."""
    return SessionStore(redis, db=db)


SessionStoreDep = Annotated[SessionStore, Depends(get_session_store)]


# ── LLMRouter ─────────────────────────────────────────────────────────────────

def get_llm_router() -> LLMRouter:
    """Return a configured LLMRouter instance (stateless, safe to create per-request)."""
    return LLMRouter()


LLMRouterDep = Annotated[LLMRouter, Depends(get_llm_router)]


# ── Retriever ─────────────────────────────────────────────────────────────────

def get_retriever() -> "Retriever":
    from config.settings import settings
    from knowledge.rag.embeddings import get_embedding_provider
    from knowledge.rag.vectorstore.chroma_store import ChromaStore
    from knowledge.rag.retriever import Retriever

    embedding_provider = get_embedding_provider(
        provider=settings.embedding_provider,
        worker_url=settings.embedding_worker_url,
        gemini_key=settings.gemini_api_key,
        gemini_model=settings.gemini_embedding_model,
    )
    vector_store = ChromaStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
        distance_fn=settings.chroma_distance_fn,
    )
    return Retriever(embedding_provider=embedding_provider, vector_store=vector_store)

RetrieverDep = Annotated["Retriever", Depends(get_retriever)]


# ── QueryLogger ───────────────────────────────────────────────────────────────

def get_query_logger(db: AsyncSession = Depends(get_db)) -> "QueryLogger":
    from analytics.query_logger import QueryLogger
    return QueryLogger(db)

QueryLoggerDep = Annotated["QueryLogger", Depends(get_query_logger)]


# ── Orchestrator ──────────────────────────────────────────────────────────────

def get_orchestrator(
    session_store: SessionStoreDep,
    llm_router: LLMRouterDep,
    retriever: RetrieverDep,
    query_logger: QueryLoggerDep,
) -> Orchestrator:
    """Return an Orchestrator wired with the request-scoped dependencies."""
    from tools.registry import ToolRegistry
    from tools.search_knowledge_tool import SearchKnowledgeTool

    # Set up tool registry and register standard tools
    registry = ToolRegistry()
    registry.register(SearchKnowledgeTool(retriever=retriever))

    return Orchestrator(
        session_store=session_store, 
        llm_router=llm_router, 
        retriever=retriever,
        tool_registry=registry,
        query_logger=query_logger,
    )


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]


# ── IngestionPipeline ─────────────────────────────────────────────────────────

def get_ingestion_pipeline() -> "IngestionPipeline":
    """Return an ``IngestionPipeline`` wired from settings.

    Mirrors the logic in ``knowledge.rag.ingestion.run._build_pipeline``.
    The pipeline is stateless (no per-request state) so it is safe to build
    once per request.
    """
    from knowledge.rag.embeddings import get_embedding_provider
    from knowledge.rag.ingestion.pipeline import IngestionPipeline
    from knowledge.rag.vectorstore.chroma_store import ChromaStore

    embedding_provider = get_embedding_provider(
        provider=settings.embedding_provider,
        worker_url=settings.embedding_worker_url,
        gemini_key=settings.gemini_api_key,
        gemini_model=settings.gemini_embedding_model,
    )
    vector_store = ChromaStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
        distance_fn=settings.chroma_distance_fn,
    )
    vector_store.ensure_collection()
    return IngestionPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


IngestionPipelineDep = Annotated["IngestionPipeline", Depends(get_ingestion_pipeline)]
