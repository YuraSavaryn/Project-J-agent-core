"""FastAPI dependency providers for shared resources.

Import the dependency functions here and use them in route handlers via ``Depends()``.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from redis.asyncio import Redis, from_url

from config.settings import settings
from core.llm.router import LLMRouter
from core.memory.session_store import SessionStore
from core.orchestrator import Orchestrator
from knowledge.rag.retriever import Retriever

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

async def get_session_store(redis: RedisDep) -> SessionStore:
    """Return a ``SessionStore`` bound to the request-scoped Redis client."""
    return SessionStore(redis)


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


# ── Orchestrator ──────────────────────────────────────────────────────────────

def get_orchestrator(
    session_store: SessionStoreDep,
    llm_router: LLMRouterDep,
    retriever: RetrieverDep,
) -> Orchestrator:
    """Return an Orchestrator wired with the request-scoped dependencies."""
    return Orchestrator(
        session_store=session_store, 
        llm_router=llm_router, 
        retriever=retriever
    )


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
