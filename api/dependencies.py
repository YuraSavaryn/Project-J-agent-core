"""FastAPI dependency providers for shared resources.

Import the dependency functions here and use them in route handlers via ``Depends()``.
"""

from __future__ import annotations

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from redis.asyncio import Redis, from_url

from config.settings import settings
from core.llm.router import LLMRouter
from core.memory.session_store import SessionStore
from core.orchestrator import Orchestrator

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


# ── Orchestrator ──────────────────────────────────────────────────────────────

def get_orchestrator(
    session_store: SessionStoreDep,
    llm_router: LLMRouterDep,
) -> Orchestrator:
    """Return an Orchestrator wired with the request-scoped session store and LLM router."""
    return Orchestrator(session_store=session_store, llm_router=llm_router)


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
