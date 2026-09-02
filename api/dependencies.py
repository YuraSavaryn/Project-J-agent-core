"""FastAPI dependency providers for shared resources.

Import the dependency functions here and use them in route handlers via ``Depends()``.
"""

from __future__ import annotations

from typing import AsyncGenerator, Annotated

from fastapi import Depends
from redis.asyncio import Redis, from_url

from config.settings import settings
from core.memory.session_store import SessionStore

# ── Redis ─────────────────────────────────────────────────────────────────────
_redis_client: Redis | None = None  # type: ignore[type-arg]


async def get_redis() -> AsyncGenerator[Redis, None]:  # type: ignore[type-arg]
    """Yield the shared async Redis client (connection pool singleton)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = await from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,  # keep bytes — SessionStore handles decoding
        )
    yield _redis_client


RedisDep = Annotated[Redis, Depends(get_redis)]  # type: ignore[type-arg]


# ── SessionStore ──────────────────────────────────────────────────────────────

async def get_session_store(redis: RedisDep) -> SessionStore:
    """Return a ``SessionStore`` bound to the request-scoped Redis client."""
    return SessionStore(redis)


SessionStoreDep = Annotated[SessionStore, Depends(get_session_store)]

