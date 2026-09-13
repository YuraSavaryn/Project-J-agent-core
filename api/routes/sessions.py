"""Sessions management endpoints.

Routes
------
GET  /v1/sessions/{session_id}  — Retrieve session metadata.
DELETE /v1/sessions/{session_id} — Delete session from Postgres (cascades via FK)
                                   and explicitly clears the Redis cache.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RedisDep, get_redis
from api.schemas.sessions import DeleteSessionResponse, SessionResponse
from core.db.base import get_db
from core.db.models import Session as ORM_Session
from core.memory.session_store import _messages_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get session metadata",
    description=(
        "Returns metadata for the given session: channel, external user ID, "
        "creation time, and last-active timestamp."
    ),
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Return metadata for *session_id*, or 404 if it does not exist."""
    result = await db.execute(
        select(ORM_Session).where(ORM_Session.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return SessionResponse(
        session_id=session.session_id,
        channel=session.channel.value if hasattr(session.channel, "value") else str(session.channel),
        external_user_id=session.external_user_id,
        started_at=session.started_at,
        last_active_at=session.last_active_at,
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Delete a session",
    description=(
        "Deletes the session and all associated messages, query logs, and feedback "
        "from PostgreSQL via cascading FK constraints.\n\n"
        "Also explicitly deletes the Redis cache key for this session — "
        "the Postgres CASCADE does **not** automatically clear Redis."
    ),
)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),  # type: ignore[type-arg]
) -> DeleteSessionResponse:
    """Delete *session_id* from Postgres and clear its Redis cache.

    Returns ``deleted: false`` (404) if the session does not exist.
    """
    # Check existence first to return 404 on unknown sessions
    result = await db.execute(
        select(ORM_Session.session_id).where(ORM_Session.session_id == session_id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    # Delete from Postgres — cascades to messages, query_logs, query_log_chunks, feedback
    await db.execute(
        delete(ORM_Session).where(ORM_Session.session_id == session_id)
    )
    logger.info("Sessions: deleted session %s from Postgres", session_id)

    # Explicitly clear Redis — FK CASCADE does NOT propagate to Redis
    cache_key = _messages_key(session_id)
    deleted_count = await redis.delete(cache_key)
    logger.info(
        "Sessions: deleted Redis key %s (existed=%s)",
        cache_key,
        bool(deleted_count),
    )

    return DeleteSessionResponse(session_id=session_id, deleted=True)
