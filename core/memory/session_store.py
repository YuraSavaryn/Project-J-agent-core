"""Redis-backed session store for conversation history.

Architecture
------------
- **Redis** is the hot cache — stores the last N messages per session with TTL.
  Writes go to Redis first for low-latency reads.
- **PostgreSQL** is the durable store — every message is persisted via
  ``append_message`` (durable write pattern).
- **Cache-miss fallback**: if Redis returns an empty list for a session that
  exists in Postgres, the history is loaded from Postgres and the Redis cache
  is warmed up for subsequent requests.

Each session is stored as a Redis List under the key ``session:{session_id}:messages``.
Every entry is a JSON-encoded ``{"role": "...", "content": "..."}`` dict.

PostgreSQL writes require a ``Session`` row to exist (FK constraint). The caller
(orchestrator/route) is responsible for creating the ``Session`` row on first use.
"""

import json
import logging
from typing import TypedDict

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.db.models import Message as ORM_Message

logger = logging.getLogger(__name__)

SESSION_KEY_PREFIX = "session"
MESSAGES_SUFFIX = "messages"


def _messages_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}:{session_id}:{MESSAGES_SUFFIX}"


class MessageDict(TypedDict):
    role: str   # "user" | "assistant" | "system"
    content: str


# Alias for backwards compatibility
Message = MessageDict


class SessionStore:
    """Async session store backed by Redis (hot cache) + PostgreSQL (durable).

    Parameters
    ----------
    redis:
        An ``redis.asyncio.Redis`` instance. Inject a ``fakeredis`` instance in tests.
    db:
        An ``AsyncSession`` for PostgreSQL. ``None`` disables durable writes and
        the cache-miss fallback (Redis-only mode, useful for tests without a DB).
    ttl:
        TTL in seconds applied to each session key on every write.
        Defaults to the value from ``config/settings.py``.
    max_history:
        Maximum number of messages kept in Redis per session.
        Older messages are trimmed from the left (oldest first).
    """

    def __init__(
        self,
        redis: Redis,  # type: ignore[type-arg]
        db: AsyncSession | None = None,
        ttl: int | None = None,
        max_history: int = 50,
    ) -> None:
        self._redis = redis
        self._db = db
        self._ttl = ttl if ttl is not None else settings.redis_session_ttl_seconds
        self._max_history = max_history

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[MessageDict]:
        """Return the conversation history for *session_id*.

        1. Try Redis (hot path).
        2. On cache-miss (empty result) AND db available → load from Postgres,
           warm the Redis cache, return the result.

        Parameters
        ----------
        session_id:
            Unique session identifier.
        limit:
            Maximum number of **most recent** messages to return.
            ``None`` returns all cached messages (up to ``max_history``).
        """
        key = _messages_key(session_id)
        start = -limit if limit else 0  # LRANGE: negative index = from the right
        raw: list[bytes] = await self._redis.lrange(key, start, -1)

        if raw:
            return [json.loads(item) for item in raw]

        # Cache-miss: try Postgres fallback
        if self._db is not None:
            logger.debug(
                "SessionStore: cache-miss for session %s — loading from Postgres", session_id
            )
            messages = await self._load_from_postgres(session_id)
            if messages:
                await self._warm_cache(session_id, messages)
            return messages[-limit:] if limit else messages

        return []

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append a message to the session history.

        Writes to **both** Redis (cache) and PostgreSQL (durable), if a DB
        session is available.

        Parameters
        ----------
        session_id:
            Unique session identifier.
        role:
            ``"user"``, ``"assistant"``, or ``"system"``.
        content:
            Message text.
        """
        # ── Durable write (Postgres) ──────────────────────────────────────────
        if self._db is not None:
            await self._persist_to_postgres(session_id, role, content)

        # ── Cache write (Redis) ───────────────────────────────────────────────
        key = _messages_key(session_id)
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)

        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self._max_history, -1)
        pipe.expire(key, self._ttl)
        await pipe.execute()

        logger.debug("SessionStore: appended %s message to session %s", role, session_id)

    async def clear(self, session_id: str) -> None:
        """Delete the entire history for *session_id* from Redis.

        Does NOT delete from Postgres — use the sessions API endpoint for that.
        """
        await self._redis.delete(_messages_key(session_id))
        logger.info("SessionStore: cleared Redis cache for session %s", session_id)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _persist_to_postgres(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Write a single message to the ``messages`` table."""
        assert self._db is not None
        msg = ORM_Message(
            session_id=session_id,
            role=role,
            content=content,
        )
        self._db.add(msg)
        # Flush only — commit is handled by the FastAPI get_db() dependency lifecycle.
        await self._db.flush()

    async def _load_from_postgres(self, session_id: str) -> list[MessageDict]:
        """Load all messages for a session from Postgres, ordered oldest-first."""
        assert self._db is not None
        result = await self._db.execute(
            select(ORM_Message)
            .where(ORM_Message.session_id == session_id)
            .order_by(ORM_Message.created_at)
        )
        rows = result.scalars().all()
        return [{"role": str(row.role), "content": row.content} for row in rows]

    async def _warm_cache(self, session_id: str, messages: list[MessageDict]) -> None:
        """Populate Redis from the Postgres result (cache warm-up)."""
        key = _messages_key(session_id)
        pipe = self._redis.pipeline()
        # Keep only the last max_history messages to avoid exceeding Redis limits
        for msg in messages[-self._max_history :]:
            payload = json.dumps(msg, ensure_ascii=False)
            pipe.rpush(key, payload)
        pipe.expire(key, self._ttl)
        await pipe.execute()
        logger.debug(
            "SessionStore: warmed Redis cache for session %s (%d messages)",
            session_id,
            len(messages),
        )
