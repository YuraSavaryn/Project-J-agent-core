"""Tests for SessionStore cache-aside logic (step 4).

Tests dual-write (Redis + Postgres) and cache-miss fallback behaviour.
Instead of running a real DB, we mock the Postgres methods to verify
the correct calls are made — keeping the tests fast and dependency-free.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import fakeredis.aioredis as fakeredis

from core.memory.session_store import MessageDict, SessionStore


@pytest.fixture
def redis():
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def store_redis_only(redis):
    """SessionStore with no DB — Redis-only mode."""
    return SessionStore(redis, db=None, ttl=60, max_history=10)


@pytest.fixture
def mock_db():
    """A mock AsyncSession that satisfies the type check."""
    db = AsyncMock()
    db.add = AsyncMock()
    db.flush = AsyncMock()
    return db


@pytest.fixture
def store_with_db(redis, mock_db):
    """SessionStore with mocked DB backend."""
    return SessionStore(redis, db=mock_db, ttl=60, max_history=10)


# ── Dual-write ────────────────────────────────────────────────────────────────

async def test_append_calls_persist_to_postgres(
    store_with_db: SessionStore,
    redis,
) -> None:
    """append_message with db= set should call _persist_to_postgres."""
    with patch.object(
        store_with_db,
        "_persist_to_postgres",
        new_callable=AsyncMock,
    ) as mock_persist:
        await store_with_db.append_message("s1", "user", "Hello!")
        mock_persist.assert_awaited_once_with("s1", "user", "Hello!")


async def test_append_also_writes_to_redis(store_with_db: SessionStore) -> None:
    """append_message should also write to Redis regardless of DB availability."""
    with patch.object(store_with_db, "_persist_to_postgres", new_callable=AsyncMock):
        await store_with_db.append_message("s2", "assistant", "Reply")

    history = await store_with_db.get_history("s2")
    assert len(history) == 1
    assert history[0] == {"role": "assistant", "content": "Reply"}


async def test_append_no_db_skips_postgres(store_redis_only: SessionStore) -> None:
    """When db=None, append_message should not call any Postgres methods."""
    with patch.object(
        store_redis_only,
        "_persist_to_postgres",
        new_callable=AsyncMock,
    ) as mock_persist:
        await store_redis_only.append_message("s3", "user", "No DB")
        mock_persist.assert_not_awaited()


# ── Cache-miss fallback ───────────────────────────────────────────────────────

async def test_cache_miss_calls_load_from_postgres(
    store_with_db: SessionStore,
) -> None:
    """On Redis cache-miss, get_history should call _load_from_postgres."""
    fake_pg_history: list[MessageDict] = [{"role": "user", "content": "From DB"}]

    with (
        patch.object(
            store_with_db,
            "_load_from_postgres",
            new_callable=AsyncMock,
            return_value=fake_pg_history,
        ) as mock_load,
        patch.object(
            store_with_db,
            "_warm_cache",
            new_callable=AsyncMock,
        ),
    ):
        result = await store_with_db.get_history("miss-session")
        mock_load.assert_awaited_once_with("miss-session")
        assert result == fake_pg_history


async def test_cache_miss_warms_redis(store_with_db: SessionStore) -> None:
    """After cache-miss, _warm_cache should be called with Postgres data."""
    fake_pg_history: list[MessageDict] = [{"role": "assistant", "content": "Cached!"}]

    with (
        patch.object(
            store_with_db,
            "_load_from_postgres",
            new_callable=AsyncMock,
            return_value=fake_pg_history,
        ),
        patch.object(
            store_with_db,
            "_warm_cache",
            new_callable=AsyncMock,
        ) as mock_warm,
    ):
        await store_with_db.get_history("warm-session")
        mock_warm.assert_awaited_once_with("warm-session", fake_pg_history)


async def test_cache_hit_skips_postgres(store_with_db: SessionStore) -> None:
    """When Redis has data, _load_from_postgres should NOT be called."""
    # Pre-populate Redis
    with patch.object(store_with_db, "_persist_to_postgres", new_callable=AsyncMock):
        await store_with_db.append_message("hit-session", "user", "Already cached")

    with patch.object(
        store_with_db,
        "_load_from_postgres",
        new_callable=AsyncMock,
    ) as mock_load:
        result = await store_with_db.get_history("hit-session")
        mock_load.assert_not_awaited()
        assert result[0]["content"] == "Already cached"


async def test_cache_miss_no_db_returns_empty(store_redis_only: SessionStore) -> None:
    """When db=None and Redis is empty, return empty list (no DB fallback)."""
    result = await store_redis_only.get_history("unknown-session")
    assert result == []

