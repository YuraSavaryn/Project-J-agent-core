"""Tests for step 14 — Sessions API endpoints.

Covers:
- GET /v1/sessions/{id} returns 200 with correct metadata
- GET /v1/sessions/{id} returns 404 for unknown session
- DELETE /v1/sessions/{id} removes session from Postgres
- DELETE /v1/sessions/{id} explicitly clears Redis cache
- DELETE /v1/sessions/{id} returns 404 for unknown session
- SessionStore.ensure_session creates the sessions row
- SessionStore.touch_session updates last_active_at
"""

import pytest
import pytest_asyncio
import fakeredis.aioredis as fakeredis
from httpx import AsyncClient, ASGITransport
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.main import app
from core.db.base import Base, get_db
from core.db.models import Session as ORM_Session, Message as ORM_Message
from core.memory.session_store import SessionStore, _messages_key

# ── In-memory SQLite engine for tests ────────────────────────────────────────

test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")


# Enable foreign key enforcement in SQLite (disabled by default)
@event.listens_for(test_engine.sync_engine, "connect")
def _enable_sqlite_fk(dbapi_connection, connection_record):  # type: ignore[type-arg]
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


TestingSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def _create_tables():
    """Create all tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session():
    """Yield a test-scoped SQLAlchemy session."""
    async with TestingSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def fake_redis():
    """Yield a fakeredis instance."""
    return fakeredis.FakeRedis(decode_responses=False)


@pytest_asyncio.fixture
async def async_client(db_session, fake_redis):
    """ASGI test client with overridden DB and Redis dependencies."""
    from api.dependencies import get_redis

    async def override_get_db():
        yield db_session

    async def override_get_redis():
        yield fake_redis

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_redis, None)


# ── Helper to seed a session directly ────────────────────────────────────────

async def _seed_session(db, session_id: str = "sess-001") -> ORM_Session:
    """Insert a Session row and return it."""
    sess = ORM_Session(
        session_id=session_id,
        channel="web",
        external_user_id="user-42",
    )
    db.add(sess)
    await db.commit()
    await db.refresh(sess)
    return sess


# ═══════════════════════════════════════════════════════════════════════════════
# GET /v1/sessions/{id}
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_session_returns_metadata(async_client, db_session):
    """GET /v1/sessions/{id} returns 200 with correct fields."""
    await _seed_session(db_session, "sess-get-1")

    resp = await async_client.get("/v1/sessions/sess-get-1")
    assert resp.status_code == 200

    data = resp.json()
    assert data["session_id"] == "sess-get-1"
    assert data["channel"] == "web"
    assert data["external_user_id"] == "user-42"
    assert "started_at" in data
    assert "last_active_at" in data


@pytest.mark.asyncio
async def test_get_session_not_found(async_client):
    """GET /v1/sessions/{id} returns 404 for non-existent session."""
    resp = await async_client.get("/v1/sessions/nonexistent-session")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /v1/sessions/{id}
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_delete_session_removes_from_postgres(async_client, db_session):
    """DELETE /v1/sessions/{id} removes the row from Postgres."""
    await _seed_session(db_session, "sess-del-1")

    resp = await async_client.delete("/v1/sessions/sess-del-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-del-1"
    assert data["deleted"] is True

    # Verify it's gone from Postgres
    result = await db_session.execute(
        select(ORM_Session).where(ORM_Session.session_id == "sess-del-1")
    )
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_session_clears_redis_cache(async_client, db_session, fake_redis):
    """DELETE /v1/sessions/{id} explicitly deletes the Redis key."""
    session_id = "sess-del-redis"
    await _seed_session(db_session, session_id)

    # Populate Redis cache manually
    cache_key = _messages_key(session_id)
    await fake_redis.rpush(cache_key, b'{"role":"user","content":"hello"}')
    assert await fake_redis.llen(cache_key) == 1

    # Delete via API
    resp = await async_client.delete(f"/v1/sessions/{session_id}")
    assert resp.status_code == 200

    # Redis key must be gone
    assert await fake_redis.llen(cache_key) == 0
    assert await fake_redis.exists(cache_key) == 0


@pytest.mark.asyncio
async def test_delete_session_not_found(async_client):
    """DELETE /v1/sessions/{id} returns 404 for non-existent session."""
    resp = await async_client.delete("/v1/sessions/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_session_cascades_messages(async_client, db_session):
    """Deleting a session also removes associated messages (FK CASCADE)."""
    await _seed_session(db_session, "sess-cascade")

    # Add a message linked to this session
    msg = ORM_Message(session_id="sess-cascade", role="user", content="test")
    db_session.add(msg)
    await db_session.commit()

    resp = await async_client.delete("/v1/sessions/sess-cascade")
    assert resp.status_code == 200

    # Expire the session identity map so we don't read stale cached objects
    db_session.expire_all()

    # Message must be gone (CASCADE)
    result = await db_session.execute(
        select(ORM_Message).where(ORM_Message.session_id == "sess-cascade")
    )
    assert result.scalars().all() == []


# ═══════════════════════════════════════════════════════════════════════════════
# SessionStore.ensure_session
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_ensure_session_creates_row(db_session, fake_redis):
    """ensure_session inserts a sessions row when it doesn't exist."""
    store = SessionStore(redis=fake_redis, db=db_session)

    await store.ensure_session("new-sess", channel="telegram", external_user_id="tg-999")
    await db_session.commit()

    result = await db_session.execute(
        select(ORM_Session).where(ORM_Session.session_id == "new-sess")
    )
    sess = result.scalar_one_or_none()
    assert sess is not None
    # ChannelType enum — compare via .value for portability across DB backends
    channel_val = sess.channel.value if hasattr(sess.channel, "value") else str(sess.channel)
    assert channel_val == "telegram"
    assert sess.external_user_id == "tg-999"


@pytest.mark.asyncio
async def test_ensure_session_is_idempotent(db_session, fake_redis):
    """ensure_session called twice does not raise and does not duplicate."""
    store = SessionStore(redis=fake_redis, db=db_session)

    await store.ensure_session("idem-sess", channel="web", external_user_id="u1")
    await db_session.commit()

    # Second call should be a no-op, not an error
    await store.ensure_session("idem-sess", channel="web", external_user_id="u1")
    await db_session.commit()

    result = await db_session.execute(
        select(ORM_Session).where(ORM_Session.session_id == "idem-sess")
    )
    rows = result.scalars().all()
    assert len(rows) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SessionStore.touch_session
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_touch_session_updates_last_active_at(db_session, fake_redis):
    """touch_session updates last_active_at to a more recent timestamp."""
    from datetime import datetime, timezone, timedelta

    store = SessionStore(redis=fake_redis, db=db_session)

    # Create the session row manually with an old last_active_at
    old_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    sess = ORM_Session(
        session_id="touch-sess",
        channel="web",
        external_user_id="u1",
        last_active_at=old_time,
    )
    db_session.add(sess)
    await db_session.commit()

    # Touch it
    await store.touch_session("touch-sess")
    await db_session.commit()
    await db_session.refresh(sess)

    # last_active_at should now be much newer than old_time.
    # SQLite returns naive datetimes — make comparison timezone-agnostic.
    refreshed_at = sess.last_active_at
    if refreshed_at.tzinfo is not None:
        refreshed_at = refreshed_at.replace(tzinfo=None)
    old_naive = old_time.replace(tzinfo=None)
    assert refreshed_at > old_naive + timedelta(days=365 * 10)


# ═══════════════════════════════════════════════════════════════════════════════
# append_message auto-creates session
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_append_message_auto_creates_session(db_session, fake_redis):
    """append_message creates the sessions row automatically on first use."""
    store = SessionStore(redis=fake_redis, db=db_session)

    # No sessions row exists yet
    result = await db_session.execute(
        select(ORM_Session).where(ORM_Session.session_id == "auto-sess")
    )
    assert result.scalar_one_or_none() is None

    # This should auto-create the session row
    await store.append_message(
        "auto-sess", role="user", content="hello",
        channel="web", external_user_id="auto-user",
    )
    await db_session.commit()

    result = await db_session.execute(
        select(ORM_Session).where(ORM_Session.session_id == "auto-sess")
    )
    sess = result.scalar_one_or_none()
    assert sess is not None
    assert sess.external_user_id == "auto-user"
