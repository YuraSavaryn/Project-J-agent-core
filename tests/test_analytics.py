"""Tests for analytics logger and feedback endpoints."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select

from api.main import app
from core.db.base import Base, get_db
from core.db.models import QueryLog, QueryLogChunk, Feedback, Chunk, Document, Session
from core.db.enums import DocumentSourceType, AnswerSource
from analytics.query_logger import QueryLogger

# In-memory SQLite engine for testing
test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
TestingSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)

@pytest_asyncio.fixture
async def db_session():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with TestingSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def async_client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def sample_db_data(db_session):
    """Seed DB with a session, document, and chunk so QueryLogger can link to it."""
    # SQLite requires FKs to exist
    sess = Session(session_id="test-session-123", channel="web", external_user_id="user1")
    sess2 = Session(session_id="test-session-999", channel="web", external_user_id="user2")
    db_session.add_all([sess, sess2])

    doc = Document(
        source_type=DocumentSourceType.manual,
        title="Test Doc",
        chroma_collection="test"
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = Chunk(
        document_id=doc.id,
        chroma_chunk_id="test-uuid-1234",
        chunk_index=0,
        content_preview="some content"
    )
    db_session.add(chunk)
    await db_session.commit()
    return chunk


@pytest.mark.asyncio
async def test_query_logger_logs_query_and_chunks(db_session, sample_db_data):
    logger = QueryLogger(db_session)
    
    # Simulate a retrieval result
    chunks = [
        {"chunk_id": "test-uuid-1234", "score": 0.85, "rank": 1},
        {"chunk_id": "unknown-uuid", "score": 0.5, "rank": 2}  # should be ignored
    ]
    
    log_entry = await logger.log_query(
        session_id="test-session-123",
        query_text="What is this?",
        source=AnswerSource.rag,
        latency_ms=1200,
        llm_provider="groq",
        llm_model="llama-3",
        retrieved_chunks=chunks
    )
    
    await db_session.commit()
    
    # Verify QueryLog
    assert log_entry.id is not None
    assert log_entry.query_text == "What is this?"
    assert log_entry.latency_ms == 1200
    
    # Verify QueryLogChunk
    stmt = select(QueryLogChunk).where(QueryLogChunk.query_log_id == log_entry.id)
    result = await db_session.execute(stmt)
    linked_chunks = result.scalars().all()
    
    assert len(linked_chunks) == 1
    assert linked_chunks[0].chunk_id == sample_db_data.id
    assert linked_chunks[0].rank == 1
    assert linked_chunks[0].score == 0.85


@pytest.mark.asyncio
async def test_feedback_endpoint(db_session, async_client: AsyncClient, sample_db_data):
    # Setup a query log first
    logger = QueryLogger(db_session)
    log_entry = await logger.log_query(
        session_id="test-session-999",
        query_text="Is this test good?",
        source=AnswerSource.curated,
        latency_ms=100,
    )
    await db_session.commit()

    # 1. Create feedback (thumbs up)
    resp = await async_client.post(
        "/v1/feedback",
        json={
            "query_log_id": log_entry.id,
            "rating": 1,
            "comment": "Very helpful"
        }
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "created"

    # Verify DB
    stmt = select(Feedback).where(Feedback.query_log_id == log_entry.id)
    result = await db_session.execute(stmt)
    fb = result.scalar_one()
    assert fb.rating == 1
    assert fb.comment == "Very helpful"

    # 2. Update feedback (change to thumbs down)
    resp = await async_client.post(
        "/v1/feedback",
        json={
            "query_log_id": log_entry.id,
            "rating": -1,
            "comment": "Actually, it was bad"
        }
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "updated"

    # Verify DB
    await db_session.refresh(fb)
    assert fb.rating == -1
    assert fb.comment == "Actually, it was bad"

    # 3. Invalid rating
    resp = await async_client.post(
        "/v1/feedback",
        json={"query_log_id": log_entry.id, "rating": 5}
    )
    assert resp.status_code == 422
