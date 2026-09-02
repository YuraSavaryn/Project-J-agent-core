"""Tests for the health and chat endpoints (steps 1–3).

Chat tests override the Redis dependency with fakeredis so no real Redis is needed.
"""

from __future__ import annotations

import pytest
import fakeredis.aioredis as fakeredis
from fastapi.testclient import TestClient

from api.dependencies import get_redis
from api.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> TestClient:
    """TestClient with Redis replaced by fakeredis."""
    fake = fakeredis.FakeRedis(decode_responses=False)

    async def _fake_redis():
        yield fake

    app.dependency_overrides[get_redis] = _fake_redis
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Chat (echo stub + session history) ───────────────────────────────────────

def test_chat_echo_returns_reply(client: TestClient) -> None:
    payload = {
        "session_id": "test-session-1",
        "message": "Привіт!",
        "channel": "web",
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-1"
    assert "[ECHO]" in data["reply"]
    assert "Привіт!" in data["reply"]
    assert isinstance(data["sources"], list)
    assert isinstance(data["suggested_actions"], list)


def test_chat_default_channel(client: TestClient) -> None:
    """channel field should be optional, defaulting to 'web'."""
    payload = {"session_id": "s2", "message": "Тест"}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200


def test_chat_empty_message_fails_validation(client: TestClient) -> None:
    payload = {"session_id": "s3", "message": ""}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422


def test_chat_missing_session_id_fails_validation(client: TestClient) -> None:
    payload = {"message": "Привіт!"}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422
