"""Tests for the health and chat endpoints (steps 1–6).

- /health: straightforward liveness check.
- /v1/chat: Orchestrator is replaced with an AsyncMock so tests don't need
  real Redis or LLM keys.  LLMUnavailableError path is also covered.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_orchestrator, get_redis
from api.main import app
from core.llm.router import LLMUnavailableError
from core.orchestrator import OrchestratorResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    """Orchestrator mock that returns a canned successful result."""
    m = AsyncMock()
    m.handle_message = AsyncMock(
        return_value=OrchestratorResult(
            reply="Привіт від LLM!",
            session_id="test-session-1",
        )
    )
    return m


@pytest.fixture
def client(mock_orchestrator: AsyncMock) -> TestClient:
    """TestClient with Redis and Orchestrator dependencies overridden."""
    # Suppress Redis by overriding get_redis with a no-op generator
    async def _fake_redis():
        yield AsyncMock()

    app.dependency_overrides[get_redis] = _fake_redis
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── Chat — happy path ─────────────────────────────────────────────────────────

def test_chat_returns_llm_reply(client: TestClient) -> None:
    payload = {
        "session_id": "test-session-1",
        "message": "Привіт!",
        "channel": "web",
    }
    response = client.post("/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-1"
    assert data["reply"] == "Привіт від LLM!"
    assert isinstance(data["sources"], list)
    assert isinstance(data["suggested_actions"], list)


def test_chat_passes_session_and_message_to_orchestrator(
    client: TestClient, mock_orchestrator: AsyncMock
) -> None:
    """Orchestrator must receive the exact session_id and message from the request."""
    payload = {"session_id": "my-session", "message": "Що таке Пласт?"}
    client.post("/v1/chat", json=payload)

    mock_orchestrator.handle_message.assert_awaited_once_with(
        session_id="my-session",
        user_message="Що таке Пласт?",
        channel="web",
        external_user_id="anonymous",
    )


def test_chat_default_channel(client: TestClient) -> None:
    """channel field is optional (defaults to 'web')."""
    payload = {"session_id": "s2", "message": "Тест"}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200


# ── Chat — validation errors ──────────────────────────────────────────────────

def test_chat_empty_message_fails_validation(client: TestClient) -> None:
    payload = {"session_id": "s3", "message": ""}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422


def test_chat_missing_session_id_fails_validation(client: TestClient) -> None:
    payload = {"message": "Привіт!"}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422


# ── Chat — LLM unavailable (503) ─────────────────────────────────────────────

def test_chat_returns_503_when_llm_unavailable(
    client: TestClient, mock_orchestrator: AsyncMock
) -> None:
    """When all LLM providers fail, /v1/chat must return 503 with a user message."""
    mock_orchestrator.handle_message.side_effect = LLMUnavailableError("All providers down")

    payload = {"session_id": "s4", "message": "Питання"}
    response = client.post("/v1/chat", json=payload)

    assert response.status_code == 503
    body = response.json()
    # New unified error envelope: {"error": {"code": "...", "message": "..."}}
    assert "error" in body
    assert "недоступний" in body["error"]["message"].lower()
