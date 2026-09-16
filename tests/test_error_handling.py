"""Tests for step 15 — Error handling, retry logic, and structured logging.

Covers:
- Unified ErrorResponse format on validation errors (422)
- Unified ErrorResponse format on LLM unavailable (503)
- Unified ErrorResponse format on HTTP 404
- LLMRouter retry with exponential backoff (transient errors)
- LLMRouter falls back to next provider after exhausting retries
- LLMRouter does NOT retry on 4xx errors
- RequestIdMiddleware adds X-Request-Id to response headers
- configure_logging does not raise
"""

from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio

import pytest
import respx
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError

from api.dependencies import get_orchestrator, get_redis
from api.main import app
from core.llm.base import LLMResponse
from core.llm.router import LLMRouter, LLMUnavailableError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    """Orchestrator mock that returns a canned successful result."""
    from core.orchestrator import OrchestratorResult

    m = AsyncMock()
    m.handle_message = AsyncMock(
        return_value=OrchestratorResult(
            reply="Відповідь",
            session_id="test-sess",
        )
    )
    return m


@pytest.fixture
def client(mock_orchestrator: AsyncMock) -> TestClient:
    """TestClient with dependencies overridden."""

    async def _fake_redis():
        yield AsyncMock()

    app.dependency_overrides[get_redis] = _fake_redis
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ── Exception handler: unified error format ───────────────────────────────────


def test_validation_error_returns_error_envelope(client: TestClient) -> None:
    """Validation errors must use the unified {error: {code, message, field}} format."""
    response = client.post("/v1/chat", json={"message": "hi"})  # missing session_id

    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "validation_error"
    assert "message" in body["error"]
    # field should reference the missing parameter
    assert body["error"].get("field") is not None


def test_llm_unavailable_returns_error_envelope(
    client: TestClient, mock_orchestrator: AsyncMock
) -> None:
    """LLMUnavailableError must return 503 with unified error envelope."""
    mock_orchestrator.handle_message.side_effect = LLMUnavailableError("down")

    response = client.post("/v1/chat", json={"session_id": "s1", "message": "питання"})

    assert response.status_code == 503
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "llm_unavailable"
    assert "недоступний" in body["error"]["message"].lower()


def test_404_returns_error_envelope(client: TestClient) -> None:
    """Non-existent routes return 404 with unified error envelope."""
    response = client.get("/v1/nonexistent-route")

    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert "http_404" in body["error"]["code"] or body["error"]["code"] == "http_error"


def test_chat_validation_error_no_detail_key(client: TestClient) -> None:
    """Unified format: no top-level 'detail' key (that's the old FastAPI default)."""
    response = client.post("/v1/chat", json={"session_id": "s1", "message": ""})

    assert response.status_code == 422
    assert "detail" not in response.json()


# ── RequestIdMiddleware ───────────────────────────────────────────────────────


def test_request_id_added_to_response_headers(client: TestClient) -> None:
    """Every response must carry X-Request-Id header."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers


def test_request_id_echoes_client_header(client: TestClient) -> None:
    """If client sends X-Request-Id, the same value is echoed back."""
    response = client.get("/health", headers={"X-Request-Id": "my-trace-123"})
    assert response.headers.get("x-request-id") == "my-trace-123"


def test_request_id_generated_when_not_provided(client: TestClient) -> None:
    """If no X-Request-Id in request, the server generates a UUID."""
    response = client.get("/health")
    request_id = response.headers.get("x-request-id", "")
    # UUID4 format: 8-4-4-4-12 hex chars
    assert len(request_id) == 36
    assert request_id.count("-") == 4


# ── LLMRouter retry logic ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_retries_on_connection_error() -> None:
    """Router retries the primary provider on APIConnectionError before giving up."""
    good_response = LLMResponse(
        content="ок", model="test-model", provider="groq"
    )
    failing_provider = AsyncMock()
    failing_provider.name = "groq"
    failing_provider.is_available = AsyncMock(return_value=True)
    # Fail first attempt, succeed second
    failing_provider.complete = AsyncMock(
        side_effect=[APIConnectionError(request=MagicMock()), good_response]
    )

    router = LLMRouter(providers=[failing_provider], retry_attempts=2, retry_backoff_seconds=0.0)
    result = await router.complete([{"role": "user", "content": "hi"}])

    assert result.content == "ок"
    assert failing_provider.complete.call_count == 2


@pytest.mark.asyncio
async def test_router_retries_on_timeout_error() -> None:
    """Router retries the primary provider on APITimeoutError."""
    good_response = LLMResponse(
        content="після retry", model="test-model", provider="groq"
    )
    provider = AsyncMock()
    provider.name = "groq"
    provider.is_available = AsyncMock(return_value=True)
    provider.complete = AsyncMock(
        side_effect=[APITimeoutError(request=MagicMock()), good_response]
    )

    router = LLMRouter(providers=[provider], retry_attempts=2, retry_backoff_seconds=0.0)
    result = await router.complete([{"role": "user", "content": "hi"}])

    assert result.content == "після retry"
    assert provider.complete.call_count == 2


@pytest.mark.asyncio
async def test_router_falls_back_after_exhausting_retries() -> None:
    """After exhausting retries on primary, router falls back to secondary."""
    fallback_response = LLMResponse(
        content="від fallback", model="fallback-model", provider="openrouter"
    )

    primary = AsyncMock()
    primary.name = "groq"
    primary.is_available = AsyncMock(return_value=True)
    primary.complete = AsyncMock(
        side_effect=APIConnectionError(request=MagicMock())
    )

    secondary = AsyncMock()
    secondary.name = "openrouter"
    secondary.is_available = AsyncMock(return_value=True)
    secondary.complete = AsyncMock(return_value=fallback_response)

    router = LLMRouter(
        providers=[primary, secondary], retry_attempts=2, retry_backoff_seconds=0.0
    )
    result = await router.complete([{"role": "user", "content": "hi"}])

    assert result.provider == "openrouter"
    assert result.content == "від fallback"
    # Primary tried retry_attempts times
    assert primary.complete.call_count == 2
    assert secondary.complete.call_count == 1


@pytest.mark.asyncio
async def test_router_raises_unavailable_when_all_exhausted() -> None:
    """LLMUnavailableError is raised when all providers exhaust all retries."""
    provider = AsyncMock()
    provider.name = "groq"
    provider.is_available = AsyncMock(return_value=True)
    provider.complete = AsyncMock(
        side_effect=APIConnectionError(request=MagicMock())
    )

    router = LLMRouter(providers=[provider], retry_attempts=2, retry_backoff_seconds=0.0)

    with pytest.raises(LLMUnavailableError):
        await router.complete([{"role": "user", "content": "hi"}])

    assert provider.complete.call_count == 2


@pytest.mark.asyncio
async def test_router_does_not_retry_on_4xx() -> None:
    """4xx status errors are re-raised immediately (no retry, no fallback)."""
    mock_response = MagicMock()
    mock_response.status_code = 401

    provider = AsyncMock()
    provider.name = "groq"
    provider.is_available = AsyncMock(return_value=True)
    provider.complete = AsyncMock(
        side_effect=APIStatusError("Unauthorized", response=mock_response, body=None)
    )

    router = LLMRouter(providers=[provider], retry_attempts=3, retry_backoff_seconds=0.0)

    with pytest.raises(APIStatusError):
        await router.complete([{"role": "user", "content": "hi"}])

    # Must only have been called once (no retry on 4xx)
    assert provider.complete.call_count == 1


# ── Structured logging ────────────────────────────────────────────────────────


def test_configure_logging_does_not_raise() -> None:
    """configure_logging() must complete without errors."""
    from core.logging_config import configure_logging

    # Should not raise even when called multiple times
    configure_logging()
    configure_logging()


def test_bind_request_id_is_readable() -> None:
    """bind_request_id sets the value readable via get_request_id."""
    from core.logging_config import bind_request_id, get_request_id

    bind_request_id("test-req-456")
    assert get_request_id() == "test-req-456"
