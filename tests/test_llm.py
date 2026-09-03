"""Tests for LLM providers and router (step 5).

Strategy
--------
- Provider tests mock ``AsyncOpenAI.chat.completions.create`` directly via
  ``unittest.mock.patch``, bypassing the HTTP layer entirely.  This avoids
  any httpx/httpx2 transport incompatibilities.
- Router tests use ``AsyncMock`` providers to verify fallback logic without
  touching any real HTTP.
- ``is_available()`` is tested by controlling ``api_key`` values.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from core.llm.base import LLMProvider, LLMResponse
from core.llm.groq_provider import GroqProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.router import LLMRouter, LLMUnavailableError


# ── Helpers ───────────────────────────────────────────────────────────────────

FAKE_MESSAGES = [{"role": "user", "content": "Привіт!"}]


def _fake_completion(
    content: str = "Відповідь від LLM",
    model: str = "llama-3.1-70b-versatile",
) -> MagicMock:
    """Build a minimal fake openai ChatCompletion object."""
    tool_calls_obj = MagicMock()
    tool_calls_obj.tool_calls = None
    tool_calls_obj.content = content

    choice = MagicMock()
    choice.message = tool_calls_obj

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20

    completion = MagicMock()
    completion.choices = [choice]
    completion.model = model
    completion.usage = usage
    return completion


# ── GroqProvider tests ────────────────────────────────────────────────────────

async def test_groq_complete_returns_llm_response() -> None:
    """GroqProvider.complete() should parse the completion into LLMResponse."""
    provider = GroqProvider(api_key="sk-test", default_model="llama-3.1-70b-versatile", timeout=5.0)

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_fake_completion("Привіт! Чим можу допомогти?"),
    ):
        result = await provider.complete(FAKE_MESSAGES)

    assert isinstance(result, LLMResponse)
    assert result.provider == "groq"
    assert result.content == "Привіт! Чим можу допомогти?"
    assert result.model == "llama-3.1-70b-versatile"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.tool_calls == []


async def test_groq_passes_model_override() -> None:
    """model= argument should be forwarded to the API call."""
    provider = GroqProvider(api_key="sk-test", timeout=5.0)

    mock_create = AsyncMock(return_value=_fake_completion())
    with patch.object(provider._client.chat.completions, "create", mock_create):
        await provider.complete(FAKE_MESSAGES, model="llama-3.1-8b-instant")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.1-8b-instant"


async def test_groq_passes_tools() -> None:
    """tools= list should be forwarded with tool_choice='auto'."""
    provider = GroqProvider(api_key="sk-test", timeout=5.0)
    fake_tool = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    mock_create = AsyncMock(return_value=_fake_completion())
    with patch.object(provider._client.chat.completions, "create", mock_create):
        await provider.complete(FAKE_MESSAGES, tools=fake_tool)

    call_kwargs = mock_create.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["tool_choice"] == "auto"


async def test_groq_passes_temperature_and_max_tokens() -> None:
    """temperature and max_tokens should be forwarded."""
    provider = GroqProvider(api_key="sk-test", timeout=5.0)

    mock_create = AsyncMock(return_value=_fake_completion())
    with patch.object(provider._client.chat.completions, "create", mock_create):
        await provider.complete(FAKE_MESSAGES, temperature=0.2, max_tokens=512)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["max_tokens"] == 512


async def test_groq_is_available_with_key() -> None:
    provider = GroqProvider(api_key="sk-test", timeout=5.0)
    assert await provider.is_available() is True


async def test_groq_is_unavailable_without_key() -> None:
    provider = GroqProvider.__new__(GroqProvider)
    provider._api_key = ""
    assert await provider.is_available() is False


# ── OpenRouterProvider tests ──────────────────────────────────────────────────

async def test_openrouter_complete_returns_llm_response() -> None:
    provider = OpenRouterProvider(
        api_key="sk-or-test",
        default_model="anthropic/claude-3.5-sonnet",
        timeout=5.0,
    )

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_fake_completion("Відповідь OR", model="anthropic/claude-3.5-sonnet"),
    ):
        result = await provider.complete(FAKE_MESSAGES)

    assert result.provider == "openrouter"
    assert result.model == "anthropic/claude-3.5-sonnet"
    assert result.content == "Відповідь OR"


async def test_openrouter_has_site_headers() -> None:
    """OpenRouter client should be initialised with HTTP-Referer and X-Title."""
    provider = OpenRouterProvider(api_key="sk-or-test", timeout=5.0)
    headers = provider._client.default_headers
    assert "http-referer" in headers or "HTTP-Referer" in headers
    assert "x-title" in headers or "X-Title" in headers


async def test_openrouter_is_available_with_key() -> None:
    provider = OpenRouterProvider(api_key="sk-or-test", timeout=5.0)
    assert await provider.is_available() is True


async def test_openrouter_is_unavailable_without_key() -> None:
    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider._api_key = ""
    assert await provider.is_available() is False


# ── LLMRouter tests ───────────────────────────────────────────────────────────

def _make_provider(name: str, available: bool, response: LLMResponse | Exception) -> AsyncMock:
    p = AsyncMock(spec=LLMProvider)
    p.name = name
    p.is_available = AsyncMock(return_value=available)
    if isinstance(response, Exception):
        p.complete = AsyncMock(side_effect=response)
    else:
        p.complete = AsyncMock(return_value=response)
    return p


_ok_resp = lambda name: LLMResponse(content="OK", model="m", provider=name)


async def test_router_uses_primary_on_success() -> None:
    primary = _make_provider("primary", True, _ok_resp("primary"))
    fallback = _make_provider("fallback", True, _ok_resp("fallback"))

    router = LLMRouter(providers=[primary, fallback])
    result = await router.complete(FAKE_MESSAGES)

    assert result.provider == "primary"
    fallback.complete.assert_not_awaited()


async def test_router_falls_back_on_connection_error() -> None:
    primary = _make_provider("primary", True, APIConnectionError(request=None))  # type: ignore[arg-type]
    fallback = _make_provider("fallback", True, _ok_resp("fallback"))

    router = LLMRouter(providers=[primary, fallback])
    result = await router.complete(FAKE_MESSAGES)

    assert result.provider == "fallback"


async def test_router_falls_back_on_timeout() -> None:
    mock_req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    primary = _make_provider("primary", True, APITimeoutError(request=mock_req))
    fallback = _make_provider("fallback", True, _ok_resp("fallback"))

    router = LLMRouter(providers=[primary, fallback])
    result = await router.complete(FAKE_MESSAGES)

    assert result.provider == "fallback"


async def test_router_falls_back_on_5xx() -> None:
    mock_req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    mock_resp = httpx.Response(503, request=mock_req)
    primary = _make_provider(
        "primary", True,
        APIStatusError("Service Unavailable", response=mock_resp, body=None)
    )
    fallback = _make_provider("fallback", True, _ok_resp("fallback"))

    router = LLMRouter(providers=[primary, fallback])
    result = await router.complete(FAKE_MESSAGES)

    assert result.provider == "fallback"


async def test_router_reraises_on_4xx() -> None:
    """4xx errors should NOT trigger fallback — re-raised immediately."""
    mock_req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    mock_resp = httpx.Response(401, request=mock_req)
    primary = _make_provider(
        "primary", True,
        APIStatusError("Unauthorized", response=mock_resp, body=None)
    )
    fallback = _make_provider("fallback", True, _ok_resp("fallback"))

    router = LLMRouter(providers=[primary, fallback])

    with pytest.raises(APIStatusError):
        await router.complete(FAKE_MESSAGES)

    fallback.complete.assert_not_awaited()


async def test_router_raises_unavailable_when_all_fail() -> None:
    p1 = _make_provider("p1", True, APIConnectionError(request=None))  # type: ignore[arg-type]
    p2 = _make_provider("p2", True, APIConnectionError(request=None))  # type: ignore[arg-type]

    router = LLMRouter(providers=[p1, p2])
    with pytest.raises(LLMUnavailableError):
        await router.complete(FAKE_MESSAGES)


async def test_router_skips_unavailable_provider() -> None:
    unavailable = _make_provider("no-key", False, _ok_resp("no-key"))
    available = _make_provider("has-key", True, _ok_resp("has-key"))

    router = LLMRouter(providers=[unavailable, available])
    result = await router.complete(FAKE_MESSAGES)

    assert result.provider == "has-key"
    unavailable.complete.assert_not_awaited()


async def test_router_provider_chain_order() -> None:
    """providers property should expose the chain in order."""
    p1 = _make_provider("p1", True, _ok_resp("p1"))
    p2 = _make_provider("p2", True, _ok_resp("p2"))
    router = LLMRouter(providers=[p1, p2])

    assert router.providers[0].name == "p1"
    assert router.providers[1].name == "p2"

