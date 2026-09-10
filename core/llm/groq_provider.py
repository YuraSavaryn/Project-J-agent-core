"""Groq LLM provider — uses the openai SDK with Groq's OpenAI-compatible API."""

import logging
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from config.settings import settings
from core.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolDefinition

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq LLM backend via the OpenAI-compatible API.

    Groq exposes an OpenAI-compatible endpoint, so the ``openai`` SDK works
    out of the box with a custom ``base_url``.

    Parameters
    ----------
    api_key:
        Groq API key. Defaults to ``settings.groq_api_key``.
    base_url:
        Groq API base URL. Defaults to ``settings.groq_base_url``.
    default_model:
        Model name to use when no override is passed to ``complete()``.
    timeout:
        HTTP request timeout in seconds.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.groq_api_key
        self._base_url = base_url or settings.groq_base_url
        self._default_model = default_model or settings.groq_default_model
        self._timeout = timeout or settings.llm_request_timeout_seconds

        self._client = AsyncOpenAI(
            api_key=self._api_key or "dummy_key",
            base_url=self._base_url,
            timeout=self._timeout,
        )

    # ── LLMProvider interface ─────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request to Groq."""
        effective_model = model or self._default_model

        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools  # type: ignore[assignment]
            kwargs["tool_choice"] = "auto"

        logger.debug("Groq request: model=%s, messages=%d", effective_model, len(messages))

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        tool_calls = _extract_tool_calls(choice)

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.name,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )

    async def is_available(self) -> bool:
        """Return True if an API key is configured (no live check — saves latency)."""
        return bool(self._api_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_tool_calls(choice: Any) -> list[dict[str, Any]]:
    """Parse tool_calls from a completion choice into a plain list of dicts."""
    if not choice.message.tool_calls:
        return []
    result = []
    for tc in choice.message.tool_calls:
        result.append(
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )
    return result

