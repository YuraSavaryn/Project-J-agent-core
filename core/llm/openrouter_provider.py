"""OpenRouter LLM provider — uses the openai SDK with OpenRouter's API."""

import logging
from typing import Any

from openai import AsyncOpenAI

from config.settings import settings
from core.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolDefinition
from core.llm.groq_provider import _extract_tool_calls

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM backend via the OpenAI-compatible API.

    OpenRouter aggregates many models (Claude, GPT-4, Mistral, …) behind a
    single OpenAI-compatible endpoint.  The ``openai`` SDK works with a custom
    ``base_url`` and the site/app metadata headers recommended by OpenRouter.

    Parameters
    ----------
    api_key:
        OpenRouter API key. Defaults to ``settings.openrouter_api_key``.
    base_url:
        OpenRouter base URL. Defaults to ``settings.openrouter_base_url``.
    default_model:
        Model to use when no override is passed to ``complete()``.
    timeout:
        HTTP request timeout in seconds.
    """

    name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.openrouter_api_key
        self._base_url = base_url or settings.openrouter_base_url
        self._default_model = default_model or settings.openrouter_default_model
        self._timeout = timeout or settings.llm_request_timeout_seconds

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            # OpenRouter recommends these headers for analytics/routing
            default_headers={
                "HTTP-Referer": "https://github.com/plastai/agent-core",
                "X-Title": "PlastAI agent-core",
            },
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
        """Send a chat-completion request to OpenRouter."""
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

        logger.debug(
            "OpenRouter request: model=%s, messages=%d", effective_model, len(messages)
        )

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
        """Return True if an API key is configured."""
        return bool(self._api_key)

