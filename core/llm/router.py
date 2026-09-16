"""LLM router — selects the primary provider and falls back on failure.

Flow
----
1. Build the provider chain from ``settings.llm_primary_provider``.
   Primary goes first; the other provider is the fallback.
2. ``complete()`` tries the primary provider with up to
   ``settings.llm_retry_attempts`` retries (exponential backoff).
3. On ``APIConnectionError``, ``APITimeoutError``, or ``APIStatusError``
   with a 5xx status → retry within provider, then log a warning and try
   the fallback.
4. If all providers fail → raise ``LLMUnavailableError``.

This is intentionally simple (two fixed providers).  A more sophisticated
router (round-robin, cost-based, model-selection) can replace this later.
"""

import asyncio
import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError

from config.settings import settings
from core.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolDefinition
from core.llm.groq_provider import GroqProvider
from core.llm.openrouter_provider import OpenRouterProvider

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when all configured LLM providers are unavailable or failed."""


class LLMRouter:
    """Selects and calls LLM providers with automatic fallback and retry.

    Parameters
    ----------
    providers:
        Ordered list of providers to try.  The first provider is primary;
        the rest are fallbacks tried in order.  If ``None``, the list is
        built from ``settings.llm_primary_provider``.
    retry_attempts:
        Number of retry attempts *per provider* on transient errors
        (connection errors, timeouts, 5xx).  Defaults to
        ``settings.llm_retry_attempts``.
    retry_backoff_seconds:
        Base delay in seconds between retries; doubles each attempt
        (exponential backoff).  Defaults to ``settings.llm_retry_backoff_seconds``.
    """

    def __init__(
        self,
        providers: list[LLMProvider] | None = None,
        retry_attempts: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        if providers is not None:
            self._providers = providers
        else:
            self._providers = _build_provider_chain()
        self._retry_attempts = (
            retry_attempts
            if retry_attempts is not None
            else settings.llm_retry_attempts
        )
        self._retry_backoff = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else settings.llm_retry_backoff_seconds
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Try providers in order with per-provider retry; return first success.

        Raises
        ------
        LLMUnavailableError
            If every provider in the chain fails all retry attempts.
        """
        last_exc: Exception | None = None

        for provider in self._providers:
            if not await provider.is_available():
                logger.warning(
                    "LLMRouter: provider %r is not available (no API key?), skipping.",
                    provider.name,
                )
                continue

            result = await self._try_provider_with_retry(
                provider, messages, tools, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )
            if isinstance(result, Exception):
                last_exc = result
                continue
            return result

        raise LLMUnavailableError(
            "All LLM providers failed or are unavailable."
        ) from last_exc

    @property
    def providers(self) -> list[LLMProvider]:
        """Expose the provider chain (useful for introspection in tests)."""
        return list(self._providers)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _try_provider_with_retry(
        self,
        provider: LLMProvider,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None,
        *,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse | Exception:
        """Attempt the provider up to ``_retry_attempts`` times with backoff.

        Returns the ``LLMResponse`` on success, or the last ``Exception`` if
        all attempts failed (so the caller can try the next provider).
        """
        last_exc: Exception | None = None
        backoff = self._retry_backoff

        for attempt in range(1, self._retry_attempts + 1):
            try:
                logger.debug(
                    "LLMRouter: trying provider %r (attempt %d/%d)",
                    provider.name,
                    attempt,
                    self._retry_attempts,
                )
                response = await provider.complete(
                    messages,
                    tools,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.info(
                    "LLMRouter: success via %r (model=%s, attempt=%d)",
                    provider.name,
                    response.model,
                    attempt,
                )
                return response

            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                logger.warning(
                    "LLMRouter: provider %r connection/timeout error (attempt %d/%d): %s",
                    provider.name,
                    attempt,
                    self._retry_attempts,
                    exc,
                )

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    logger.warning(
                        "LLMRouter: provider %r returned %d (attempt %d/%d): %s",
                        provider.name,
                        exc.status_code,
                        attempt,
                        self._retry_attempts,
                        exc,
                    )
                else:
                    # 4xx → don't retry, don't fall back — bad request/key
                    raise

            if attempt < self._retry_attempts:
                logger.debug(
                    "LLMRouter: waiting %.2fs before retry (provider=%r)",
                    backoff,
                    provider.name,
                )
                await asyncio.sleep(backoff)
                backoff *= 2  # exponential backoff

        logger.warning(
            "LLMRouter: provider %r exhausted %d attempt(s) — trying next provider.",
            provider.name,
            self._retry_attempts,
        )
        assert last_exc is not None
        return last_exc


# ── Factory helpers ───────────────────────────────────────────────────────────

def _build_provider_chain() -> list[LLMProvider]:
    """Build the [primary, fallback] chain from settings."""
    all_providers: dict[str, LLMProvider] = {
        "groq": GroqProvider(),
        "openrouter": OpenRouterProvider(),
    }

    primary = settings.llm_primary_provider.lower()
    if primary not in all_providers:
        raise ValueError(
            f"Unknown LLM_PRIMARY_PROVIDER={primary!r}. Choose 'groq' or 'openrouter'."
        )

    # Put primary first, fallback second
    ordered = [primary] + [k for k in all_providers if k != primary]
    return [all_providers[name] for name in ordered]


def get_llm_router() -> LLMRouter:
    """FastAPI dependency / convenience factory for ``LLMRouter``."""
    return LLMRouter()
