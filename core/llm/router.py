"""LLM router — selects the primary provider and falls back on failure.

Flow
----
1. Build the provider chain from ``settings.llm_primary_provider``.
   Primary goes first; the other provider is the fallback.
2. ``complete()`` tries the primary provider.
3. On ``APIConnectionError``, ``APITimeoutError``, or ``APIStatusError``
   with a 5xx status → log a warning and try the fallback.
4. If all providers fail → raise ``LLMUnavailableError``.

This is intentionally simple (two fixed providers).  A more sophisticated
router (round-robin, cost-based, model-selection) can replace this later.
"""

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
    """Selects and calls LLM providers with automatic fallback.

    Parameters
    ----------
    providers:
        Ordered list of providers to try.  The first provider is primary;
        the rest are fallbacks tried in order.  If ``None``, the list is
        built from ``settings.llm_primary_provider``.
    """

    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        if providers is not None:
            self._providers = providers
        else:
            self._providers = _build_provider_chain()

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
        """Try providers in order; return the first successful response.

        Raises
        ------
        LLMUnavailableError
            If every provider in the chain fails.
        """
        last_exc: Exception | None = None

        for provider in self._providers:
            if not await provider.is_available():
                logger.warning(
                    "LLMRouter: provider %r is not available (no API key?), skipping.",
                    provider.name,
                )
                continue

            try:
                logger.debug("LLMRouter: trying provider %r", provider.name)
                response = await provider.complete(
                    messages,
                    tools,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                logger.info(
                    "LLMRouter: success via %r (model=%s)", provider.name, response.model
                )
                return response

            except (APIConnectionError, APITimeoutError) as exc:
                logger.warning(
                    "LLMRouter: provider %r connection/timeout error — trying next. %s",
                    provider.name,
                    exc,
                )
                last_exc = exc

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    logger.warning(
                        "LLMRouter: provider %r returned %d — trying next. %s",
                        provider.name,
                        exc.status_code,
                        exc,
                    )
                    last_exc = exc
                else:
                    # 4xx errors (bad key, quota, bad request) — re-raise immediately,
                    # no point in trying the next provider with the same bad input.
                    raise

        raise LLMUnavailableError(
            "All LLM providers failed or are unavailable."
        ) from last_exc

    @property
    def providers(self) -> list[LLMProvider]:
        """Expose the provider chain (useful for introspection in tests)."""
        return list(self._providers)


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

