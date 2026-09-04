"""Abstract base class for LLM providers.

All providers (Groq, OpenRouter, …) implement this interface so the
orchestrator and tests are decoupled from any specific vendor SDK.
"""

from abc import ABC, abstractmethod
from typing import Any


# ── Type aliases ──────────────────────────────────────────────────────────────

# A single chat message dict, e.g. {"role": "user", "content": "Привіт!"}
ChatMessage = dict[str, str]

# OpenAI-style tool/function definition schema
ToolDefinition = dict[str, Any]


class LLMResponse:
    """Structured response returned by every provider's ``complete()`` call."""

    __slots__ = ("content", "model", "provider", "tool_calls", "input_tokens", "output_tokens")

    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        tool_calls: list[dict[str, Any]] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.provider = provider
        self.tool_calls: list[dict[str, Any]] = tool_calls or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LLMResponse(provider={self.provider!r}, model={self.model!r}, "
            f"content={self.content[:60]!r}…)"
        )


class LLMProvider(ABC):
    """Abstract LLM provider interface.

    Concrete implementations wrap different backends (Groq, OpenRouter, …)
    but all expose the same ``complete()`` method so the orchestrator stays
    vendor-agnostic.

    Attributes
    ----------
    name:
        Short identifier used in logs and ``LLMResponse.provider``.
    """

    name: str = "unknown"

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request and return the response.

        Parameters
        ----------
        messages:
            Ordered list of chat messages (OpenAI message format).
        tools:
            Optional list of tool/function definitions for function calling.
        model:
            Override the provider's default model for this call.
        temperature:
            Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens:
            Maximum number of tokens in the completion.
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if the provider is reachable and has a configured API key.

        Used by the LLM router to decide whether to attempt this provider
        before falling back to the next one.
        """
        ...

