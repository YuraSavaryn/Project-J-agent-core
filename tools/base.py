"""Base interface for all agent tools (function calling).

Each tool maps to one OpenAI-compatible function definition and can be
invoked by the orchestrator when the LLM emits a `tool_call`.
"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for all callable tools.

    Subclasses must define:
    - `name`        — unique identifier used in the OpenAI tool schema.
    - `description` — human-readable description sent to the LLM.
    - `parameters`  — JSON Schema dict describing the function's arguments.
    - `run()`       — the actual implementation that returns a plain string.
    """

    # ── Tool metadata (override in subclass) ──────────────────────────────────

    #: Unique snake_case identifier.  Must match what the LLM sends back.
    name: str

    #: Short description that guides the LLM when to use this tool.
    description: str

    #: JSON Schema `object` describing the function's arguments.
    parameters: dict[str, Any]

    # ── Public API ────────────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute the tool and return the result as a plain string.

        Parameters
        ----------
        **kwargs:
            Arguments parsed from the LLM's `tool_call.function.arguments`
            JSON string.  The caller is responsible for parsing the JSON before
            passing kwargs here.

        Returns
        -------
        str
            A plain-text result that will be sent back to the LLM as a
            `tool` role message.
        """

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
