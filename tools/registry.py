"""Tool registry — central catalog of available tools.

Usage
-----
Register tools once at startup:

    from tools.registry import ToolRegistry
    from tools.search_knowledge_tool import SearchKnowledgeTool

    registry = ToolRegistry()
    registry.register(SearchKnowledgeTool(retriever=retriever))

Then pass to the orchestrator:

    orchestrator = Orchestrator(..., tool_registry=registry)
"""

import logging
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """A simple dict-backed registry for agent tools.

    Tools are keyed by `tool.name` so lookups are O(1).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Parameters
        ----------
        tool:
            Any concrete `Tool` subclass.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"ToolRegistry: tool {tool.name!r} is already registered.  "
                "Use replace() if you intentionally want to overwrite it."
            )
        self._tools[tool.name] = tool
        logger.debug("ToolRegistry: registered tool %r", tool.name)

    def replace(self, tool: Tool) -> None:
        """Register or overwrite a tool (useful in tests)."""
        self._tools[tool.name] = tool
        logger.debug("ToolRegistry: (re)registered tool %r", tool.name)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Tool | None:
        """Return the tool with *name*, or `None` if not found."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools (stable insertion order)."""
        return list(self._tools.values())

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
