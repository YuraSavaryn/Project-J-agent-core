"""Base interfaces for the curated wiki."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Article:
    """A high-quality, pre-authored wiki article that directly answers a query."""

    id: str
    content: str
    title: str | None = None
    url: str | None = None


class KnowledgeProvider(ABC):
    """Abstract interface for querying curated wiki content."""

    @abstractmethod
    async def lookup(self, query: str) -> Article | None:
        """Search the curated wiki for a direct answer to the user's query.

        Parameters
        ----------
        query:
            The raw text sent by the user.

        Returns
        -------
        Article | None:
            The matched article if found, otherwise None.
        """
        pass

