"""Stub implementation of KnowledgeProvider — always returns None.

This stub keeps the pipeline intact while the real curated wiki is being
built.  It satisfies the ``KnowledgeProvider`` interface contract so the
orchestrator can call it unconditionally without any conditional checks.

Replace this with a real provider once the wiki content is ready.
"""

from __future__ import annotations

import logging

from knowledge.curated_wiki.base import Article, KnowledgeProvider

logger = logging.getLogger(__name__)


class StubProvider(KnowledgeProvider):
    """No-op knowledge provider — always signals "no match found".

    Returning ``None`` causes the orchestrator to fall through to the
    next knowledge source (RAG retrieval) or directly to the LLM.
    """

    async def lookup(self, query: str) -> Article | None:
        """Always return ``None`` — no curated article available yet."""
        logger.debug("StubProvider: no curated article for query=%r", query[:80])
        return None

