"""SearchKnowledgeTool — wraps the RAG Retriever as an agent tool.

The LLM can call this tool when it needs factual context from the
indexed knowledge base.  Results are returned as a formatted text
block the LLM can use in its next turn.
"""

import json
import logging
from typing import Any

from knowledge.rag.retriever import Retriever
from tools.base import Tool

logger = logging.getLogger(__name__)


class SearchKnowledgeTool(Tool):
    """Tool that queries the vector knowledge base via the RAG Retriever.

    Parameters
    ----------
    retriever:
        A configured `Retriever` instance.
    top_k:
        Number of chunks to retrieve per query.  Defaults to `settings.rag_top_k`.
    """

    name = "search_knowledge"
    description = (
        "Пошук у базі знань організації Пласт.  "
        "Використовуй цей інструмент, щоб знайти конкретні факти, "
        "статути, документи або інструкції з проіндексованих матеріалів."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Пошуковий запит українською мовою.",
            },
            "top_k": {
                "type": "integer",
                "description": "Кількість результатів (за замовчуванням 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever, top_k: int | None = None) -> None:
        self._retriever = retriever
        self._default_top_k = top_k

    async def run(self, query: str, top_k: int | None = None, **_: Any) -> str:
        """Retrieve relevant chunks and format them as a numbered list.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            How many results to return. Falls back to the instance default.

        Returns
        -------
        str
            A formatted block of retrieved chunks, or a "not found" message.
        """
        k = top_k or self._default_top_k
        logger.info("SearchKnowledgeTool: query=%r top_k=%s", query, k)

        results = await self._retriever.retrieve(query=query, top_k=k)

        if not results:
            return "Релевантних матеріалів у базі знань не знайдено."

        parts: list[str] = [f"Знайдено {len(results)} результат(ів):"]
        for qr in results:
            title = qr.metadata.get("title") or "Без назви"
            url = qr.metadata.get("source_url") or ""
            text = qr.document or qr.metadata.get("content_preview") or ""
            header = f"[{qr.rank}] {title}"
            if url:
                header += f" ({url})"
            parts.append(f"{header}\n{text.strip()}")

        return "\n\n".join(parts)
