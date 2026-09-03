"""Agent orchestrator — the main pipeline that handles a user message.

Pipeline (step 6 — no RAG, no tools yet):
  1. Load conversation history from SessionStore.
  2. Build the messages list (system + history + user message) via prompt builder.
  3. Call LLMRouter to get the model's reply.
  4. Persist both the user message and assistant reply to SessionStore.
  5. Return an OrchestratorResult with the reply.

Future steps will extend this pipeline:
  - Step 7:  curated wiki lookup (before LLM call)
  - Step 11: RAG retrieval (inject context_chunks into build_messages)
  - Step 12: tool-calling loop
  - Step 13: query logging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.llm.base import LLMResponse
from core.llm.router import LLMRouter, LLMUnavailableError
from core.memory.session_store import MessageDict, SessionStore
from core.prompts.builder import build_messages

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Value object returned by ``handle_message()``."""

    reply: str
    session_id: str
    sources: list[dict] = field(default_factory=list)
    suggested_actions: list[dict] = field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None


class Orchestrator:
    """Coordinates the full agent pipeline for a single request.

    Parameters
    ----------
    session_store:
        Provides conversation history (Redis + Postgres).
    llm_router:
        Routes LLM calls across available providers with fallback.
    history_limit:
        Maximum number of recent messages to include in the prompt.
        Keeps token usage bounded for long conversations.
    """

    def __init__(
        self,
        session_store: SessionStore,
        llm_router: LLMRouter,
        history_limit: int = 20,
    ) -> None:
        self._store = session_store
        self._router = llm_router
        self._history_limit = history_limit

    async def handle_message(
        self,
        session_id: str,
        user_message: str,
        context_chunks: list[str] | None = None,
    ) -> OrchestratorResult:
        """Run the full pipeline for a user message and return the result.

        Parameters
        ----------
        session_id:
            Unique conversation identifier.
        user_message:
            The text sent by the user.
        context_chunks:
            Optional RAG-retrieved text snippets (injected into the prompt).
            Empty / None → no context injected (step 6 behaviour).
        """
        logger.info("Orchestrator: session=%s, message_len=%d", session_id, len(user_message))

        # ── 1. Load history ───────────────────────────────────────────────────
        history: list[MessageDict] = await self._store.get_history(
            session_id, limit=self._history_limit
        )

        # ── 2. Build prompt ───────────────────────────────────────────────────
        messages = build_messages(
            history=history,
            user_message=user_message,
            context_chunks=context_chunks or [],
        )

        # ── 3. Call LLM ───────────────────────────────────────────────────────
        try:
            llm_response: LLMResponse = await self._router.complete(messages)
        except LLMUnavailableError:
            logger.error(
                "Orchestrator: all LLM providers unavailable for session=%s", session_id
            )
            raise

        reply = llm_response.content
        logger.info(
            "Orchestrator: reply from provider=%s model=%s (session=%s)",
            llm_response.provider,
            llm_response.model,
            session_id,
        )

        # ── 4. Persist to session store ───────────────────────────────────────
        await self._store.append_message(session_id, role="user", content=user_message)
        await self._store.append_message(session_id, role="assistant", content=reply)

        # ── 5. Return result ──────────────────────────────────────────────────
        return OrchestratorResult(
            reply=reply,
            session_id=session_id,
            sources=[],            # filled in step 11 (RAG)
            suggested_actions=[],  # filled in future steps
            llm_provider=llm_response.provider,
            llm_model=llm_response.model,
        )

