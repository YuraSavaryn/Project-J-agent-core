"""Agent orchestrator — the main pipeline that handles a user message.

Pipeline (step 12 — tool-calling loop added):
  1. Load conversation history from SessionStore.
  2. Curated wiki lookup (or RAG retrieval if wiki misses).
  3. Build the messages list (system + history + [context] + user message).
  4. Call LLMRouter; if tool_calls are present → execute tools → re-call LLM.
     Repeat until the LLM returns a plain reply (no more tool_calls).
  5. Persist both the user message and assistant reply to SessionStore.
  6. Return an OrchestratorResult with the reply and sources.

Future steps:
  - Step 13: query logging to PostgreSQL
"""

import json
import logging
import time
from dataclasses import dataclass, field

from core.llm.base import ChatMessage, LLMResponse
from core.llm.router import LLMRouter, LLMUnavailableError
from core.memory.session_store import MessageDict, SessionStore
from core.prompts.builder import build_messages
from core.db.enums import AnswerSource
from knowledge.curated_wiki.base import Article, KnowledgeProvider
from knowledge.curated_wiki.stub_provider import StubProvider
from knowledge.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# Maximum number of tool-call iterations to prevent infinite loops.
_MAX_TOOL_ITERATIONS = 5


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
    knowledge_provider:
        Curated wiki provider.  Defaults to ``StubProvider`` (always returns None).
        Inject a real provider in production or a mock in tests.
    retriever:
        Optional RAG retriever used when the wiki misses.
    tool_registry:
        Optional registry of callable tools for function-calling.
        When provided, tool schemas are sent to the LLM and any returned
        ``tool_call`` responses are executed in a loop before the final reply.
    query_logger:
        Optional logger to record analytics into Postgres.
    history_limit:
        Maximum number of recent messages to include in the prompt.
        Keeps token usage bounded for long conversations.
    """

    def __init__(
        self,
        session_store: SessionStore,
        llm_router: LLMRouter,
        knowledge_provider: KnowledgeProvider | None = None,
        retriever: "Retriever | None" = None,
        tool_registry: "ToolRegistry | None" = None,
        query_logger: "QueryLogger | None" = None,
        history_limit: int = 20,
    ) -> None:
        self._store = session_store
        self._router = llm_router
        self._wiki = knowledge_provider or StubProvider()
        self._retriever = retriever
        self._tools = tool_registry
        self._query_logger = query_logger
        self._history_limit = history_limit

    async def handle_message(
        self,
        session_id: str,
        user_message: str,
        context_chunks: list[str] | None = None,
        channel: str = "web",
        external_user_id: str = "anonymous",
    ) -> OrchestratorResult:
        """Run the full pipeline for a user message and return the result.

        Parameters
        ----------
        session_id:
            Unique session identifier.
        user_message:
            The user's message text.
        context_chunks:
            Optional pre-built RAG context chunks to inject into the prompt.
        channel:
            Source channel (``"web"``, ``"telegram"``, …). Used to create the
            ``sessions`` row on first use.
        external_user_id:
            External user identifier (Telegram chat_id, web cookie, …). Used
            to create the ``sessions`` row on first use.
        """
        logger.info("Orchestrator: session=%s, message_len=%d", session_id, len(user_message))
        start_time = time.perf_counter()
        answer_source = AnswerSource.fallback

        # ── 1. Load history ───────────────────────────────────────────────────
        history: list[MessageDict] = await self._store.get_history(
            session_id, limit=self._history_limit
        )

        # ── 2. Curated wiki lookup ────────────────────────────────────────────
        wiki_article: Article | None = await self._wiki.lookup(user_message)

        # Merge wiki content with any caller-supplied RAG chunks
        extra_chunks: list[str] = list(context_chunks or [])
        sources_list: list[dict] = []

        if wiki_article is not None:
            logger.info(
                "Orchestrator: curated wiki hit — article=%r (session=%s)",
                wiki_article.id,
                session_id,
            )
            answer_source = AnswerSource.curated
            extra_chunks.insert(0, wiki_article.content)
            sources_list.append({
                "document_id": wiki_article.id,
                "title": wiki_article.title or "Вікі-стаття",
                "url": wiki_article.url,
                "excerpt": wiki_article.content[:200] + "..." if len(wiki_article.content) > 200 else wiki_article.content,
                "score": 1.0,
                "rank": 1,
            })
        elif self._retriever is not None:
            # ── 2b. Vector DB RAG lookup (if wiki missed) ─────────────────────
            logger.info("Orchestrator: wiki miss, querying vector store (session=%s)", session_id)
            query_results = await self._retriever.retrieve(query=user_message)
            
            if query_results:
                answer_source = AnswerSource.rag

            for qr in query_results:
                chunk_text = qr.document or qr.metadata.get("content_preview") or ""
                extra_chunks.append(chunk_text)
                sources_list.append({
                    "document_id": str(qr.metadata.get("document_id", qr.id)),
                    "title": qr.metadata.get("title") or "RAG Документ",
                    "url": qr.metadata.get("source_url"),
                    "excerpt": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "score": qr.score,
                    "rank": qr.rank,
                    "chunk_id": qr.id,
                })

        # ── 3. Build initial prompt ───────────────────────────────────────────
        messages: list[ChatMessage] = build_messages(
            history=history,
            user_message=user_message,
            context_chunks=extra_chunks,
        )

        # ── 4. LLM call + tool-calling loop ──────────────────────────────────
        tool_schemas = self._tools.to_openai_schemas() if self._tools else None

        llm_response: LLMResponse | None = None
        for iteration in range(_MAX_TOOL_ITERATIONS):
            try:
                llm_response = await self._router.complete(
                    messages, tools=tool_schemas or None
                )
            except LLMUnavailableError:
                logger.error(
                    "Orchestrator: all LLM providers unavailable for session=%s", session_id
                )
                raise

            # No tool calls — the LLM gave a final answer, exit the loop.
            if not llm_response.tool_calls:
                logger.debug(
                    "Orchestrator: no tool_calls in iteration %d, done.", iteration
                )
                break

            # ── Execute tool calls ────────────────────────────────────────────
            logger.info(
                "Orchestrator: %d tool_call(s) in iteration %d (session=%s)",
                len(llm_response.tool_calls),
                iteration,
                session_id,
            )

            # Append the assistant's tool-call message to the thread
            messages.append({
                "role": "assistant",
                "content": llm_response.content or "",
                "tool_calls": llm_response.tool_calls,  # type: ignore[typeddict-unknown-key]
            })

            for tc in llm_response.tool_calls:
                tool_name: str = tc["function"]["name"]
                tool_args: dict = json.loads(tc["function"]["arguments"] or "{}")
                tool_call_id: str = tc["id"]

                tool = self._tools.get(tool_name) if self._tools else None
                if tool is None:
                    tool_result = f"[Помилка] Інструмент {tool_name!r} не знайдено."
                    logger.warning("Orchestrator: unknown tool %r", tool_name)
                else:
                    try:
                        tool_result = await tool.run(**tool_args)
                        logger.debug(
                            "Orchestrator: tool %r returned %d chars", tool_name, len(tool_result)
                        )
                    except Exception as exc:  # noqa: BLE001
                        tool_result = f"[Помилка виконання інструменту {tool_name!r}]: {exc}"
                        logger.exception("Orchestrator: tool %r raised an error", tool_name)

                # Append tool result as a 'tool' message
                messages.append({
                    "role": "tool",  # type: ignore[typeddict-unknown-key]
                    "tool_call_id": tool_call_id,  # type: ignore[typeddict-unknown-key]
                    "content": tool_result,
                })
        else:
            # Exhausted iterations without a plain reply — use whatever we have.
            logger.warning(
                "Orchestrator: reached max tool iterations (%d) for session=%s",
                _MAX_TOOL_ITERATIONS,
                session_id,
            )

        assert llm_response is not None  # always set after at least one iteration
        reply = llm_response.content
        logger.info(
            "Orchestrator: reply from provider=%s model=%s (session=%s)",
            llm_response.provider,
            llm_response.model,
            session_id,
        )

        # ── 5. Persist to session store ───────────────────────────────────────
        await self._store.append_message(
            session_id, role="user", content=user_message,
            channel=channel, external_user_id=external_user_id,
        )
        await self._store.append_message(
            session_id, role="assistant", content=reply,
            channel=channel, external_user_id=external_user_id,
        )

        # ── 6. Log query analytics ────────────────────────────────────────────
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        query_log_id = None
        if self._query_logger:
            try:
                log_entry = await self._query_logger.log_query(
                    session_id=session_id,
                    query_text=user_message,
                    source=answer_source,
                    latency_ms=latency_ms,
                    llm_provider=llm_response.provider,
                    llm_model=llm_response.model,
                    retrieved_chunks=sources_list,
                )
                query_log_id = log_entry.id
            except Exception:
                logger.exception("Orchestrator: failed to log query analytics")

        # ── 7. Return result ──────────────────────────────────────────────────
        return OrchestratorResult(
            reply=reply,
            session_id=session_id,
            sources=sources_list,
            suggested_actions=[],
            llm_provider=llm_response.provider,
            llm_model=llm_response.model,
        )

