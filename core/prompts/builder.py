"""Prompt builder — assembles the full messages list sent to the LLM.

Responsibility: take raw ingredients (system prompt, session history, current
user message, and optional RAG context) and return an OpenAI-format messages
list ready to pass to ``LLMRouter.complete()``.

This is the single place that controls message ordering and context injection.
RAG chunks are injected as a ``system`` message right before the user turn so
the LLM sees them as authoritative context without polluting the conversation
history.
"""

from __future__ import annotations

from core.llm.base import ChatMessage
from core.memory.session_store import MessageDict
from core.prompts.system_prompt import get_system_prompt


def build_messages(
    history: list[MessageDict],
    user_message: str,
    context_chunks: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """Assemble the full messages list for an LLM call.

    Layout
    ------
    1. ``system`` — agent persona & instructions (from ``get_system_prompt()``
       or the ``system_prompt`` override).
    2. ``user`` / ``assistant`` — conversation history (ordered oldest → newest).
    3. (Optional) ``system`` — RAG context block, injected just before the
       current user turn so the model treats it as the most relevant context.
    4. ``user`` — the current user message.

    Parameters
    ----------
    history:
        Past messages for this session, as returned by ``SessionStore.get_history()``.
        Each item must have ``"role"`` and ``"content"`` keys.
    user_message:
        The new user message to append.
    context_chunks:
        Plain-text snippets retrieved from the knowledge base (RAG).
        When provided, they are injected as a ``system`` context block.
    system_prompt:
        Override the default system prompt (useful for testing or A/B variants).

    Returns
    -------
    list[ChatMessage]
        Ready-to-send messages list in OpenAI chat format.
    """
    messages: list[ChatMessage] = []

    # 1. System persona
    messages.append({
        "role": "system",
        "content": system_prompt or get_system_prompt(),
    })

    # 2. Conversation history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 3. RAG context (injected as a privileged system message)
    if context_chunks:
        context_text = _format_context(context_chunks)
        messages.append({
            "role": "system",
            "content": (
                "Використай наступний контекст для відповіді на запит користувача.\n"
                "Якщо контекст не стосується питання — проігноруй його.\n\n"
                f"{context_text}"
            ),
        })

    # 4. Current user turn
    messages.append({"role": "user", "content": user_message})

    return messages


def _format_context(chunks: list[str]) -> str:
    """Format RAG chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Джерело {i}]\n{chunk.strip()}")
    return "\n\n".join(parts)

