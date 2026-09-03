"""Tests for step 6: prompt builder and orchestrator pipeline.

- ``build_messages``: verifies message structure and context injection.
- ``Orchestrator.handle_message``: verifies the pipeline stages using mocked
  SessionStore and LLMRouter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.llm.base import LLMResponse
from core.llm.router import LLMUnavailableError
from core.orchestrator import Orchestrator, OrchestratorResult
from core.prompts.builder import build_messages, _format_context
from core.prompts.system_prompt import get_system_prompt


# ── build_messages tests ──────────────────────────────────────────────────────

def test_build_messages_structure_no_history() -> None:
    """Without history or context, result is [system, user]."""
    msgs = build_messages(history=[], user_message="Привіт!")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Привіт!"


def test_build_messages_includes_history() -> None:
    """History messages are placed between system and current user turn."""
    history = [
        {"role": "user", "content": "Питання 1"},
        {"role": "assistant", "content": "Відповідь 1"},
    ]
    msgs = build_messages(history=history, user_message="Питання 2")

    assert len(msgs) == 4  # system + 2 history + user
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Питання 1"
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["content"] == "Питання 2"


def test_build_messages_injects_rag_context() -> None:
    """RAG chunks should appear as a system message before the user turn."""
    msgs = build_messages(
        history=[],
        user_message="Що таке Пласт?",
        context_chunks=["Пласт — скаутська організація.", "Заснована у 1911 році."],
    )

    # [system_persona, system_context, user]
    assert len(msgs) == 3
    assert msgs[1]["role"] == "system"
    assert "Пласт — скаутська організація." in msgs[1]["content"]
    assert "Заснована у 1911 році." in msgs[1]["content"]
    assert msgs[2]["role"] == "user"


def test_build_messages_empty_context_not_injected() -> None:
    """Empty context_chunks list must NOT add a system context message."""
    msgs = build_messages(history=[], user_message="Q", context_chunks=[])

    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]


def test_build_messages_system_prompt_override() -> None:
    """Custom system_prompt override replaces the default."""
    msgs = build_messages(history=[], user_message="Q", system_prompt="Custom prompt")

    assert msgs[0]["content"] == "Custom prompt"


def test_format_context_numbers_chunks() -> None:
    chunks = ["Chunk A", "Chunk B"]
    result = _format_context(chunks)

    assert "[Джерело 1]" in result
    assert "[Джерело 2]" in result
    assert "Chunk A" in result
    assert "Chunk B" in result


def test_system_prompt_is_non_empty() -> None:
    assert len(get_system_prompt()) > 50


# ── Orchestrator tests ────────────────────────────────────────────────────────

def _make_store(history: list | None = None) -> AsyncMock:
    store = AsyncMock()
    store.get_history = AsyncMock(return_value=history or [])
    store.append_message = AsyncMock()
    return store


def _make_router(content: str = "Відповідь LLM") -> AsyncMock:
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="llama-3.1-70b",
            provider="groq",
        )
    )
    return router


async def test_orchestrator_returns_result() -> None:
    """handle_message should return an OrchestratorResult with the LLM reply."""
    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=_make_router("Привіт!"),
    )
    result = await orch.handle_message("session-1", "Привіт")

    assert isinstance(result, OrchestratorResult)
    assert result.reply == "Привіт!"
    assert result.session_id == "session-1"
    assert result.llm_provider == "groq"
    assert result.llm_model == "llama-3.1-70b"


async def test_orchestrator_loads_history() -> None:
    """Orchestrator must call get_history with the correct session_id."""
    store = _make_store()
    orch = Orchestrator(session_store=store, llm_router=_make_router())

    await orch.handle_message("sess-42", "Q")

    store.get_history.assert_awaited_once_with("sess-42", limit=20)


async def test_orchestrator_persists_user_and_assistant_messages() -> None:
    """Both user and assistant messages must be persisted after a successful call."""
    store = _make_store()
    orch = Orchestrator(session_store=store, llm_router=_make_router("Reply"))

    await orch.handle_message("s1", "User message")

    calls = store.append_message.await_args_list
    assert len(calls) == 2

    first_call = calls[0]
    assert first_call.kwargs["role"] == "user"
    assert first_call.kwargs["content"] == "User message"

    second_call = calls[1]
    assert second_call.kwargs["role"] == "assistant"
    assert second_call.kwargs["content"] == "Reply"


async def test_orchestrator_passes_history_to_prompt() -> None:
    """History returned by the store must appear in the messages sent to the LLM."""
    history = [
        {"role": "user", "content": "Попереднє питання"},
        {"role": "assistant", "content": "Попередня відповідь"},
    ]
    store = _make_store(history=history)
    router = _make_router()
    orch = Orchestrator(session_store=store, llm_router=router)

    await orch.handle_message("s1", "Нове питання")

    messages_sent = router.complete.call_args.args[0]
    contents = [m["content"] for m in messages_sent]
    assert "Попереднє питання" in contents
    assert "Попередня відповідь" in contents
    assert "Нове питання" in contents


async def test_orchestrator_reraises_llm_unavailable() -> None:
    """LLMUnavailableError from the router should propagate to the caller."""
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=LLMUnavailableError("no providers"))

    orch = Orchestrator(session_store=_make_store(), llm_router=router)

    with pytest.raises(LLMUnavailableError):
        await orch.handle_message("s1", "Question")


async def test_orchestrator_respects_history_limit() -> None:
    """Orchestrator should request only history_limit messages from the store."""
    store = _make_store()
    orch = Orchestrator(
        session_store=store,
        llm_router=_make_router(),
        history_limit=5,
    )

    await orch.handle_message("s1", "Q")

    store.get_history.assert_awaited_once_with("s1", limit=5)


async def test_orchestrator_passes_context_chunks_to_prompt() -> None:
    """context_chunks must appear in the messages sent to the LLM."""
    router = _make_router()
    orch = Orchestrator(session_store=_make_store(), llm_router=router)

    await orch.handle_message("s1", "Q", context_chunks=["RAG chunk text"])

    messages_sent = router.complete.call_args.args[0]
    all_content = " ".join(m["content"] for m in messages_sent)
    assert "RAG chunk text" in all_content

