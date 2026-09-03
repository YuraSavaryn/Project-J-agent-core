"""Tests for step 7: curated wiki interface and orchestrator integration.

Covers:
- Article dataclass construction
- StubProvider always returns None
- KnowledgeProvider is a proper ABC
- Orchestrator skips wiki context when lookup() returns None (stub behaviour)
- Orchestrator injects wiki content when lookup() returns an Article
- Orchestrator prepends wiki content before any caller-supplied RAG chunks
- Full pipeline still works end-to-end with StubProvider (no regression)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.llm.base import LLMResponse
from core.orchestrator import Orchestrator, OrchestratorResult
from knowledge.curated_wiki.base import Article, KnowledgeProvider
from knowledge.curated_wiki.stub_provider import StubProvider


# ── Article and interface tests ───────────────────────────────────────────────

def test_article_fields() -> None:
    """Article dataclass should store all provided fields."""
    article = Article(
        id="plast-history",
        title="Історія Пласту",
        content="Пласт заснований у 1911 році...",
        url="https://plast.ua/history",
        tags=["history", "plast"],
    )
    assert article.id == "plast-history"
    assert article.title == "Історія Пласту"
    assert "1911" in article.content
    assert article.url == "https://plast.ua/history"
    assert "history" in article.tags


def test_article_defaults() -> None:
    """url and tags should have sensible defaults."""
    article = Article(id="a", title="T", content="C")
    assert article.url is None
    assert article.tags == []


def test_knowledge_provider_is_abstract() -> None:
    """Cannot instantiate KnowledgeProvider directly — it's an ABC."""
    with pytest.raises(TypeError):
        KnowledgeProvider()  # type: ignore[abstract]


async def test_stub_provider_returns_none() -> None:
    """StubProvider.lookup() must always return None."""
    stub = StubProvider()
    result = await stub.lookup("Що таке Пласт?")
    assert result is None


async def test_stub_provider_returns_none_for_any_query() -> None:
    """StubProvider returns None regardless of the query content."""
    stub = StubProvider()
    for query in ["", "Пласт", "scouting", "a" * 500]:
        assert await stub.lookup(query) is None


def test_stub_provider_is_knowledge_provider() -> None:
    """StubProvider must satisfy the KnowledgeProvider interface."""
    stub = StubProvider()
    assert isinstance(stub, KnowledgeProvider)


# ── Orchestrator + wiki integration ──────────────────────────────────────────

def _make_store() -> AsyncMock:
    store = AsyncMock()
    store.get_history = AsyncMock(return_value=[])
    store.append_message = AsyncMock()
    return store


def _make_router(content: str = "Відповідь LLM") -> AsyncMock:
    router = AsyncMock()
    router.complete = AsyncMock(
        return_value=LLMResponse(content=content, model="m", provider="groq")
    )
    return router


async def test_orchestrator_stub_wiki_does_not_add_context() -> None:
    """With StubProvider (returns None), no extra context chunks are added to prompt."""
    router = _make_router()
    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=router,
        knowledge_provider=StubProvider(),
    )

    await orch.handle_message("s1", "Питання без wiki")

    messages = router.complete.call_args.args[0]
    # Should be: [system, user] — no RAG/wiki system block injected
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


async def test_orchestrator_wiki_hit_injects_content_into_prompt() -> None:
    """When wiki returns an Article, its content must appear in the LLM messages."""
    wiki_article = Article(
        id="plast-origin",
        title="Заснування Пласту",
        content="Пласт заснований у Львові 1911 року.",
    )

    mock_wiki = AsyncMock(spec=KnowledgeProvider)
    mock_wiki.lookup = AsyncMock(return_value=wiki_article)

    router = _make_router()
    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=router,
        knowledge_provider=mock_wiki,
    )

    await orch.handle_message("s1", "Коли заснований Пласт?")

    messages = router.complete.call_args.args[0]
    all_content = " ".join(m["content"] for m in messages)
    assert "Пласт заснований у Львові 1911 року." in all_content


async def test_orchestrator_wiki_content_prepended_before_rag_chunks() -> None:
    """Wiki content must appear before caller-supplied RAG chunks in the prompt."""
    wiki_article = Article(id="w1", title="Wiki", content="Wiki content")

    mock_wiki = AsyncMock(spec=KnowledgeProvider)
    mock_wiki.lookup = AsyncMock(return_value=wiki_article)

    router = _make_router()
    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=router,
        knowledge_provider=mock_wiki,
    )

    await orch.handle_message("s1", "Q", context_chunks=["RAG chunk"])

    messages = router.complete.call_args.args[0]
    # Find the context system message
    context_msg = next(
        (m for m in messages if m["role"] == "system" and "Wiki content" in m["content"]),
        None,
    )
    assert context_msg is not None, "Wiki content not found in any system message"
    assert "RAG chunk" in context_msg["content"], "RAG chunk should be in the same context block"
    # Wiki content should appear before RAG chunk in text
    assert context_msg["content"].index("Wiki content") < context_msg["content"].index("RAG chunk")


async def test_orchestrator_wiki_lookup_called_with_user_message() -> None:
    """wiki.lookup() must receive the original user message."""
    mock_wiki = AsyncMock(spec=KnowledgeProvider)
    mock_wiki.lookup = AsyncMock(return_value=None)

    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=_make_router(),
        knowledge_provider=mock_wiki,
    )

    await orch.handle_message("s1", "Хто такий Олег Кандиба?")

    mock_wiki.lookup.assert_awaited_once_with("Хто такий Олег Кандиба?")


async def test_orchestrator_defaults_to_stub_provider() -> None:
    """When no knowledge_provider is supplied, StubProvider should be used."""
    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=_make_router(),
    )
    assert isinstance(orch._wiki, StubProvider)


async def test_orchestrator_pipeline_works_with_stub_end_to_end() -> None:
    """Full pipeline with StubProvider should still return a valid OrchestratorResult."""
    orch = Orchestrator(
        session_store=_make_store(),
        llm_router=_make_router("Все ОК!"),
        knowledge_provider=StubProvider(),
    )

    result = await orch.handle_message("e2e-session", "Питання")

    assert isinstance(result, OrchestratorResult)
    assert result.reply == "Все ОК!"
    assert result.session_id == "e2e-session"

