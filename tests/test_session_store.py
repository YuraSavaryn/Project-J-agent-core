"""Unit tests for SessionStore using fakeredis (no real Redis required)."""

from __future__ import annotations

import pytest
import fakeredis.aioredis as fakeredis

from core.memory.session_store import SessionStore


@pytest.fixture
async def store() -> SessionStore:
    """SessionStore backed by an in-memory fakeredis instance."""
    redis = fakeredis.FakeRedis(decode_responses=False)
    return SessionStore(redis, ttl=60, max_history=10)


# ── append_message / get_history ─────────────────────────────────────────────


async def test_empty_history(store: SessionStore) -> None:
    history = await store.get_history("session-x")
    assert history == []


async def test_append_and_retrieve(store: SessionStore) -> None:
    await store.append_message("s1", "user", "Привіт!")
    await store.append_message("s1", "assistant", "Привіт, чим можу допомогти?")

    history = await store.get_history("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Привіт!"}
    assert history[1] == {"role": "assistant", "content": "Привіт, чим можу допомогти?"}


async def test_get_history_with_limit(store: SessionStore) -> None:
    for i in range(5):
        await store.append_message("s2", "user", f"message {i}")

    last_2 = await store.get_history("s2", limit=2)
    assert len(last_2) == 2
    assert last_2[0]["content"] == "message 3"
    assert last_2[1]["content"] == "message 4"


async def test_sessions_are_isolated(store: SessionStore) -> None:
    await store.append_message("session-a", "user", "від A")
    await store.append_message("session-b", "user", "від B")

    hist_a = await store.get_history("session-a")
    hist_b = await store.get_history("session-b")

    assert len(hist_a) == 1
    assert hist_a[0]["content"] == "від A"
    assert len(hist_b) == 1
    assert hist_b[0]["content"] == "від B"


async def test_max_history_trim(store: SessionStore) -> None:
    """Messages exceeding max_history should be trimmed (oldest dropped)."""
    for i in range(15):  # max_history=10
        await store.append_message("trim-session", "user", f"msg {i}")

    history = await store.get_history("trim-session")
    assert len(history) == 10
    # Oldest messages (0–4) should be gone; only 5–14 remain
    assert history[0]["content"] == "msg 5"
    assert history[-1]["content"] == "msg 14"


async def test_clear_removes_all(store: SessionStore) -> None:
    await store.append_message("s-clear", "user", "текст")
    await store.clear("s-clear")

    history = await store.get_history("s-clear")
    assert history == []


async def test_unicode_content_preserved(store: SessionStore) -> None:
    """Ukrainian text, emoji, and special chars must survive JSON round-trip."""
    text = "Права дитини 🇺🇦 — стаття 1 «Конвенції ООН»"
    await store.append_message("s-unicode", "user", text)

    history = await store.get_history("s-unicode")
    assert history[0]["content"] == text

