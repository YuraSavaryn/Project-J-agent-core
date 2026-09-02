"""PostgreSQL enum types for agent-core.

All enums are defined as Python ``enum.Enum`` subclasses AND registered as
native PostgreSQL ``ENUM`` types via SQLAlchemy ``PgEnum``.

Naming convention: Python class name → snake_case PostgreSQL type name.
"""

from __future__ import annotations

import enum

from sqlalchemy import Enum as PgEnum


class ChannelType(str, enum.Enum):
    """Source channel of a conversation session."""
    telegram = "telegram"
    web = "web"


class MessageRole(str, enum.Enum):
    """Role of the message author in a conversation."""
    user = "user"
    assistant = "assistant"
    system = "system"


class AnswerSource(str, enum.Enum):
    """Which knowledge source produced the answer."""
    curated = "curated"   # curated wiki hit
    rag = "rag"           # RAG retrieval
    fallback = "fallback" # LLM without retrieved context


class LLMProviderType(str, enum.Enum):
    """LLM backend used to generate the answer."""
    groq = "groq"
    openrouter = "openrouter"
    gemini = "gemini"


class DocumentSourceType(str, enum.Enum):
    """How a document entered the RAG knowledge base."""
    pdf = "pdf"
    docx = "docx"
    website = "website"
    manual = "manual"


# ── SQLAlchemy type objects (reuse these in model column definitions) ──────────

channel_type_pg = PgEnum(ChannelType, name="channel_type", create_type=True)
message_role_pg = PgEnum(MessageRole, name="message_role", create_type=True)
answer_source_pg = PgEnum(AnswerSource, name="answer_source", create_type=True)
llm_provider_type_pg = PgEnum(LLMProviderType, name="llm_provider_type", create_type=True)
document_source_type_pg = PgEnum(DocumentSourceType, name="document_source_type", create_type=True)

