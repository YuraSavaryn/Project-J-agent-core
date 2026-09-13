"""ORM models for agent-core — all tables from DB_SCHEMA.md.

Tables
------
- Session       — conversation session metadata
- Message       — full message history (source of truth)
- Document      — RAG knowledge base document metadata
- Chunk         — link between ChromaDB vector and its source document
- QueryLog      — per-request analytics log
- QueryLogChunk — many-to-many join between QueryLog and Chunk
- Feedback      — user rating on a query result
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    REAL,
    SmallInteger,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from core.db.base import Base
from core.db.enums import (
    AnswerSource,
    ChannelType,
    DocumentSourceType,
    LLMProviderType,
    MessageRole,
    answer_source_pg,
    channel_type_pg,
    document_source_type_pg,
    llm_provider_type_pg,
    message_role_pg,
)


# ── Session ───────────────────────────────────────────────────────────────────

class Session(Base):
    """One continuous conversation, regardless of the channel."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_external_user", "channel", "external_user_id"),
        Index("idx_sessions_last_active", "last_active_at"),
    )

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    channel: Mapped[ChannelType] = mapped_column(channel_type_pg, nullable=False)
    external_user_id: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    query_logs: Mapped[list[QueryLog]] = relationship(
        "QueryLog", back_populates="session", cascade="all, delete-orphan"
    )


# ── Message ───────────────────────────────────────────────────────────────────

class Message(Base):
    """Full conversation history — source of truth (Redis holds only a hot tail)."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(message_role_pg, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    session: Mapped[Session] = relationship("Session", back_populates="messages")


# ── Document ──────────────────────────────────────────────────────────────────

class Document(Base):
    """Metadata for RAG knowledge base documents. Vectors live in ChromaDB."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_documents_active", "is_active"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    source_type: Mapped[DocumentSourceType] = mapped_column(
        document_source_type_pg, nullable=False
    )
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    chroma_collection: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true", nullable=False, default=True)

    # Relationships
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


# ── Chunk ─────────────────────────────────────────────────────────────────────

class Chunk(Base):
    """Links a ChromaDB vector ID to its source document."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chroma_chunk_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content_preview: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    query_log_chunks: Mapped[list[QueryLogChunk]] = relationship(
        "QueryLogChunk", back_populates="chunk", cascade="all, delete-orphan"
    )


# ── QueryLog ──────────────────────────────────────────────────────────────────

class QueryLog(Base):
    """Per-request analytics: which source answered, which LLM, how long."""

    __tablename__ = "query_logs"
    __table_args__ = (
        Index("idx_query_logs_session", "session_id"),
        Index("idx_query_logs_source", "source"),
        Index("idx_query_logs_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[AnswerSource] = mapped_column(answer_source_pg, nullable=False)
    llm_provider: Mapped[LLMProviderType | None] = mapped_column(
        llm_provider_type_pg, nullable=True
    )
    llm_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    session: Mapped[Session] = relationship("Session", back_populates="query_logs")
    query_log_chunks: Mapped[list[QueryLogChunk]] = relationship(
        "QueryLogChunk", back_populates="query_log", cascade="all, delete-orphan"
    )
    feedback: Mapped[Feedback | None] = relationship(
        "Feedback", back_populates="query_log", uselist=False, cascade="all, delete-orphan"
    )


# ── QueryLogChunk ─────────────────────────────────────────────────────────────

class QueryLogChunk(Base):
    """Join table: which chunks were retrieved for a given query log entry."""

    __tablename__ = "query_log_chunks"
    __table_args__ = (
        Index("idx_query_log_chunks_chunk", "chunk_id"),
    )

    query_log_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("query_logs.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(REAL, nullable=True)

    # Relationships
    query_log: Mapped[QueryLog] = relationship(
        "QueryLog", back_populates="query_log_chunks"
    )
    chunk: Mapped[Chunk] = relationship("Chunk", back_populates="query_log_chunks")


# ── Feedback ──────────────────────────────────────────────────────────────────

class Feedback(Base):
    """User rating (👍 = 1, 👎 = -1) on a single query result."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        UniqueConstraint("query_log_id", name="idx_feedback_one_per_query"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    query_log_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    query_log: Mapped[QueryLog] = relationship("QueryLog", back_populates="feedback")
