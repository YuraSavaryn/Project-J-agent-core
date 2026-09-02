"""Initial schema — all tables from DB_SCHEMA.md.

Revision ID: 64de3441303b
Revises:
Create Date: 2026-09-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "64de3441303b"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── Enum types ────────────────────────────────────────────────────────────
    channel_type = sa.Enum("telegram", "web", name="channel_type")
    message_role = sa.Enum("user", "assistant", "system", name="message_role")
    answer_source = sa.Enum("curated", "rag", "fallback", name="answer_source")
    llm_provider_type = sa.Enum("groq", "openrouter", "gemini", name="llm_provider_type")
    document_source_type = sa.Enum(
        "pdf", "docx", "website", "manual", name="document_source_type"
    )

    channel_type.create(op.get_bind(), checkfirst=True)
    message_role.create(op.get_bind(), checkfirst=True)
    answer_source.create(op.get_bind(), checkfirst=True)
    llm_provider_type.create(op.get_bind(), checkfirst=True)
    document_source_type.create(op.get_bind(), checkfirst=True)

    # ── sessions ──────────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("channel", channel_type, nullable=False),
        sa.Column("external_user_id", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_active_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("idx_sessions_external_user", "sessions", ["channel", "external_user_id"])
    op.create_index("idx_sessions_last_active", "sessions", ["last_active_at"])

    # ── messages ──────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column(
            "session_id",
            sa.Text(),
            sa.ForeignKey("sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("idx_messages_session_created", "messages", ["session_id", "created_at"])

    # ── documents ─────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("source_type", document_source_type, nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("chroma_collection", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_index("idx_documents_active", "documents", ["is_active"])

    # ── chunks ────────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chroma_chunk_id", sa.Text(), unique=True, nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content_preview", sa.Text(), nullable=False),
    )
    op.create_index("idx_chunks_document", "chunks", ["document_id"])

    # ── query_logs ────────────────────────────────────────────────────────────
    op.create_table(
        "query_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column(
            "session_id",
            sa.Text(),
            sa.ForeignKey("sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("source", answer_source, nullable=False),
        sa.Column("llm_provider", llm_provider_type, nullable=True),
        sa.Column("llm_model", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("idx_query_logs_session", "query_logs", ["session_id"])
    op.create_index("idx_query_logs_source", "query_logs", ["source"])
    op.create_index("idx_query_logs_created", "query_logs", ["created_at"])

    # ── query_log_chunks ──────────────────────────────────────────────────────
    op.create_table(
        "query_log_chunks",
        sa.Column(
            "query_log_id",
            sa.BigInteger(),
            sa.ForeignKey("query_logs.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.REAL(), nullable=True),
    )
    op.create_index("idx_query_log_chunks_chunk", "query_log_chunks", ["chunk_id"])

    # ── feedback ──────────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column(
            "query_log_id",
            sa.BigInteger(),
            sa.ForeignKey("query_logs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        sa.UniqueConstraint("query_log_id", name="idx_feedback_one_per_query"),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("query_log_chunks")
    op.drop_table("query_logs")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("sessions")

    # Drop enum types
    for name in [
        "document_source_type",
        "llm_provider_type",
        "answer_source",
        "message_role",
        "channel_type",
    ]:
        sa.Enum(name=name).drop(op.get_bind(), checkfirst=True)
