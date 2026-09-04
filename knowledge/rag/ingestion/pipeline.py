"""Ingestion pipeline — orchestrates the full load → chunk → embed → store flow.

Pipeline steps
--------------
1. **Load**   — ``DocumentLoader.load(source)`` extracts raw text + metadata.
2. **Chunk**  — ``TextChunker.split(text)`` produces overlapping ``TextChunk`` objects.
3. **Embed**  — ``EmbeddingProvider.embed(texts)`` returns one vector per chunk.
4. **UUID**   — Generate a UUID per chunk (same ID used in Chroma AND Postgres).
5. **Chroma** — ``ChromaStore.upsert_batch()`` stores vectors + denormalised metadata.
6. **Postgres** (optional, when ``db`` is provided):
   a. Insert a ``Document`` row.
   b. Insert one ``Chunk`` row per chunk (with ``chroma_chunk_id`` = the UUID).

The pipeline is designed so that Chroma is always written first; if the Postgres
write fails, the vectors are already searchable and the ingestion can be retried
with the ``force=True`` flag (which upserts existing Chroma vectors).

Usage
-----
>>> pipeline = IngestionPipeline(embedding_provider=..., vector_store=...)
>>> result = await pipeline.run("https://plast.ua/about")
>>> print(result.document_title, result.chunk_count)
"""

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.db.enums import DocumentSourceType
from knowledge.rag.embeddings import EmbeddingProvider
from knowledge.rag.ingestion.chunker import TextChunker
from knowledge.rag.ingestion.loaders import DocumentLoader, LoadedDocument, get_loader
from knowledge.rag.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


# ── Result value object ───────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Summary of what the pipeline produced."""

    document_title: str
    source_url: str | None
    source_type: DocumentSourceType
    chunk_count: int
    chunk_ids: list[str] = field(default_factory=list)
    postgres_document_id: int | None = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """Coordinates the full ingestion flow for a single document.

    Parameters
    ----------
    embedding_provider:
        Used to embed chunk texts.
    vector_store:
        ChromaStore wrapper for vector upsert.
    chunker:
        Text splitter. Defaults to ``TextChunker()`` with settings defaults.
    collection_name:
        Chroma collection name. Defaults to ``settings.chroma_collection``.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaStore,
        chunker: TextChunker | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store
        self._chunker = chunker or TextChunker()
        self._collection_name = collection_name or settings.chroma_collection

    async def run(
        self,
        source: str,
        *,
        loader: DocumentLoader | None = None,
        db: AsyncSession | None = None,
        title_override: str | None = None,
    ) -> IngestionResult:
        """Run the full ingestion pipeline for *source*.

        Parameters
        ----------
        source:
            Path to a PDF/DOCX file or an HTTP(S) URL.
        loader:
            Override the auto-detected loader (useful for testing).
        db:
            Optional async SQLAlchemy session.  When provided, writes
            ``Document`` and ``Chunk`` rows to Postgres.
        title_override:
            Override the title extracted by the loader.

        Returns
        -------
        IngestionResult
        """
        # ── 1. Load ───────────────────────────────────────────────────────────
        effective_loader = loader or get_loader(source)
        loaded: LoadedDocument = effective_loader.load(source)
        title = title_override or loaded.title

        logger.info(
            "IngestionPipeline: loaded %r (%s, %d chars)",
            title, loaded.source_type.value, len(loaded.text),
        )

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        chunks = self._chunker.split(loaded.text)
        if not chunks:
            logger.warning("IngestionPipeline: no chunks produced for source=%r", source)
            return IngestionResult(
                document_title=title,
                source_url=loaded.source_url,
                source_type=loaded.source_type,
                chunk_count=0,
            )

        logger.info("IngestionPipeline: produced %d chunks", len(chunks))

        # ── 3. Embed ──────────────────────────────────────────────────────────
        texts = [c.text for c in chunks]
        vectors = await self._embedder.embed(texts)

        logger.info("IngestionPipeline: embedded %d chunks", len(vectors))

        # ── 4. Generate UUIDs — one per chunk ─────────────────────────────────
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        # ── 5. Upsert into Chroma ─────────────────────────────────────────────
        metadatas = [
            {
                "document_id": "",       # filled in after Postgres insert (or "" if no DB)
                "chunk_index": chunk.chunk_index,
                "title": title,
                "source_url": loaded.source_url or "",
            }
            for chunk in chunks
        ]

        self._store.upsert_batch(
            chunk_ids=chunk_ids,
            vectors=vectors,
            metadatas=metadatas,
            documents=texts,  # store text alongside vector for convenience
        )

        logger.info(
            "IngestionPipeline: upserted %d vectors into Chroma collection %r",
            len(chunk_ids), self._collection_name,
        )

        # ── 6. Persist to Postgres (optional) ────────────────────────────────
        postgres_doc_id: int | None = None

        if db is not None:
            postgres_doc_id = await self._persist_to_postgres(
                db=db,
                title=title,
                loaded=loaded,
                chunks_text=texts,
                chunk_ids=chunk_ids,
                chunk_objects=chunks,
            )
            # Back-fill document_id into Chroma metadata
            await self._backfill_document_id(
                chunk_ids=chunk_ids,
                metadatas=metadatas,
                document_id=postgres_doc_id,
            )

        return IngestionResult(
            document_title=title,
            source_url=loaded.source_url,
            source_type=loaded.source_type,
            chunk_count=len(chunks),
            chunk_ids=chunk_ids,
            postgres_document_id=postgres_doc_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _persist_to_postgres(
        self,
        db: AsyncSession,
        title: str,
        loaded: LoadedDocument,
        chunks_text: list[str],
        chunk_ids: list[str],
        chunk_objects,
    ) -> int:
        """Write Document + Chunk rows to Postgres.

        Returns the auto-generated ``Document.id``.
        """
        from core.db.models import Chunk as ORM_Chunk, Document as ORM_Document

        doc = ORM_Document(
            source_type=loaded.source_type,
            source_url=loaded.source_url,
            title=title,
            chroma_collection=self._collection_name,
            is_active=True,
        )
        db.add(doc)
        await db.flush()  # get doc.id without committing

        for chunk_obj, chunk_text, chunk_id in zip(chunk_objects, chunks_text, chunk_ids):
            preview = chunk_text[:200]  # store first 200 chars as preview
            orm_chunk = ORM_Chunk(
                document_id=doc.id,
                chroma_chunk_id=chunk_id,
                chunk_index=chunk_obj.chunk_index,
                content_preview=preview,
            )
            db.add(orm_chunk)

        await db.commit()
        logger.info(
            "IngestionPipeline: persisted document_id=%d with %d chunks to Postgres",
            doc.id, len(chunk_ids),
        )
        return doc.id

    async def _backfill_document_id(
        self,
        chunk_ids: list[str],
        metadatas: list[dict],
        document_id: int,
    ) -> None:
        """Update Chroma metadata with the now-known Postgres document_id."""
        updated_metadatas = [
            {**m, "document_id": str(document_id)} for m in metadatas
        ]
        # Chroma upsert with updated metadata only (re-use same IDs and vectors)
        # We only update metadata here — pass embeddings=None is not supported,
        # so we re-fetch and upsert. For simplicity, update via collection directly.
        col = self._store.collection
        col.update(ids=chunk_ids, metadatas=updated_metadatas)
        logger.debug(
            "IngestionPipeline: back-filled document_id=%d into %d Chroma records",
            document_id, len(chunk_ids),
        )

