"""CLI entry point for the ingestion pipeline.

Usage
-----
    python -m knowledge.rag.ingestion.run --source <path_or_url> [--title "My Title"]

Examples
--------
    # Ingest a PDF file
    python -m knowledge.rag.ingestion.run --source docs/statute.pdf

    # Ingest a web page
    python -m knowledge.rag.ingestion.run --source https://plast.ua/about

    # Ingest a DOCX with a custom title
    python -m knowledge.rag.ingestion.run --source docs/handbook.docx --title "Пластовий довідник"

Environment
-----------
All configuration (ChromaDB, embedding provider, etc.) is read from
environment variables / ``.env`` via ``config.settings``.

Requires a running embedding-worker and ChromaDB server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


def _build_pipeline():
    """Construct a ready-to-use IngestionPipeline from settings."""
    from config.settings import settings
    from knowledge.rag.embeddings import get_embedding_provider
    from knowledge.rag.ingestion.pipeline import IngestionPipeline
    from knowledge.rag.vectorstore.chroma_store import ChromaStore

    embedding_provider = get_embedding_provider(
        provider=settings.embedding_provider,
        worker_url=settings.embedding_worker_url,
        gemini_key=settings.gemini_api_key,
        gemini_model=settings.gemini_embedding_model,
    )

    vector_store = ChromaStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
        distance_fn=settings.chroma_distance_fn,
    )
    vector_store.ensure_collection()

    return IngestionPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


async def _run(source: str, title: str | None) -> None:
    pipeline = _build_pipeline()
    result = await pipeline.run(source, title_override=title)

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info("  Title       : %s", result.document_title)
    logger.info("  Source      : %s", result.source_url or source)
    logger.info("  Type        : %s", result.source_type.value)
    logger.info("  Chunks      : %d", result.chunk_count)
    if result.postgres_document_id:
        logger.info("  Postgres ID : %d", result.postgres_document_id)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m knowledge.rag.ingestion.run",
        description="Ingest a document (PDF, DOCX, or web URL) into the RAG knowledge base.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a PDF/DOCX file or an HTTP(S) URL to ingest.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the document title (optional).",
    )

    args = parser.parse_args()

    try:
        asyncio.run(_run(source=args.source, title=args.title))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

