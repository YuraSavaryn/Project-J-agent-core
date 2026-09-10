"""CLI entry point for the ingestion pipeline.

Usage
-----
    python -m knowledge.rag.ingestion.run --source <path_or_url> [--title "My Title"]
    python -m knowledge.rag.ingestion.run --source <directory> [--recursive]

Examples
--------
    # Ingest a single PDF file
    python -m knowledge.rag.ingestion.run --source docs/statute.pdf

    # Ingest a web page
    python -m knowledge.rag.ingestion.run --source https://plast.ua/about

    # Ingest a DOCX with a custom title
    python -m knowledge.rag.ingestion.run --source docs/handbook.docx --title "Пластовий довідник"

    # Ingest every supported file directly inside a directory (no subfolders)
    python -m knowledge.rag.ingestion.run --source data/raw/

    # Ingest every supported file in a directory, including subfolders
    python -m knowledge.rag.ingestion.run --source data/raw/ --recursive

Notes
-----
When ``--source`` is a directory, ``--title`` is ignored (each file gets its
own title derived from its filename) and a warning is logged. Individual
file failures do not stop the batch — they are logged and the run exits
with a non-zero status if at least one file failed.

Supported file extensions for directory ingestion: .pdf, .docx

Environment
-----------
All configuration (ChromaDB, embedding provider, etc.) is read from
environment variables / ``.env`` via ``config.settings``.

Requires a running embedding-worker and ChromaDB server.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

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


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _iter_source_files(directory: Path, recursive: bool) -> list[Path]:
    """List files under ``directory`` with a supported extension.

    Non-recursive by default (only files directly inside the directory,
    matching the shell-glob mental model most people expect). Skips
    unsupported extensions with a warning rather than failing the batch.
    """
    pattern = "**/*" if recursive else "*"
    candidates = sorted(p for p in directory.glob(pattern) if p.is_file())

    files, skipped = [], []
    for path in candidates:
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        else:
            skipped.append(path)

    if skipped:
        logger.warning(
            "Skipping %d file(s) with unsupported extension: %s",
            len(skipped),
            ", ".join(str(p) for p in skipped),
        )
    return files


async def _ingest_one(pipeline, source: str, title: str | None) -> bool:
    """Run ingestion for a single source. Returns True on success."""
    try:
        result = await pipeline.run(source, title_override=title)
    except Exception as exc:
        logger.error("  ✗ Failed: %s — %s", source, exc)
        return False

    logger.info("  ✓ %s", result.document_title)
    logger.info("      Source : %s", result.source_url or source)
    logger.info("      Type   : %s", result.source_type.value)
    logger.info("      Chunks : %d", result.chunk_count)
    if result.postgres_document_id:
        logger.info("      PG ID  : %d", result.postgres_document_id)
    return True


async def _run(source: str, title: str | None, recursive: bool) -> None:
    pipeline = _build_pipeline()

    # URL or a single file — original single-source behaviour, unchanged.
    path = None if _is_url(source) else Path(source)
    if path is None or path.is_file():
        logger.info("Ingesting single source: %s", source)
        ok = await _ingest_one(pipeline, source, title)
        if not ok:
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("Ingestion complete!")
        logger.info("=" * 60)
        return

    if not path.is_dir():
        logger.error("Source not found: %s", source)
        sys.exit(1)

    if title is not None:
        logger.warning("--title is ignored when --source is a directory")

    files = _iter_source_files(path, recursive)
    if not files:
        logger.warning("No supported files (%s) found under %s", ", ".join(sorted(SUPPORTED_EXTENSIONS)), path)
        return

    logger.info("Found %d file(s) to ingest under %s%s", len(files), path, " (recursive)" if recursive else "")

    succeeded = 0
    for i, file_path in enumerate(files, start=1):
        logger.info("[%d/%d] %s", i, len(files), file_path)
        if await _ingest_one(pipeline, str(file_path), title=None):
            succeeded += 1

    failed = len(files) - succeeded
    logger.info("=" * 60)
    logger.info("Batch ingestion complete: %d succeeded, %d failed", succeeded, failed)
    logger.info("=" * 60)
    if failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m knowledge.rag.ingestion.run",
        description="Ingest a document (PDF, DOCX, or web URL) into the RAG knowledge base.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a PDF/DOCX file, a directory of files, or an HTTP(S) URL to ingest.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Override the document title (optional; ignored when --source is a directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --source is a directory, also walk into subdirectories.",
    )

    args = parser.parse_args()

    try:
        asyncio.run(_run(source=args.source, title=args.title, recursive=args.recursive))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()