"""Document upload endpoint — ingests a file into the RAG knowledge base.

Supported formats
-----------------
- ``.pdf``  — text extracted by pypdf
- ``.docx`` — text extracted by python-docx
- ``.txt``  — plain text, read as UTF-8

The endpoint is synchronous from the client's perspective: it completes the full
ingestion pipeline (load → chunk → embed → Chroma upsert → Postgres persist) and
then returns a summary.  For very large documents this may take several seconds.

Authentication
--------------
Requires the ``X-API-Key`` header when ``API_KEY`` is configured (same as /v1/chat).
"""

import io
import logging

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import ApiKeyDep, DbDep, IngestionPipelineDep
from knowledge.rag.ingestion.loaders import DocxLoader, PdfLoader

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

# Maximum accepted file size: 20 MB
_MAX_FILE_BYTES = 20 * 1024 * 1024

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/octet-stream",  # generic fallback — determined by extension
}

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UploadResponse(BaseModel):
    """Result of a successful document ingestion."""

    document_title: str
    chunk_count: int
    postgres_document_id: int | None = None
    message: str = "Document ingested successfully."


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to the RAG knowledge base",
    description=(
        "Upload a PDF, DOCX, or plain-text file for ingestion into the vector knowledge base.\\n\\n"
        "The pipeline runs synchronously: the document is loaded, chunked, embedded, stored in "
        "ChromaDB and recorded in Postgres.  Returns a summary when complete.\\n\\n"
        "**Limits**: max file size 20 MB, max filename length 255 characters.\\n\\n"
        "Requires `X-API-Key` header when `API_KEY` is configured."
    ),
)
async def upload_document(
    file: UploadFile,
    pipeline: IngestionPipelineDep,
    db: DbDep,
    _auth: ApiKeyDep,
    title: str | None = None,
) -> UploadResponse:
    """Ingest an uploaded file into the RAG knowledge base.

    Parameters
    ----------
    file:
        The uploaded file (multipart/form-data).
    title:
        Optional title override.  If omitted, the title is derived from the
        document metadata or filename stem.
    """
    # ── Validate filename & extension ─────────────────────────────────────────
    filename = file.filename or "upload"
    if len(filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_filename", "message": "Filename exceeds 255 characters."},
        )

    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "unsupported_file_type",
                "message": (
                    f"Unsupported file type '{suffix}'. "
                    "Allowed: .pdf, .docx, .txt"
                ),
            },
        )

    # ── Read file bytes ───────────────────────────────────────────────────────
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_file", "message": "Uploaded file is empty."},
        )
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "file_too_large",
                "message": f"File exceeds maximum allowed size of 20 MB.",
            },
        )

    # ── Parse the document in memory ─────────────────────────────────────────
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    try:
        if suffix == ".pdf":
            loaded = PdfLoader().load_from_bytes(data, filename=stem)
        elif suffix == ".docx":
            loaded = DocxLoader().load_from_bytes(data, filename=stem)
        else:  # .txt
            text = data.decode("utf-8", errors="replace")
            from knowledge.rag.ingestion.loaders import LoadedDocument
            from core.db.enums import DocumentSourceType
            loaded = LoadedDocument(
                text=text,
                title=stem,
                source_type=DocumentSourceType.txt,
                source_url=None,
            )
    except Exception as exc:
        logger.exception("Failed to parse uploaded file %r", filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "parse_error",
                "message": f"Could not parse the uploaded file: {exc}",
            },
        )

    if not loaded.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "empty_document",
                "message": "The document contains no extractable text.",
            },
        )

    # ── Run the ingestion pipeline ────────────────────────────────────────────
    logger.info(
        "upload_document: ingesting %r (%d bytes, type=%s)",
        filename, len(data), loaded.source_type.value,
    )

    try:
        result = await pipeline.run(
            source=filename,              # used as fallback source label
            loader=_MemoryLoader(loaded), # pre-loaded — skip disk IO in pipeline
            db=db,
            title_override=title or loaded.title,
        )
    except Exception as exc:
        logger.exception("Ingestion pipeline failed for file %r", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "ingestion_error",
                "message": f"Ingestion failed: {exc}",
            },
        )

    logger.info(
        "upload_document: ingested %r → %d chunks, doc_id=%s",
        result.document_title, result.chunk_count, result.postgres_document_id,
    )
    return UploadResponse(
        document_title=result.document_title,
        chunk_count=result.chunk_count,
        postgres_document_id=result.postgres_document_id,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

class _MemoryLoader:
    """Thin adapter that makes an already-loaded ``LoadedDocument`` look like a
    ``DocumentLoader``.  Passed to ``IngestionPipeline.run(loader=...)`` so the
    pipeline skips the disk-read step and uses the in-memory bytes we already have.
    """

    def __init__(self, doc: "LoadedDocument") -> None:
        self._doc = doc

    def load(self, source: str) -> "LoadedDocument":  # noqa: ARG002
        return self._doc
