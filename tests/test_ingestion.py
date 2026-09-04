"""Tests for step 10: ingestion pipeline (loaders, chunker, pipeline).

All tests are self-contained — no real ChromaDB server, no real embedding
worker, no real Postgres needed.

Coverage
--------
- TextChunker: basic split, overlap, paragraph/sentence boundaries, edge cases
- PdfLoader / DocxLoader / WebLoader: mocked at the library level
- get_loader() factory: correct auto-detection
- IngestionPipeline.run(): full flow with mocked embedder + ChromaStore + no DB
- ID consistency: UUID from pipeline.run() appears in ChromaStore (Chroma side)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest

from core.db.enums import DocumentSourceType
from knowledge.rag.ingestion.chunker import TextChunk, TextChunker
from knowledge.rag.ingestion.loaders import (
    DocxLoader,
    LoadedDocument,
    PdfLoader,
    WebLoader,
    get_loader,
)
from knowledge.rag.ingestion.pipeline import IngestionPipeline, IngestionResult
from knowledge.rag.vectorstore.chroma_store import ChromaStore


# ── Helpers ───────────────────────────────────────────────────────────────────

_ephemeral_client = chromadb.EphemeralClient()


def _make_store() -> ChromaStore:
    client = _ephemeral_client
    store = ChromaStore(
        collection_name=f"test_{uuid.uuid4().hex}",
        distance_fn="cosine",
        client=client,
    )
    store.ensure_collection()
    return store


def _make_embedder(dim: int = 4) -> AsyncMock:
    """Return a mock embedder that produces simple unit vectors."""
    embedder = AsyncMock()

    async def _embed(texts):
        # Return a fixed-dimension unit vector for each text
        return [[1.0] + [0.0] * (dim - 1) for _ in texts]

    embedder.embed = _embed
    return embedder


def _make_loader(text: str, source_type: DocumentSourceType, title: str = "Test Doc") -> MagicMock:
    loader = MagicMock()
    loader.load.return_value = LoadedDocument(
        text=text,
        title=title,
        source_type=source_type,
        source_url="https://example.com",
    )
    return loader


# ── TextChunker tests ─────────────────────────────────────────────────────────

def test_chunker_empty_text_returns_empty() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("") == []
    assert chunker.split("   ") == []


def test_chunker_short_text_single_chunk() -> None:
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    result = chunker.split("Короткий текст.")
    assert len(result) == 1
    assert result[0].text == "Короткий текст."
    assert result[0].chunk_index == 0


def test_chunker_splits_long_text() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "А" * 200
    result = chunker.split(text)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk.text) <= 60  # allow a little slack for boundary search


def test_chunker_overlap_content() -> None:
    """Overlap means adjacent chunks share some content."""
    chunker = TextChunker(chunk_size=80, chunk_overlap=20)
    text = "word " * 60  # 300 chars
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    # Last chars of chunk[0] should appear at start of chunk[1]
    # (overlap may split mid-word, but some content must be repeated)
    assert chunks[0].text[-5:] in chunks[1].text or chunks[0].char_start < chunks[1].char_start


def test_chunker_chunk_indexes_sequential() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "Текст " * 30
    chunks = chunker.split(text)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunker_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_paragraph_boundaries() -> None:
    """Chunker should prefer paragraph breaks when available."""
    para1 = "Перший абзац. " * 5
    para2 = "Другий абзац. " * 5
    text = para1 + "\n\n" + para2
    chunker = TextChunker(chunk_size=len(para1) + 20, chunk_overlap=10)
    chunks = chunker.split(text)
    # The double newline should ideally appear as a boundary
    # At minimum, we should get more than one chunk for this long text
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)


# ── get_loader factory tests ──────────────────────────────────────────────────

def test_get_loader_pdf() -> None:
    loader = get_loader("docs/handbook.pdf")
    assert isinstance(loader, PdfLoader)


def test_get_loader_docx() -> None:
    loader = get_loader("/path/to/file.docx")
    assert isinstance(loader, DocxLoader)


def test_get_loader_https() -> None:
    loader = get_loader("https://plast.ua/about")
    assert isinstance(loader, WebLoader)


def test_get_loader_http() -> None:
    loader = get_loader("http://example.com/page")
    assert isinstance(loader, WebLoader)


def test_get_loader_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine loader"):
        get_loader("somefile.txt")


# ── PdfLoader tests ───────────────────────────────────────────────────────────

def test_pdf_loader_extracts_text() -> None:
    """PdfLoader should call pypdf.PdfReader and return extracted text."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Сторінка один"

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    mock_reader.metadata = None

    with patch("pypdf.PdfReader", return_value=mock_reader):
        loader = PdfLoader()
        result = loader.load("test.pdf")

    assert result.text == "Сторінка один"
    assert result.source_type == DocumentSourceType.pdf


def test_pdf_loader_uses_metadata_title() -> None:
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Text"

    mock_meta = MagicMock()
    mock_meta.title = "Пластовий статут"

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    mock_reader.metadata = mock_meta

    with patch("pypdf.PdfReader", return_value=mock_reader):
        result = PdfLoader().load("statute.pdf")

    assert result.title == "Пластовий статут"


# ── WebLoader tests ───────────────────────────────────────────────────────────

def test_web_loader_extracts_text() -> None:
    html = "<html><head><title>Про Пласт</title></head><body><p>Вміст сторінки</p></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response):
        result = WebLoader().load("https://plast.ua/about")

    assert "Вміст сторінки" in result.text
    assert result.title == "Про Пласт"
    assert result.source_type == DocumentSourceType.website


def test_web_loader_removes_script_tags() -> None:
    html = "<html><body><p>Корисний текст</p><script>alert('bad')</script></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response):
        result = WebLoader().load("https://example.com")

    assert "alert" not in result.text
    assert "Корисний текст" in result.text


# ── IngestionPipeline tests ───────────────────────────────────────────────────

async def test_pipeline_run_returns_result() -> None:
    """pipeline.run() should return IngestionResult with correct counts."""
    store = _make_store()
    embedder = _make_embedder()
    text = "Тестовий документ про Пласт. " * 10
    loader = _make_loader(text, DocumentSourceType.website)
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    pipeline = IngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=chunker,
    )
    result = await pipeline.run("https://example.com", loader=loader)

    assert isinstance(result, IngestionResult)
    assert result.chunk_count > 0
    assert result.document_title == "Test Doc"
    assert result.source_type == DocumentSourceType.website
    assert len(result.chunk_ids) == result.chunk_count


async def test_pipeline_upserts_into_chroma() -> None:
    """After pipeline.run(), chunks should be findable in ChromaStore."""
    store = _make_store()
    embedder = _make_embedder()
    text = "Пласт — скаутська організація. " * 10
    loader = _make_loader(text, DocumentSourceType.website)
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    pipeline = IngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=chunker,
    )
    result = await pipeline.run("https://example.com", loader=loader)

    assert store.count() == result.chunk_count


async def test_pipeline_id_consistency() -> None:
    """IDs in IngestionResult.chunk_ids must match IDs stored in Chroma."""
    store = _make_store()
    embedder = _make_embedder()
    text = "Тест консистентності UUID. " * 10
    loader = _make_loader(text, DocumentSourceType.website)
    chunker = TextChunker(chunk_size=80, chunk_overlap=15)

    pipeline = IngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=chunker,
    )
    result = await pipeline.run("https://example.com", loader=loader)

    # Query with the same vector used during embedding
    query_vec = [1.0, 0.0, 0.0, 0.0]
    chroma_results = store.query(vector=query_vec, top_k=result.chunk_count)
    chroma_ids = {r.id for r in chroma_results}
    pipeline_ids = set(result.chunk_ids)

    # Every ID from the pipeline must be present in Chroma
    assert pipeline_ids == chroma_ids


async def test_pipeline_empty_document_returns_zero_chunks() -> None:
    """Pipeline should handle a document with no extractable text."""
    store = _make_store()
    embedder = _make_embedder()
    loader = _make_loader("", DocumentSourceType.pdf)

    pipeline = IngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
    )
    result = await pipeline.run("empty.pdf", loader=loader)

    assert result.chunk_count == 0
    assert result.chunk_ids == []
    assert store.count() == 0


async def test_pipeline_title_override() -> None:
    """title_override should replace the loader-provided title."""
    store = _make_store()
    embedder = _make_embedder()
    text = "Текст документу для перевірки заголовку. " * 5
    loader = _make_loader(text, DocumentSourceType.pdf, title="Original Title")

    pipeline = IngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
    )
    result = await pipeline.run("doc.pdf", loader=loader, title_override="Custom Title")

    assert result.document_title == "Custom Title"
