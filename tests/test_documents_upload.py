"""Tests for POST /v1/documents/upload endpoint."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_ingestion_pipeline, get_redis
from api.main import app
from knowledge.rag.ingestion.pipeline import IngestionResult
from core.db.enums import DocumentSourceType


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_pipeline_mock(chunk_count: int = 3, doc_id: int = 99) -> MagicMock:
    mock = MagicMock()
    mock.run = AsyncMock(
        return_value=IngestionResult(
            document_title="Test Document",
            source_url=None,
            source_type=DocumentSourceType.pdf,
            chunk_count=chunk_count,
            chunk_ids=["id1", "id2", "id3"],
            postgres_document_id=doc_id,
        )
    )
    return mock


@pytest.fixture
def client() -> TestClient:
    """TestClient with pipeline and redis mocked out."""
    mock_pipeline = _make_pipeline_mock()

    async def _fake_redis():
        yield AsyncMock()

    app.dependency_overrides[get_redis] = _fake_redis
    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Happy paths ───────────────────────────────────────────────────────────────

def test_upload_txt_returns_201(client: TestClient) -> None:
    """Plain text upload should succeed with 201."""
    content = b"Hello, this is a test document with some content."
    resp = client.post(
        "/v1/documents/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["chunk_count"] == 3
    assert data["postgres_document_id"] == 99
    assert data["document_title"] == "Test Document"
    assert "successfully" in data["message"].lower()


def test_upload_pdf_returns_201(client: TestClient) -> None:
    """A PDF upload should succeed (loader is mocked to avoid real PDF parsing)."""
    from knowledge.rag.ingestion.loaders import LoadedDocument
    from core.db.enums import DocumentSourceType

    fake_loaded = LoadedDocument(
        text="Extracted text from PDF",
        title="Test PDF",
        source_type=DocumentSourceType.pdf,
        source_url=None,
    )
    with patch(
        "api.routes.documents.PdfLoader.load_from_bytes",
        return_value=fake_loaded,
    ):
        content = b"%PDF-1.4 fake"
        resp = client.post(
            "/v1/documents/upload",
            files={"file": ("document.pdf", io.BytesIO(content), "application/pdf")},
        )
    assert resp.status_code == 201


def test_upload_title_override(client: TestClient) -> None:
    """title query param must be forwarded to the pipeline."""
    content = b"Some text content"
    resp = client.post(
        "/v1/documents/upload?title=Custom+Title",
        files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 201
    # The mock returns "Test Document" regardless — just verify the call reached pipeline
    app.dependency_overrides[get_ingestion_pipeline]().run.assert_called_once()


# ── Validation errors ─────────────────────────────────────────────────────────

def test_upload_unsupported_extension_returns_422(client: TestClient) -> None:
    """.xlsx should be rejected with 422."""
    content = b"not a valid format"
    resp = client.post(
        "/v1/documents/upload",
        files={"file": ("report.xlsx", io.BytesIO(content), "application/octet-stream")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_upload_empty_file_returns_422(client: TestClient) -> None:
    """Empty file should be rejected."""
    resp = client.post(
        "/v1/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "empty_file"


def test_upload_too_large_returns_413(client: TestClient) -> None:
    """File exceeding 20 MB should be rejected with 413."""
    big_content = b"x" * (21 * 1024 * 1024)  # 21 MB
    resp = client.post(
        "/v1/documents/upload",
        files={"file": ("huge.txt", io.BytesIO(big_content), "text/plain")},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "file_too_large"
