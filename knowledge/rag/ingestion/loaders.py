"""Document loaders for the ingestion pipeline.

Each loader extracts raw text from a specific source format.
All loaders implement the ``DocumentLoader`` interface:
    ``load(source) -> LoadedDocument``

Supported formats
-----------------
- PDF  — via ``pypdf``
- DOCX — via ``python-docx``
- Web  — via ``httpx`` + ``beautifulsoup4`` (scrapes visible text)

The returned ``LoadedDocument`` carries the raw text plus minimal metadata
(title, source_url, source_type) needed by the ingestion pipeline.
"""

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from core.db.enums import DocumentSourceType

logger = logging.getLogger(__name__)


# ── Value object ──────────────────────────────────────────────────────────────

@dataclass
class LoadedDocument:
    """Raw extracted text plus source metadata.

    Produced by a loader and consumed by the chunker.
    """

    text: str
    title: str
    source_type: DocumentSourceType
    source_url: str | None = None
    extra_metadata: dict = field(default_factory=dict)


# ── Interface ─────────────────────────────────────────────────────────────────

class DocumentLoader(ABC):
    """Abstract loader interface."""

    @abstractmethod
    def load(self, source: str) -> LoadedDocument:
        """Load and extract text from *source*.

        Parameters
        ----------
        source:
            Filesystem path (PDF/DOCX) or HTTP URL (web).

        Returns
        -------
        LoadedDocument
        """
        ...


# ── PDF loader ────────────────────────────────────────────────────────────────

class PdfLoader(DocumentLoader):
    """Load text from a PDF file using ``pypdf``.

    Parameters
    ----------
    title:
        Human-readable title. Defaults to the filename stem.
    """

    def load(self, source: str) -> LoadedDocument:
        try:
            import pypdf  # deferred — not needed for web/docx paths
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF loading. Run: uv add pypdf") from exc

        path = Path(source)
        logger.info("PdfLoader: loading %s", path)

        reader = pypdf.PdfReader(str(path))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        text = "\n".join(pages).strip()
        title = reader.metadata.title if reader.metadata and reader.metadata.title else path.stem

        logger.info("PdfLoader: extracted %d chars from %d pages", len(text), len(reader.pages))
        return LoadedDocument(
            text=text,
            title=title,
            source_type=DocumentSourceType.pdf,
            source_url=path.resolve().as_uri(),
        )


# ── DOCX loader ───────────────────────────────────────────────────────────────

class DocxLoader(DocumentLoader):
    """Load text from a Word (.docx) file using ``python-docx``."""

    def load(self, source: str) -> LoadedDocument:
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise ImportError(
                "python-docx is required for DOCX loading. Run: uv add python-docx"
            ) from exc

        path = Path(source)
        logger.info("DocxLoader: loading %s", path)

        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()

        # Try document core properties for a title
        title = (
            doc.core_properties.title
            if doc.core_properties and doc.core_properties.title
            else path.stem
        )

        logger.info("DocxLoader: extracted %d chars from %d paragraphs", len(text), len(paragraphs))
        return LoadedDocument(
            text=text,
            title=title,
            source_type=DocumentSourceType.docx,
            source_url=path.resolve().as_uri(),
        )


# ── Web loader ────────────────────────────────────────────────────────────────

class WebLoader(DocumentLoader):
    """Fetch and extract visible text from a web page.

    Uses ``httpx`` for the HTTP request and ``beautifulsoup4`` for HTML parsing.
    Removes script/style tags and collapses whitespace.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    def load(self, source: str) -> LoadedDocument:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "httpx and beautifulsoup4 are required for web loading."
            ) from exc

        url = source
        logger.info("WebLoader: fetching %s", url)

        response = httpx.get(url, timeout=self._timeout, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Try to extract a sensible title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Collapse excessive blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)

        logger.info("WebLoader: extracted %d chars from %s", len(text), url)
        return LoadedDocument(
            text=text,
            title=title,
            source_type=DocumentSourceType.website,
            source_url=url,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_loader(source: str) -> DocumentLoader:
    """Return the appropriate loader for *source* based on extension or URL scheme."""
    lower = source.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return WebLoader()
    if lower.endswith(".pdf"):
        return PdfLoader()
    if lower.endswith(".docx"):
        return DocxLoader()
    raise ValueError(
        f"Cannot determine loader for source={source!r}. "
        "Expected .pdf, .docx, or an HTTP(S) URL."
    )
