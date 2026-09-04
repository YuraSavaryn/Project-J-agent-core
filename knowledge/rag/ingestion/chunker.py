"""Text chunker for the ingestion pipeline.

Splits a long text into overlapping chunks suitable for embedding.

Strategy
--------
- Split by sentence boundary (``.\n`` or ``\n\n`` preferred) when possible.
- Fall back to a simple sliding-window character split when no natural
  boundary is found within the current window.
- Each chunk is stripped and must be non-empty.

Parameters (all configurable)
-----------------------------
chunk_size    — target maximum characters per chunk (default: 512)
chunk_overlap — characters to repeat at the start of the next chunk (default: 64)

The defaults come from ``settings.rag_chunk_size`` / ``settings.rag_chunk_overlap``.
"""

import re
import logging
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single text chunk produced by the chunker.

    Attributes
    ----------
    text:
        The chunk content (stripped).
    chunk_index:
        0-based position of this chunk within the source document.
    char_start:
        Character offset of the first character within the original text.
    """

    text: str
    chunk_index: int
    char_start: int


class TextChunker:
    """Splits text into overlapping chunks.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters to repeat as context at the start of the next
        chunk.  Must be strictly less than ``chunk_size``.
    """

    # Paragraph / sentence boundary patterns (ordered by preference)
    _PARA_SPLIT = re.compile(r"\n{2,}")      # blank line → paragraph break
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")  # sentence boundary

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    def split(self, text: str) -> list[TextChunk]:
        """Split *text* into overlapping ``TextChunk`` objects.

        The algorithm:
        1. Prefer paragraph breaks (double newline) as boundaries.
        2. Within a paragraph, prefer sentence breaks.
        3. Hard-cut at ``chunk_size`` if no boundary is found.

        Parameters
        ----------
        text:
            Raw input text (e.g. from a loader).

        Returns
        -------
        list[TextChunk]
            Ordered list of chunks, index 0 = beginning of text.
        """
        text = text.strip()
        if not text:
            return []

        raw_chunks = self._sliding_window(text)
        result = []
        for idx, (chunk_text, char_start) in enumerate(raw_chunks):
            stripped = chunk_text.strip()
            if stripped:
                result.append(TextChunk(text=stripped, chunk_index=idx, char_start=char_start))

        logger.debug(
            "TextChunker: split %d chars → %d chunks (size=%d, overlap=%d)",
            len(text),
            len(result),
            self.chunk_size,
            self.chunk_overlap,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sliding_window(self, text: str) -> list[tuple[str, int]]:
        """Character-level sliding window with boundary-aware split."""
        chunks: list[tuple[str, int]] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)

            # If we're not at the end of the text, try to find a clean boundary
            if end < length:
                # Look for paragraph break in the last 20% of the window
                search_start = start + int(self.chunk_size * 0.8)
                para_match = None
                for m in self._PARA_SPLIT.finditer(text, search_start, end + 50):
                    para_match = m
                    break

                if para_match:
                    end = para_match.start()
                else:
                    # Try sentence boundary
                    sent_match = None
                    for m in self._SENT_SPLIT.finditer(text, search_start, end + 50):
                        sent_match = m
                        break

                    if sent_match:
                        end = sent_match.start() + 1  # include the final punctuation

            chunks.append((text[start:end], start))

            # Advance by (chunk_size - overlap)
            advance = self.chunk_size - self.chunk_overlap
            start += advance

        return chunks

