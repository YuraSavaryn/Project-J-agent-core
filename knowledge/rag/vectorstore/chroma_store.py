"""ChromaDB vector store wrapper.

Conventions (step 9)
--------------------
- **One collection**: ``ngo_knowledge_base`` — all knowledge in a single
  collection, not one per document.
- **Cosine distance** (``hnsw:space = "cosine"``) — consistent with bge-m3/e5
  embedding models that produce unit-normalised vectors.
- **Vector dimension**: 1024 (bge-m3 output size).
- **ID strategy**: UUID generated at ingestion time, used as both:
    - the Chroma vector ``id``
    - ``chunks.chroma_chunk_id`` in PostgreSQL (referential consistency)
- **Metadata stored with each vector** (denormalised on purpose — avoids extra
  Postgres round-trips during retrieval):
    ``document_id``, ``chunk_index``, ``title``, ``source_url``

Usage
-----
Instantiate ``ChromaStore`` once (or via DI) and call:
    - ``ensure_collection()``  — idempotent, creates if missing
    - ``upsert()``             — add / update a single chunk
    - ``query()``              — nearest-neighbour search
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings as app_settings

logger = logging.getLogger(__name__)


# ── Query result type ─────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """A single chunk returned from a vector similarity search.

    Attributes
    ----------
    id:
        The Chroma / Postgres ``chroma_chunk_id`` (UUID string).
    score:
        Cosine distance converted to similarity score (1 − distance).
        Higher is more similar.
    rank:
        1-based position in the result list (1 = most similar).
    metadata:
        Denormalised fields: ``document_id``, ``chunk_index``, ``title``,
        ``source_url``.  May also contain ``content_preview`` if stored.
    """

    id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── ChromaStore ───────────────────────────────────────────────────────────────

class ChromaStore:
    """Thin async-friendly wrapper around the ChromaDB HTTP client.

    Although ``chromadb.AsyncHttpClient`` is available, the standard
    ``chromadb.HttpClient`` is used here to keep dependency surface minimal.
    The wrapper is designed for easy injection (tests can pass a
    ``chromadb.Client()`` in-memory client).

    Parameters
    ----------
    host:
        ChromaDB server hostname. Defaults to ``settings.chroma_host``.
    port:
        ChromaDB server port. Defaults to ``settings.chroma_port``.
    collection_name:
        Name of the Chroma collection to use. Defaults to
        ``settings.chroma_collection``.
    distance_fn:
        HNSW distance function. Defaults to ``settings.chroma_distance_fn``.
    client:
        Optional pre-configured Chroma client (used in tests to inject an
        in-memory ``chromadb.EphemeralClient()``).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        distance_fn: str | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._collection_name = collection_name or app_settings.chroma_collection
        self._distance_fn = distance_fn or app_settings.chroma_distance_fn

        if client is not None:
            self._client = client
        else:
            self._client = chromadb.HttpClient(
                host=host or app_settings.chroma_host,
                port=port or app_settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        self._collection: chromadb.Collection | None = None

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection(self) -> chromadb.Collection:
        """Get or create the collection with correct HNSW settings.

        Idempotent — safe to call on every startup.
        """
        if self._collection is not None:
            return self._collection

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_fn},
        )
        logger.info(
            "ChromaStore: collection %r ready (distance=%s, count=%d)",
            self._collection_name,
            self._distance_fn,
            self._collection.count(),
        )
        return self._collection

    @property
    def collection(self) -> chromadb.Collection:
        """Return the active collection, creating it if necessary."""
        return self.ensure_collection()

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        metadata: dict[str, Any],
        document: str | None = None,
    ) -> None:
        """Insert or update a single chunk vector.

        Parameters
        ----------
        chunk_id:
            UUID string — must match ``chunks.chroma_chunk_id`` in Postgres.
        vector:
            Embedding vector (length must match ``chroma_vector_dim``).
        metadata:
            Dict with ``document_id``, ``chunk_index``, ``title``,
            ``source_url`` (and optionally ``content_preview``).
        document:
            Optional raw text stored alongside the vector (Chroma supports
            storing the document text for convenience retrieval).
        """
        col = self.collection
        kwargs: dict[str, Any] = {
            "ids": [chunk_id],
            "embeddings": [vector],
            "metadatas": [metadata],
        }
        if document is not None:
            kwargs["documents"] = [document]

        col.upsert(**kwargs)
        logger.debug("ChromaStore: upserted chunk id=%s", chunk_id)

    def upsert_batch(
        self,
        chunk_ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str] | None = None,
    ) -> None:
        """Batch upsert multiple chunks in a single round-trip."""
        col = self.collection
        kwargs: dict[str, Any] = {
            "ids": chunk_ids,
            "embeddings": vectors,
            "metadatas": metadatas,
        }
        if documents is not None:
            kwargs["documents"] = documents

        col.upsert(**kwargs)
        logger.debug("ChromaStore: batch upserted %d chunks", len(chunk_ids))

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k: int | None = None,
    ) -> list[QueryResult]:
        """Nearest-neighbour search.

        Parameters
        ----------
        vector:
            Query embedding vector.
        top_k:
            Number of results to return. Defaults to ``settings.rag_top_k``.

        Returns
        -------
        list[QueryResult]
            Results ordered by descending similarity (rank 1 = most similar).
        """
        k = top_k or app_settings.rag_top_k
        col = self.collection

        raw = col.query(
            query_embeddings=[vector],
            n_results=k,
            include=["metadatas", "distances"],
        )

        ids: list[str] = raw["ids"][0]
        distances: list[float] = raw["distances"][0]
        metadatas: list[dict] = raw["metadatas"][0]  # type: ignore[assignment]

        results: list[QueryResult] = []
        for rank, (chunk_id, dist, meta) in enumerate(
            zip(ids, distances, metadatas), start=1
        ):
            # Chroma cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            # For unit-normalised vectors: similarity = 1 − distance
            score = float(1.0 - dist)
            results.append(
                QueryResult(id=chunk_id, score=score, rank=rank, metadata=meta)
            )

        logger.debug("ChromaStore: query returned %d results (top_k=%d)", len(results), k)
        return results

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        return self.collection.count()

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by their IDs."""
        self.collection.delete(ids=chunk_ids)
        logger.debug("ChromaStore: deleted %d chunks", len(chunk_ids))

