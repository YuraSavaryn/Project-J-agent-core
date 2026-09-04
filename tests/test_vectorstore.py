"""Tests for ChromaStore vectorstore wrapper (step 9).

Uses ``chromadb.EphemeralClient()`` (in-memory) — no real ChromaDB server needed.

Covers:
- Collection creation with cosine HNSW settings
- Single upsert and query
- Batch upsert
- Cosine similarity score conversion
- ID consistency: chroma_chunk_id == vector id in Chroma
- Metadata stored and retrieved correctly
- Ranked results ordering
- Delete
- Count
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from knowledge.rag.vectorstore.chroma_store import ChromaStore, QueryResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

_ephemeral_client = chromadb.EphemeralClient()


@pytest.fixture
def store() -> ChromaStore:
    """In-memory ChromaStore — unique collection per test, auto-cleaned up."""
    collection_name = f"test_{uuid.uuid4().hex}"
    s = ChromaStore(
        collection_name=collection_name,
        distance_fn="cosine",
        client=_ephemeral_client,
    )
    yield s
    # Cleanup: delete the collection so counts don't bleed across tests
    try:
        _ephemeral_client.delete_collection(collection_name)
    except Exception:
        pass


def _vec(dim: int = 4, val: float = 1.0) -> list[float]:
    """Return a simple unit vector of given dimension."""
    v = [0.0] * dim
    v[0] = val
    # Normalise to unit length
    length = sum(x ** 2 for x in v) ** 0.5
    return [x / length for x in v]


def _meta(doc_id: str = "doc-1", chunk_idx: int = 0) -> dict:
    return {
        "document_id": doc_id,
        "chunk_index": chunk_idx,
        "title": "Test Document",
        "source_url": "https://example.com",
    }


# ── Collection tests ──────────────────────────────────────────────────────────

def test_ensure_collection_creates_collection(store: ChromaStore) -> None:
    col = store.ensure_collection()
    assert col is not None
    # Name is dynamically generated per-fixture — verify it matches the store
    assert col.name == store._collection_name


def test_ensure_collection_is_idempotent(store: ChromaStore) -> None:
    """Calling ensure_collection twice should return the same collection."""
    col1 = store.ensure_collection()
    col2 = store.ensure_collection()
    assert col1.name == col2.name


def test_collection_has_cosine_metadata(store: ChromaStore) -> None:
    """Collection must be created with hnsw:space=cosine."""
    col = store.ensure_collection()
    assert col.metadata is not None
    assert col.metadata.get("hnsw:space") == "cosine"


# ── Upsert and query tests ────────────────────────────────────────────────────

def test_upsert_single_chunk(store: ChromaStore) -> None:
    chunk_id = str(uuid.uuid4())
    store.upsert(chunk_id=chunk_id, vector=_vec(), metadata=_meta())

    assert store.count() == 1


def test_query_returns_query_result(store: ChromaStore) -> None:
    chunk_id = str(uuid.uuid4())
    vec = _vec()
    store.upsert(chunk_id=chunk_id, vector=vec, metadata=_meta())

    results = store.query(vector=vec, top_k=1)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, QueryResult)
    assert result.id == chunk_id
    assert result.rank == 1


def test_query_score_is_similarity_not_distance(store: ChromaStore) -> None:
    """Score must be 1 - cosine_distance (higher = more similar)."""
    vec = _vec()
    chunk_id = str(uuid.uuid4())
    store.upsert(chunk_id=chunk_id, vector=vec, metadata=_meta())

    results = store.query(vector=vec, top_k=1)
    # Identical vectors → cosine distance ≈ 0 → score ≈ 1.0
    assert results[0].score > 0.99


def test_query_metadata_preserved(store: ChromaStore) -> None:
    """Metadata stored with the vector must be returned in query results."""
    meta = _meta(doc_id="doc-42", chunk_idx=7)
    chunk_id = str(uuid.uuid4())
    store.upsert(chunk_id=chunk_id, vector=_vec(), metadata=meta)

    results = store.query(vector=_vec(), top_k=1)
    assert results[0].metadata["document_id"] == "doc-42"
    assert results[0].metadata["chunk_index"] == 7
    assert results[0].metadata["title"] == "Test Document"
    assert results[0].metadata["source_url"] == "https://example.com"


def test_upsert_batch(store: ChromaStore) -> None:
    """Batch upsert should insert all chunks."""
    ids = [str(uuid.uuid4()) for _ in range(3)]
    vecs = [_vec(val=float(i + 1)) for i in range(3)]
    metas = [_meta(chunk_idx=i) for i in range(3)]

    store.upsert_batch(chunk_ids=ids, vectors=vecs, metadatas=metas)

    assert store.count() == 3


def test_query_ranked_results(store: ChromaStore) -> None:
    """Results must be ordered by descending similarity (rank 1 first)."""
    # Two chunks: one identical to query, one orthogonal
    query_vec = _vec(dim=4, val=1.0)  # [1, 0, 0, 0]
    close_vec = _vec(dim=4, val=1.0)  # same direction
    far_vec = [0.0, 1.0, 0.0, 0.0]   # orthogonal

    id_close = str(uuid.uuid4())
    id_far = str(uuid.uuid4())

    store.upsert_batch(
        chunk_ids=[id_close, id_far],
        vectors=[close_vec, far_vec],
        metadatas=[_meta(), _meta()],
    )

    results = store.query(vector=query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2
    # Most similar chunk should be ranked first
    assert results[0].id == id_close
    # Scores should be in descending order
    assert results[0].score >= results[1].score


def test_id_consistency_chroma_chunk_id(store: ChromaStore) -> None:
    """The UUID used as chroma vector id must be retrievable and match exactly.

    This validates the ID convention: same UUID in Chroma and in
    chunks.chroma_chunk_id in Postgres.
    """
    # Generate a UUID as we would in ingestion
    chunk_id = str(uuid.uuid4())
    store.upsert(chunk_id=chunk_id, vector=_vec(), metadata=_meta())

    results = store.query(vector=_vec(), top_k=1)

    # The ID returned from Chroma must match the one we inserted
    assert results[0].id == chunk_id


def test_delete_removes_chunk(store: ChromaStore) -> None:
    chunk_id = str(uuid.uuid4())
    store.upsert(chunk_id=chunk_id, vector=_vec(), metadata=_meta())
    assert store.count() == 1

    store.delete([chunk_id])
    assert store.count() == 0


def test_count_returns_number_of_vectors(store: ChromaStore) -> None:
    for i in range(5):
        store.upsert(chunk_id=str(uuid.uuid4()), vector=_vec(), metadata=_meta(chunk_idx=i))

    assert store.count() == 5


def test_upsert_with_document_text(store: ChromaStore) -> None:
    """upsert() should accept optional document text without errors."""
    chunk_id = str(uuid.uuid4())
    store.upsert(
        chunk_id=chunk_id,
        vector=_vec(),
        metadata=_meta(),
        document="Це текст чанку для пошуку.",
    )
    assert store.count() == 1

