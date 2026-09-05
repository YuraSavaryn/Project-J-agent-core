"""End-to-end RAG test.

Flow:
1. Ingest a document via IngestionPipeline.
2. Send a query via /v1/chat.
3. Verify the response contains the relevant source chunk from the ingested document.
"""

import pytest
from httpx import AsyncClient

from core.db.enums import DocumentSourceType
from knowledge.rag.ingestion.pipeline import IngestionPipeline
from knowledge.rag.ingestion.loaders import LoadedDocument


import pytest
import uuid
import chromadb
from typing import AsyncGenerator
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_redis, get_retriever, get_orchestrator, get_llm_router
from core.llm.base import LLMResponse
from core.llm.router import LLMRouter
from core.db.enums import DocumentSourceType
from knowledge.rag.ingestion.pipeline import IngestionPipeline
from knowledge.rag.ingestion.loaders import LoadedDocument
from knowledge.rag.ingestion.chunker import TextChunker
from knowledge.rag.retriever import Retriever
from knowledge.rag.vectorstore.chroma_store import ChromaStore


_ephemeral_client = chromadb.EphemeralClient()

@pytest.fixture
def store() -> ChromaStore:
    collection_name = f"test_{uuid.uuid4().hex}"
    s = ChromaStore(
        collection_name=collection_name,
        distance_fn="cosine",
        client=_ephemeral_client,
    )
    s.ensure_collection()
    yield s
    try:
        _ephemeral_client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.fixture
def mock_embedder() -> AsyncMock:
    m = AsyncMock()
    # Dummy embedding: always returns a fixed vector
    async def _embed(texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    m.embed = _embed
    return m


@pytest.fixture
def retriever(mock_embedder: AsyncMock, store: ChromaStore) -> Retriever:
    return Retriever(embedding_provider=mock_embedder, vector_store=store)


@pytest.fixture
def mock_llm_router() -> LLMRouter:
    m = AsyncMock(spec=LLMRouter)
    # LLM always responds affirmatively
    m.complete.return_value = LLMResponse(
        content="Пласт був заснований у 1911 році.",
        provider="mock",
        model="mock-model",
    )
    return m


@pytest.fixture
def test_client(retriever: Retriever, mock_llm_router: LLMRouter) -> TestClient:
    from fakeredis.aioredis import FakeRedis
    
    # We override dependencies:
    # 1. get_redis (fakeredis)
    # 2. get_retriever (test retriever)
    # 3. get_llm_router (test llm)
    async def _fake_redis():
        redis = FakeRedis(decode_responses=True)
        yield redis
        await redis.aclose()
        
    app.dependency_overrides[get_redis] = _fake_redis
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm_router] = lambda: mock_llm_router
    
    with TestClient(app) as client:
        yield client
        
    app.dependency_overrides.clear()


@pytest.fixture
def ingestion_pipeline(mock_embedder: AsyncMock, store: ChromaStore) -> IngestionPipeline:
    return IngestionPipeline(
        embedding_provider=mock_embedder,
        vector_store=store,
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
    )


@pytest.mark.asyncio
async def test_e2e_rag_ingest_and_chat(
    test_client: TestClient,
    ingestion_pipeline: IngestionPipeline,
) -> None:
    # 1. Ingest a mock document
    class MockLoader:
        def load(self, source: str) -> LoadedDocument:
            return LoadedDocument(
                text="Пласт — це скаутська організація, яка була заснована у 1911 році.",
                title="Історія Пласту",
                source_type=DocumentSourceType.manual,
                source_url="https://plast.ua/history",
            )
            
    await ingestion_pipeline.run(
        source="mock_history.txt",
        loader=MockLoader(),
    )

    # 2. Send query to /v1/chat
    payload = {
        "session_id": "test-session-rag-e2e",
        "message": "В якому році засновано Пласт?",
        "channel": "web",
    }
    response = test_client.post("/v1/chat", json=payload)

    # 3. Verify
    assert response.status_code == 200
    data = response.json()
    
    assert "reply" in data
    assert data["session_id"] == "test-session-rag-e2e"
    
    sources = data["sources"]
    assert len(sources) > 0
    assert sources[0]["title"] == "Історія Пласту"
    assert sources[0]["url"] == "https://plast.ua/history"
    assert "скаутська організація" in sources[0]["excerpt"]
