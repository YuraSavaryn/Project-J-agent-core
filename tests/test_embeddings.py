import httpx
import pytest
import respx

from knowledge.rag.embeddings import (
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
)


@pytest.mark.asyncio
async def test_local_embedding_provider():
    provider = LocalEmbeddingProvider(base_url="http://localhost:8002")
    texts = ["hello", "world"]
    
    expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with respx.mock(base_url="http://localhost:8002") as respx_mock:
        route = respx_mock.post("/embed").mock(
            return_value=httpx.Response(200, json=expected_embeddings)
        )
        
        result = await provider.embed(texts)
        
        assert route.called
        assert result == expected_embeddings
        
        # Check that the request payload was correct
        request = route.calls.last.request
        import json
        payload = json.loads(request.content)
        assert payload["inputs"] == texts


@pytest.mark.asyncio
async def test_gemini_embedding_provider():
    provider = GeminiEmbeddingProvider(api_key="fake-key", model_name="text-embedding-004")
    texts = ["hello", "world"]
    
    mock_response_data = {
        "embeddings": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.4, 0.5, 0.6]}
        ]
    }
    
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as respx_mock:
        route = respx_mock.post("/v1beta/models/text-embedding-004:batchEmbedContents?key=fake-key").mock(
            return_value=httpx.Response(200, json=mock_response_data)
        )
        
        result = await provider.embed(texts)
        
        assert route.called
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        request = route.calls.last.request
        assert "key=fake-key" in str(request.url)
        import json
        payload = json.loads(request.content)
        assert "requests" in payload
        assert len(payload["requests"]) == 2
        assert payload["requests"][0]["content"]["parts"][0]["text"] == "hello"


def test_get_embedding_provider():
    local_prov = get_embedding_provider(
        provider="local",
        worker_url="http://test:8002",
        gemini_key="",
        gemini_model="text-embedding-004",
    )
    assert isinstance(local_prov, LocalEmbeddingProvider)
    assert local_prov.base_url == "http://test:8002"

    gemini_prov = get_embedding_provider(
        provider="gemini",
        worker_url="http://test:8002",
        gemini_key="secret",
        gemini_model="my-model",
    )
    assert isinstance(gemini_prov, GeminiEmbeddingProvider)
    assert gemini_prov.api_key == "secret"
    assert gemini_prov.model_name == "my-model"

    with pytest.raises(ValueError, match="gemini_api_key is required"):
        get_embedding_provider(
            provider="gemini",
            worker_url="http://test:8002",
            gemini_key="",
            gemini_model="my-model",
        )

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider(
            provider="unknown",
            worker_url="",
            gemini_key="",
            gemini_model="",
        )

