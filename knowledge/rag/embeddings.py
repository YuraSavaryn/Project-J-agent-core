"""Embedding providers for RAG."""

import logging
from abc import ABC, abstractmethod
from typing import List

import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for generating text embeddings."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Parameters
        ----------
        texts:
            List of strings to embed.

        Returns
        -------
        List[List[float]]:
            List of embedding vectors, one for each input text.
        """
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using a local text-embeddings-inference worker."""

    def __init__(self, base_url: str, timeout: float | None = None):
        from config.settings import settings as _settings
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else _settings.embedding_request_timeout_seconds
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        logger.debug("LocalEmbeddingProvider: generating embeddings for %d texts", len(texts))
        
        # TEI has a max_client_batch_size (default 32). We must batch the request locally.
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self._client.post("/embed", json={"inputs": batch, "truncate": True})
            response.raise_for_status()
            all_embeddings.extend(response.json())
            
        return all_embeddings


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using the Gemini REST API."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        self._client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com",
            timeout=30.0,
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        logger.debug("GeminiEmbeddingProvider: generating embeddings for %d texts", len(texts))
        url = f"/v1beta/models/{self.model_name}:batchEmbedContents"
        
        requests = [
            {
                "model": f"models/{self.model_name}",
                "content": {"parts": [{"text": t}]}
            }
            for t in texts
        ]
        
        response = await self._client.post(
            url,
            params={"key": self.api_key},
            json={"requests": requests},
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = [item["values"] for item in data["embeddings"]]
        return embeddings


def get_embedding_provider(
    provider: str,
    worker_url: str,
    gemini_key: str,
    gemini_model: str,
) -> EmbeddingProvider:
    """Factory to get the configured embedding provider."""
    if provider == "local":
        return LocalEmbeddingProvider(base_url=worker_url)
    elif provider == "gemini":
        if not gemini_key:
            raise ValueError("gemini_api_key is required for gemini embedding provider")
        return GeminiEmbeddingProvider(api_key=gemini_key, model_name=gemini_model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

