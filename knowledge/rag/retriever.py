"""RAG Retriever module.

Coordinates generating an embedding for a query and searching ChromaDB.
"""

import logging
from typing import List

from knowledge.rag.embeddings import EmbeddingProvider
from knowledge.rag.vectorstore.chroma_store import ChromaStore, QueryResult

logger = logging.getLogger(__name__)


class Retriever:
    """Searches the vector store using query text embeddings.

    Parameters
    ----------
    embedding_provider:
        The provider used to generate the embedding for the query.
    vector_store:
        The ChromaStore wrapper to search in.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaStore,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store

    async def retrieve(self, query: str, top_k: int | None = None) -> List[QueryResult]:
        """Retrieve relevant chunks for the given query.

        1. Generates an embedding for the query string.
        2. Queries ChromaDB for the closest vectors (cosine similarity).
        3. Returns the chunks wrapped in ``QueryResult`` with metadata and score.

        Parameters
        ----------
        query:
            The raw text string to search for.
        top_k:
            Number of results to return. If None, defaults to settings.rag_top_k.

        Returns
        -------
        list[QueryResult]
            Ordered list of chunks, most relevant first (highest score).
        """
        logger.debug("Retriever: embedding query: %r", query)
        
        # embed() expects a list of texts, so wrap the query in a list
        vectors = await self._embedder.embed([query])
        
        # We only sent one query, so we get one vector back
        query_vector = vectors[0]

        logger.debug("Retriever: querying ChromaStore for top_k=%s", top_k)
        results = self._store.query(vector=query_vector, top_k=top_k)
        
        logger.info("Retriever: found %d relevant chunks", len(results))
        return results

