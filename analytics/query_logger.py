"""Analytics logger for user queries and RAG retrieval."""

import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.db.models import QueryLog, QueryLogChunk, Chunk
from core.db.enums import AnswerSource

logger = logging.getLogger(__name__)


class QueryLogger:
    """Records each query, LLM provider info, latency, and retrieved chunks to Postgres.
    
    Parameters
    ----------
    db:
        Active AsyncSession for database operations.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def log_query(
        self,
        session_id: str,
        query_text: str,
        source: AnswerSource,
        latency_ms: int,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        retrieved_chunks: list[dict] | None = None,
    ) -> QueryLog:
        """Create a QueryLog and associated QueryLogChunk entries."""
        # 1. Create QueryLog
        log_entry = QueryLog(
            session_id=session_id,
            query_text=query_text,
            source=source,
            llm_provider=llm_provider,
            llm_model=llm_model,
            latency_ms=latency_ms,
        )
        self._db.add(log_entry)
        await self._db.flush()  # get log_entry.id
        
        # 2. Add QueryLogChunks if provided
        if retrieved_chunks:
            # We need to map chroma_chunk_ids to DB Chunk.id
            chroma_ids = [c["chunk_id"] for c in retrieved_chunks if "chunk_id" in c]
            if chroma_ids:
                stmt = select(Chunk.id, Chunk.chroma_chunk_id).where(
                    Chunk.chroma_chunk_id.in_(chroma_ids)
                )
                result = await self._db.execute(stmt)
                chunk_id_map = {chroma_id: db_id for db_id, chroma_id in result}
                
                for rc in retrieved_chunks:
                    chroma_id = rc.get("chunk_id")
                    if not chroma_id or chroma_id not in chunk_id_map:
                        continue
                    
                    db_chunk_id = chunk_id_map[chroma_id]
                    link = QueryLogChunk(
                        query_log_id=log_entry.id,
                        chunk_id=db_chunk_id,
                        rank=rc.get("rank", 1),
                        score=rc.get("score"),
                    )
                    self._db.add(link)
                    
        # Commit at caller's level or wait for get_db dependency to commit
        return log_entry
