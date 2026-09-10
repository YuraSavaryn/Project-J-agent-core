"""Analytics routes (Feedback, etc.)."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core.db.base import get_db
from core.db.models import QueryLog, Feedback

router = APIRouter(prefix="/v1", tags=["analytics"])


class FeedbackRequest(BaseModel):
    query_log_id: int = Field(..., description="ID of the query log to rate.")
    rating: int = Field(..., description="Rating: 1 for 👍, -1 for 👎.")
    comment: str | None = Field(None, description="Optional feedback text.")


@router.post("/feedback", status_code=201)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit user feedback (thumbs up/down) for a specific query log.
    
    If feedback for this query_log_id already exists, it will be updated.
    """
    if request.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="Rating must be 1 or -1")

    # Verify query log exists
    stmt = select(QueryLog.id).where(QueryLog.id == request.query_log_id)
    result = await db.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Query log not found")

    # Upsert logic: attempt insert, handle integrity error or just do a select first
    existing_stmt = select(Feedback).where(Feedback.query_log_id == request.query_log_id)
    existing_feedback = (await db.execute(existing_stmt)).scalar_one_or_none()

    if existing_feedback:
        existing_feedback.rating = request.rating
        existing_feedback.comment = request.comment
        await db.flush()
        # return OK
        return {"status": "updated", "id": existing_feedback.id}
    else:
        new_feedback = Feedback(
            query_log_id=request.query_log_id,
            rating=request.rating,
            comment=request.comment,
        )
        db.add(new_feedback)
        await db.flush()
        return {"status": "created", "id": new_feedback.id}
