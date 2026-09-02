"""Health-check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns `{\"status\": \"ok\"}` when the service is running.",
)
async def health() -> HealthResponse:
    """Liveness probe endpoint."""
    return HealthResponse(status="ok")

