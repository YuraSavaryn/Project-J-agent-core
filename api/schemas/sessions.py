"""Pydantic schemas for the /v1/sessions endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    """Session metadata returned by ``GET /v1/sessions/{session_id}``."""

    session_id: str = Field(description="Unique session identifier.")
    channel: str = Field(description="Source channel (telegram, web, …).")
    external_user_id: str = Field(
        description="External user identifier (Telegram chat_id, web cookie, …)."
    )
    started_at: datetime = Field(description="When the session was first created.")
    last_active_at: datetime = Field(description="When the session was last active.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "tg-user-12345",
                    "channel": "telegram",
                    "external_user_id": "12345",
                    "started_at": "2026-09-10T15:00:00Z",
                    "last_active_at": "2026-09-10T18:00:00Z",
                }
            ]
        }
    }


class DeleteSessionResponse(BaseModel):
    """Response returned by ``DELETE /v1/sessions/{session_id}``."""

    session_id: str = Field(description="Unique session identifier.")
    deleted: bool = Field(description="``true`` if the session was found and deleted.")

