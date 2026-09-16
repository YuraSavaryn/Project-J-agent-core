"""Pydantic schemas for the /v1/chat endpoint."""

from typing import Annotated

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from any client channel."""

    session_id: Annotated[
        str,
        Field(
            description="Unique session identifier. The caller must generate and persist this.",
            examples=["tg-user-12345", "web-session-uuid"],
        ),
    ]
    message: Annotated[
        str,
        Field(
            min_length=1,
            max_length=8192,
            description="User message text.",
            examples=["Розкажи про права дитини в Україні"],
        ),
    ]
    channel: Annotated[
        str,
        Field(
            description="Source channel identifier (telegram, web, …).",
            examples=["telegram", "web"],
        ),
    ] = "web"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "tg-user-12345",
                    "message": "Розкажи про права дитини в Україні",
                    "channel": "telegram",
                }
            ]
        }
    }


class SourceChunk(BaseModel):
    """A single retrieved knowledge chunk included in the response."""

    document_id: str = Field(description="Source document identifier.")
    title: str = Field(description="Document or chunk title.")
    url: str | None = Field(default=None, description="Optional URL to the source.")
    excerpt: str = Field(description="Short excerpt from the chunk.")


class SuggestedAction(BaseModel):
    """A quick-reply / suggested follow-up action for the client to display."""

    label: str = Field(description="Button label shown to the user.")
    payload: str = Field(description="Value sent back as next message if the user taps it.")


class ChatResponse(BaseModel):
    """Response returned by the /v1/chat endpoint."""

    session_id: str = Field(description="Echo of the incoming session_id.")
    reply: str = Field(description="Agent's text reply.")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Knowledge chunks used to generate the reply.",
    )
    suggested_actions: list[SuggestedAction] = Field(
        default_factory=list,
        description="Optional quick-reply suggestions for the client.",
    )
    query_log_id: int | None = Field(
        default=None,
        description="ID of the logged query, used for submitting feedback.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "tg-user-12345",
                    "reply": "Права дитини в Україні захищаються Конвенцією ООН...",
                    "sources": [],
                    "suggested_actions": [],
                }
            ]
        }
    }

