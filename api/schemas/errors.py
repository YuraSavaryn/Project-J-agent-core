"""Unified error response schemas for the agent-core API."""

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Machine-readable error detail."""

    code: str = Field(description="Short snake_case error code.", examples=["validation_error"])
    message: str = Field(description="Human-readable error description.")
    field: str | None = Field(
        default=None,
        description="Field name that caused the error (for validation errors).",
    )


class ErrorResponse(BaseModel):
    """Standard error envelope returned on all non-2xx responses."""

    error: ErrorDetail

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "code": "validation_error",
                        "message": "field 'message' must not be empty",
                        "field": "message",
                    }
                }
            ]
        }
    }

