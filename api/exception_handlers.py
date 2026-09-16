"""Global exception handlers for the FastAPI application.

All unhandled exceptions are caught here and converted to the unified
``ErrorResponse`` format defined in ``api/schemas/errors.py``.

Registered handlers:
- ``RequestValidationError`` → 422 with field-level details
- ``HTTPException``          → pass-through (already has detail)
- ``LLMUnavailableError``    → 503 Service Unavailable
- ``Exception``              → 500 Internal Server Error (logs traceback)
"""

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas.errors import ErrorDetail, ErrorResponse
from core.llm.router import LLMUnavailableError

logger = logging.getLogger(__name__)


# ── Handler: Pydantic / FastAPI validation errors ─────────────────────────────


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert Pydantic validation errors to the unified ``ErrorResponse`` format.

    Returns one ``ErrorDetail`` per validation error with the ``field`` set to
    the dot-joined location path (e.g. ``"body.message"``).
    """
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first.get("loc", []))
    msg = first.get("msg", "Validation error")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                code="validation_error",
                message=msg,
                field=loc or None,
            )
        ).model_dump(),
    )


# ── Handler: HTTP exceptions (pass-through with unified envelope) ─────────────


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Wrap Starlette/FastAPI HTTP exceptions in the unified ``ErrorResponse`` format."""
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "http_error")
        message = detail.get("message", str(detail))
        field = detail.get("field")
    else:
        code = f"http_{exc.status_code}"
        message = str(detail)
        field = None

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(code=code, message=message, field=field)
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


# ── Handler: LLM unavailable ──────────────────────────────────────────────────


async def llm_unavailable_handler(
    request: Request,
    exc: LLMUnavailableError,
) -> JSONResponse:
    """Return a user-friendly 503 when all LLM providers are down."""
    logger.error(
        "LLMUnavailableError: all providers failed — returning 503",
        extra={"path": request.url.path},
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error=ErrorDetail(
                code="llm_unavailable",
                message=(
                    "Наразі сервіс тимчасово недоступний. "
                    "Спробуйте ще раз через кілька хвилин."
                ),
            )
        ).model_dump(),
    )


# ── Handler: catch-all 500 ────────────────────────────────────────────────────


async def internal_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log the unexpected exception and return a generic 500."""
    logger.exception(
        "Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                code="internal_error",
                message="Внутрішня помилка сервера. Будь ласка, спробуйте ще раз.",
            )
        ).model_dump(),
    )

