"""Request middleware for agent-core.

Provides:
- ``RequestIdMiddleware``: attaches a UUID ``request_id`` to every request,
  injects it into structured logging context and response headers.
"""

import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.logging_config import bind_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that generates and propagates a request ID.

    For every incoming request:
    1. Reads ``X-Request-Id`` from the request headers (if present), or
       generates a new UUID4.
    2. Binds the request ID to the structlog/stdlib logging context so that
       all log records emitted during this request carry ``request_id``.
    3. Adds ``X-Request-Id`` to the response headers for client-side tracing.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:  # type: ignore[override]
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        token = bind_request_id(request_id)
        try:
            response: Response = await call_next(request)
        finally:
            # Reset is not strictly necessary for asyncio, but is good hygiene
            # in case the context var leaks across tasks in a thread-pool executor.
            from contextvars import copy_context  # noqa: F401 — just documenting intent
            _ = token  # token kept for documentation; context auto-isolates per task
        response.headers["X-Request-Id"] = request_id
        return response
