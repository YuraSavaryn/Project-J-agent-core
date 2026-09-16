"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.exception_handlers import (
    http_exception_handler,
    internal_error_handler,
    llm_unavailable_handler,
    validation_exception_handler,
)
from api.middleware import RequestIdMiddleware
from api.routes import chat, health, analytics, sessions
from core.llm.router import LLMUnavailableError
from core.logging_config import configure_logging

# ── Logging ───────────────────────────────────────────────────────────────────
# Must be configured before FastAPI creates any loggers of its own.
configure_logging()

app = FastAPI(
    title="agent-core",
    description="Transport-agnostic AI agent brain service.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
# RequestIdMiddleware must be added first (outermost) so the request_id is
# available in all subsequent middleware and request handlers.
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ────────────────────────────────────────────────────────
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(LLMUnavailableError, llm_unavailable_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, internal_error_handler)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(chat.router, prefix="/v1")
app.include_router(analytics.router)
app.include_router(sessions.router, prefix="/v1")
