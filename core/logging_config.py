"""Structured logging configuration for agent-core.

Integrates structlog with the standard-library ``logging`` module so that
all existing ``logging.getLogger(__name__)`` calls produce structured JSON
output with a ``request_id`` field that is bound per-request via context vars.

Usage
-----
Call ``configure_logging()`` once at application startup (in ``api/main.py``).
In request middleware, call ``bind_request_id(request_id)`` to attach the
``request_id`` to all log records emitted within that request.

Example structured output (JSON, production)::

    {
      "timestamp": "2024-01-01T12:00:00.000Z",
      "level": "info",
      "logger": "core.orchestrator",
      "request_id": "a1b2c3d4",
      "event": "Orchestrator: session=sess-1, message_len=42"
    }

In development, output is human-readable coloured text (structlog ConsoleRenderer).
"""

import contextvars
import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from config.settings import settings

# ── Context variable for per-request tracing ─────────────────────────────────

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def get_request_id() -> str:
    """Return the current request_id from context, or ``'-'`` if not set."""
    return _request_id_ctx.get()


def bind_request_id(request_id: str) -> contextvars.Token[str]:
    """Bind *request_id* to the current async context.

    Returns the token that can be used to reset the context variable if needed.
    """
    return _request_id_ctx.set(request_id)


# ── Structlog processors ──────────────────────────────────────────────────────


def _add_request_id(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject the current ``request_id`` into every log record."""
    event_dict["request_id"] = get_request_id()
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Remove the ``color_message`` key that uvicorn injects (not needed in JSON)."""
    event_dict.pop("color_message", None)
    return event_dict


# ── Public API ────────────────────────────────────────────────────────────────


def configure_logging() -> None:
    """Configure structlog + stdlib logging for the application.

    Call once at application startup.  After this call:
    - ``logging.getLogger()`` calls produce structlog-processed output.
    - JSON output in production (``APP_ENV != 'development'``).
    - Human-readable coloured output in development.
    - Every log record includes ``request_id`` from the async context.
    """
    is_development = settings.app_env.lower() == "development"

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        _add_request_id,
        _drop_color_message_key,
        structlog.processors.StackInfoRenderer(),
    ]

    if is_development:
        # Pretty coloured output for local development
        processors: list[Any] = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        # JSON for production / Docker logs aggregation
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── Wire structlog into the stdlib logging system ─────────────────────────
    # This ensures that libraries using logging.getLogger() also go through
    # structlog's processors (including our request_id injection).
    stdlib_processors: list[Any] = [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        # These processors run only on stdlib log records (not structlog native):
        foreign_pre_chain=shared_processors,
        # This renderer runs last, after structlog's pre-chain:
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer() if is_development else structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

