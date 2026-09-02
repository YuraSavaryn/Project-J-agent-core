# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml .
# Generate a lockfile if it doesn't exist (CI-friendly)
RUN uv sync --no-dev --no-install-project

# Copy source
COPY . .

# Install project itself
RUN uv sync --no-dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Run with uvicorn (FastAPI[standard] ships uvicorn)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

