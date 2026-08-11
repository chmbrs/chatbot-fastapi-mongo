# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Install deps only (no project code yet) so this layer is cached across code changes.
RUN uv sync --frozen --no-install-project --no-dev

# Not needed at runtime, but the test stage (below) asserts every Settings
# field is documented here — that check needs the file in the build context.
COPY .env.example ./

COPY app/ ./app/
RUN uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

# Non-root user — no reason for this process to run as root.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
# Same WORKDIR as the builder stage: uv bakes an absolute shebang into venv
# console scripts at sync time, so the venv only works if the path it was
# built at (/app) matches the path it's copied to here.
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app app/ ./app/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

# slim images don't ship curl; urllib is stdlib and needs no extra install.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/api/health', timeout=2)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM builder AS test

COPY tests/ ./tests/
RUN uv sync --frozen

CMD ["uv", "run", "pytest", "-v"]
