# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Use the system Python across both stages (no managed-interpreter download),
# precompile bytecode for faster container startup, and let a cache mount
# survive across builds instead of re-downloading packages every time.
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Install deps only (no project code yet) so this layer is cached across code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Not needed at runtime, but the test stage (below) asserts every Settings
# field is documented here, and that check needs the file in the build context.
COPY .env.example ./

COPY app/ ./app/
# pyproject.toml has no [build-system] table, so uv never installs this
# project as a package regardless, which makes this second sync currently a
# no-op, kept only so the step still does the right thing if a build-system
# is ever added later.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.13-slim AS runtime

# Non-root user: no reason for this process to run as root.
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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD ["uv", "run", "pytest", "-v"]


FROM python:3.13-slim AS ui

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app streamlit_app.py ./
# Without this, Streamlit finds no config.toml in the container, silently
# falls back to its own defaults (default palette, toolbarMode="auto", which
# shows the Deploy button on localhost), and every theme/chrome decision in
# .streamlit/config.toml quietly does nothing. Caught by checking the actual
# running container, not by the native `streamlit run` test, which worked
# only because it happened to run from the repo root where the file sits.
COPY --chown=app:app .streamlit/ ./.streamlit/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=2)"]

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
