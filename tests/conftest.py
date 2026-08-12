"""A fresh, throwaway database per test — see app/repository.py's docstring
on why the Repository boundary is tested against real Mongo rather than a
fake. Requires MONGO_URI to point at a real instance (the `tests` compose
profile provides one on the internal Docker network).
"""

import asyncio
import os
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.repository import Repository, create_client


def _mongo_uri() -> str:
    return os.environ.get("MONGO_URI", "mongodb://mongo:27017")


@pytest.fixture(autouse=True)
def no_outbound_http(monkeypatch):
    """`docker compose --profile test run --rm tests` is green with no API key
    set, and that green only means something if nothing in here can reach the
    network to begin with. This is what makes the claim enforced rather than
    merely true today.

    Patched at httpx's real *transport* classes, deliberately not at the socket
    layer: Mongo is a genuine dependency of these tests and talks over a real
    socket. `httpx.MockTransport` (test_llm_openai_compatible.py) and the TestClient's
    in-process transport are different classes and are unaffected — only actual
    outbound HTTP raises. tests/test_llm_openai_compatible.py proves this guard fires.
    """

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "This test tried to make a real outbound HTTP request. The suite is "
            "hermetic by design — use tests/fakes.py or httpx.MockTransport."
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _blocked)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)


@pytest.fixture
def throwaway_db_name() -> str:
    """A database name that is dropped when the test ends. Sync, so both the
    sync `client` fixture and sync tests can use it; the test Mongo shares the
    `mongo_data` volume with the dev stack, so leaving databases behind would
    litter the developer's own volume on every run."""
    name = f"chatbot_test_{uuid.uuid4().hex[:8]}"
    yield name

    async def _drop() -> None:
        mongo = create_client(_mongo_uri())
        await mongo.drop_database(name)
        await mongo.close()

    asyncio.run(_drop())


# Plain @pytest.fixture, not @pytest_asyncio.fixture: pyproject.toml's
# asyncio_mode="auto" already manages async-generator fixtures the same way,
# same as the sync `client` fixture below needs no special-casing either.
@pytest.fixture
async def repository(throwaway_db_name):
    client = create_client(_mongo_uri())
    repo = Repository(client, throwaway_db_name)
    await repo.ensure_indexes(retries=3, delay_seconds=0.5)
    try:
        yield repo
    finally:
        await repo.close()


@pytest.fixture
def app_settings(throwaway_db_name) -> Settings:
    return Settings(
        _env_file=None,
        mongo_uri=_mongo_uri(),
        mongo_db=throwaway_db_name,
        llm_provider="demo",
    )


@pytest.fixture
def client(app_settings):
    """A TestClient over the real app, wired to its own throwaway database
    and the deterministic demo LLM provider — no network calls anywhere in
    this fixture."""
    with TestClient(create_app(app_settings)) as test_client:
        yield test_client
