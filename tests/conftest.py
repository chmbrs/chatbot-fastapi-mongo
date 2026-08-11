"""A fresh, throwaway database per test — see app/repository.py's docstring
on why the Repository boundary is tested against real Mongo rather than a
fake. Requires MONGO_URI to point at a real instance (the `tests` compose
profile provides one on the internal Docker network).
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.repository import Repository, create_client


def _mongo_uri() -> str:
    return os.environ.get("MONGO_URI", "mongodb://mongo:27017")


# Plain @pytest.fixture, not @pytest_asyncio.fixture: pyproject.toml's
# asyncio_mode="auto" already manages async-generator fixtures the same way,
# same as the sync `client` fixture below needs no special-casing either.
@pytest.fixture
async def repository():
    db_name = f"chatbot_test_{uuid.uuid4().hex[:8]}"
    client = create_client(_mongo_uri())
    repo = Repository(client, db_name)
    await repo.ensure_indexes(retries=3, delay_seconds=0.5)
    try:
        yield repo
    finally:
        await client.drop_database(db_name)
        await repo.close()


@pytest.fixture
def client():
    """A TestClient over the real app, wired to its own throwaway database
    and the deterministic demo LLM provider — no network calls anywhere in
    this fixture."""
    settings = Settings(
        _env_file=None,
        mongo_uri=_mongo_uri(),
        mongo_db=f"chatbot_test_{uuid.uuid4().hex[:8]}",
        llm_provider="demo",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
