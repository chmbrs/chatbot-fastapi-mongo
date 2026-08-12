"""HTTP-layer tests for routes.py, run against the real app (real Mongo, a
throwaway database per test) with the deterministic demo LLM provider — no
network calls. The turn lifecycle itself is covered in depth in
test_chat.py; this file is about the routes wiring: status codes, id
validation, the dual JSON/SSE renderer, and health reporting.
"""

import json

from fastapi.testclient import TestClient

from app.chat import ChatService
from app.errors import AppError, RateLimited
from app.llm.base import LLMChunk
from app.main import create_app
from tests.fakes import FakeLLMClient


def test_health_is_always_200_and_reports_demo_provider(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["mongo"] == "ok"
    assert body["llm"]["provider"] == "demo"
    assert body["llm"]["configured"] is False
    assert "demo" in body["llm"]["degraded_reason"]


def test_health_is_still_200_when_mongo_is_unreachable(client, monkeypatch):
    """The load-bearing half of "always 200": a health endpoint that 503s when
    its database is down turns a degraded stack into a flapping container under
    Docker's HEALTHCHECK, which is the opposite of what this app promises.
    Repository.ping already swallows PyMongoError into False — what's pinned
    here is that the route does something sane with that False.
    """

    async def ping_fails() -> bool:
        return False

    monkeypatch.setattr(client.app.state.repository, "ping", ping_fails)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["mongo"] == "unreachable"
    assert body["llm"]["degraded_reason"] == "mongo is unreachable"


def test_create_list_get_delete_conversation_round_trip(client):
    created = client.post("/api/conversations", json={"title": "my chat"})
    assert created.status_code == 201
    conversation_id = created.json()["id"]
    assert created.json()["title"] == "my chat"

    listed = client.get("/api/conversations")
    assert listed.status_code == 200
    assert any(c["id"] == conversation_id for c in listed.json())

    fetched = client.get(f"/api/conversations/{conversation_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == conversation_id

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    assert deleted.status_code == 204

    missing = client.get(f"/api/conversations/{conversation_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "conversation_not_found"


def test_malformed_conversation_id_is_422_not_500(client):
    response = client.get("/api/conversations/not-a-valid-object-id")
    assert response.status_code == 422


def test_rename_conversation(client):
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    renamed = client.patch(f"/api/conversations/{conversation_id}", json={"title": "new name"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "new name"


def test_send_message_json_mode_persists_and_returns_the_assistant_reply(client):
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    response = client.post(
        f"/api/conversations/{conversation_id}/messages", json={"content": "hello there"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "assistant"
    assert body["status"] == "complete"
    assert body["provider"] == "demo"

    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hello there"


def test_send_message_to_a_missing_conversation_is_404(client):
    response = client.post(
        "/api/conversations/000000000000000000000000/messages", json={"content": "hi"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


def test_send_message_rejects_empty_content(client):
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    response = client.post(f"/api/conversations/{conversation_id}/messages", json={"content": ""})
    assert response.status_code == 422


def test_send_message_sse_mode_streams_start_delta_and_done_events(client):
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    with client.stream(
        "POST",
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "hi"},
        headers={"accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: start" in body
    assert "event: delta" in body
    assert "event: done" in body


def test_send_message_sse_mode_reports_failure_as_an_error_event_not_an_http_status(client):
    """The streaming half of the failure story. By the time the provider fails,
    `start` (and possibly `delta`) frames are already on the wire and HTTP 200
    is committed — so the failure has to arrive as an SSE `error` event carrying
    the same envelope the JSON renderer would have returned. This is what the
    UI's red bubble and its Retry button are built on.
    """
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    repository = client.app.state.repository
    client.app.state.chat_service = ChatService(
        repository,
        FakeLLMClient(
            chunks=[LLMChunk(text="partial ")], error=RateLimited(retry_after_seconds=30)
        ),
    )

    with client.stream(
        "POST",
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "hi"},
        headers={"accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200  # already committed before the failure
        body = "".join(response.iter_text())

    assert "event: delta" in body
    assert "event: error" in body
    assert "event: done" not in body

    error = json.loads(
        next(
            block.split("data:", 1)[1]
            for block in body.replace("\r\n", "\n").split("\n\n")
            if "event: error" in block
        )
    )
    assert error["code"] == "rate_limited"
    assert error["retry_after_seconds"] == 30
    assert error["request_id"]

    # And the partial text is persisted as `failed`, not lost.
    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
    assert messages[-1]["status"] == "failed"
    assert messages[-1]["content"] == "partial "


def test_conversations_and_messages_survive_a_restart(app_settings):
    """The brief's one hard persistence requirement. Two independent app
    instances over the same database — new FastAPI app, new Mongo client, new
    lifespan — which is what `docker compose down && docker compose up` is, given
    the named `mongo_data` volume. Deliberately not a mock of a restart: the
    second boot re-runs ensure_indexes against a database that already has them,
    so this also pins that startup is idempotent.
    """
    with TestClient(create_app(app_settings)) as first_boot:
        conversation_id = first_boot.post("/api/conversations", json={}).json()["id"]
        first_boot.post(
            f"/api/conversations/{conversation_id}/messages", json={"content": "remember me"}
        )

    with TestClient(create_app(app_settings)) as second_boot:
        conversations = second_boot.get("/api/conversations").json()
        assert [c["id"] for c in conversations] == [conversation_id]
        assert conversations[0]["title"] == "remember me"  # derived title survived too

        messages = second_boot.get(f"/api/conversations/{conversation_id}/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "remember me"
        assert messages[1]["status"] == "complete"


def test_retry_regenerates_the_last_failed_reply(client):
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    # No failed/interrupted reply exists yet — nothing to retry.
    premature = client.post(f"/api/conversations/{conversation_id}/retry")
    assert premature.status_code == 400
    assert premature.json()["error"]["code"] == "nothing_to_retry"

    # The demo provider never fails, so a real failure is forced by swapping
    # in a failing fake for this one call — a test-only seam, not app code.
    repository = client.app.state.repository
    client.app.state.chat_service = ChatService(repository, FakeLLMClient(error=AppError("boom")))
    failed = client.post(f"/api/conversations/{conversation_id}/messages", json={"content": "hi"})
    assert failed.status_code == 500

    client.app.state.chat_service = ChatService(
        repository, FakeLLMClient(chunks=[LLMChunk(text="fixed reply")])
    )
    retried = client.post(f"/api/conversations/{conversation_id}/retry")
    assert retried.status_code == 200
    assert retried.json()["content"] == "fixed reply"

    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]  # no extra user message
    assert messages[-1]["status"] == "complete"


def test_every_error_response_carries_a_request_id(client):
    response = client.get("/api/conversations/000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["request_id"]
    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]


def test_request_id_is_present_even_for_errors_raised_inside_the_chat_generator(client):
    """A regression test: NothingToRetry is raised from inside
    ChatService.retry_turn — a different code path than ConversationNotFound
    (raised directly in a route handler) — and it used to skip the global
    exception handler entirely, silently coming back with request_id: null.
    """
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]

    response = client.post(f"/api/conversations/{conversation_id}/retry")

    assert response.status_code == 400
    assert response.json()["error"]["request_id"]
    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]
