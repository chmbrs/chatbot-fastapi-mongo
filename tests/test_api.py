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


def test_the_roster_lists_every_conversation_by_handle(client):
    created = client.post("/api/conversations", json={"title": "payments API"})
    conversation_id = created.json()["id"]

    roster = client.get("/api/agents")
    assert roster.status_code == 200
    entry = next(c for c in roster.json() if c["id"] == conversation_id)
    assert entry["handle"] == created.json()["handle"]
    assert entry["handle"].endswith(conversation_id[-4:])


def test_an_inbound_policy_round_trips(client):
    created = client.post("/api/conversations", json={})
    conversation_id = created.json()["id"]
    assert created.json()["inbound"] == "accept"

    updated = client.put(
        f"/api/conversations/{conversation_id}/inbound", json={"policy": "hold"}
    )
    assert updated.status_code == 200
    assert updated.json()["inbound"] == "hold"

    fetched = client.get(f"/api/conversations/{conversation_id}")
    assert fetched.json()["inbound"] == "hold"


def test_send_delivers_and_the_reply_lands_in_the_other_conversation(client):
    """No sleeps: TestClient drives the background task to completion inside
    client.post() itself (see app/routes.py's send_peer_message docstring),
    so the receiving turn has already run by the time this call returns."""
    sender = client.post("/api/conversations", json={"title": "sender"}).json()
    receiver = client.post("/api/conversations", json={"title": "receiver"}).json()

    response = client.post(
        f"/api/conversations/{sender['id']}/send",
        json={"to_handle": receiver["handle"], "text": "ping"},
    )
    assert response.status_code == 202
    assert response.json()["outcome"] == "delivered"

    # Not an exact-length assertion: with the default PEER_HOP_LIMIT, a demo
    # reply is real content, so the exchange forwards back to the sender too
    # (see app/peers.py's run_exchange) — checking the first two rows pins
    # what this test is actually about without coupling it to the hop count.
    receiver_messages = client.get(f"/api/conversations/{receiver['id']}/messages").json()
    assert receiver_messages[0]["role"] == "peer"
    assert receiver_messages[0]["peer"]["from_handle"] == sender["handle"]
    assert receiver_messages[1]["role"] == "assistant"
    assert receiver_messages[1]["status"] == "complete"


def test_send_to_an_unknown_handle_is_404(client):
    sender = client.post("/api/conversations", json={}).json()

    response = client.post(
        f"/api/conversations/{sender['id']}/send",
        json={"to_handle": "nobody-home", "text": "hello?"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "peer_not_found"


def test_a_refused_send_returns_202_with_no_new_message(client):
    sender = client.post("/api/conversations", json={}).json()
    receiver = client.post("/api/conversations", json={}).json()
    client.put(f"/api/conversations/{receiver['id']}/inbound", json={"policy": "refuse"})

    response = client.post(
        f"/api/conversations/{sender['id']}/send",
        json={"to_handle": receiver["handle"], "text": "ping"},
    )
    assert response.status_code == 202
    assert response.json()["outcome"] == "refused"
    assert client.get(f"/api/conversations/{receiver['id']}/messages").json() == []


def test_a_held_send_appears_in_the_inbox_and_approving_it_delivers(client):
    sender = client.post("/api/conversations", json={}).json()
    receiver = client.post("/api/conversations", json={}).json()
    client.put(f"/api/conversations/{receiver['id']}/inbound", json={"policy": "hold"})

    sent = client.post(
        f"/api/conversations/{sender['id']}/send",
        json={"to_handle": receiver["handle"], "text": "ping"},
    )
    assert sent.json()["outcome"] == "held"

    inbox = client.get(f"/api/conversations/{receiver['id']}/inbox").json()
    assert len(inbox) == 1
    assert inbox[0]["text"] == "ping"
    held_id = inbox[0]["id"]

    approved = client.post(f"/api/conversations/{receiver['id']}/inbox/{held_id}/approve")
    assert approved.status_code == 202

    # The approved held row is gone. A conversation left on `hold` still
    # holds anything that bounces back to it later in the *same* exchange
    # (see app/peers.py's run_exchange), so the inbox isn't necessarily
    # empty afterward — only this specific row is.
    remaining_ids = {
        h["id"] for h in client.get(f"/api/conversations/{receiver['id']}/inbox").json()
    }
    assert held_id not in remaining_ids

    receiver_messages = client.get(f"/api/conversations/{receiver['id']}/messages").json()
    assert [m["role"] for m in receiver_messages] == ["peer", "assistant"]


def test_denying_a_held_message_removes_it_without_running_a_turn(client):
    sender = client.post("/api/conversations", json={}).json()
    receiver = client.post("/api/conversations", json={}).json()
    client.put(f"/api/conversations/{receiver['id']}/inbound", json={"policy": "hold"})
    client.post(
        f"/api/conversations/{sender['id']}/send",
        json={"to_handle": receiver["handle"], "text": "ping"},
    )
    held_id = client.get(f"/api/conversations/{receiver['id']}/inbox").json()[0]["id"]

    denied = client.delete(f"/api/conversations/{receiver['id']}/inbox/{held_id}")
    assert denied.status_code == 204

    assert client.get(f"/api/conversations/{receiver['id']}/inbox").json() == []
    assert client.get(f"/api/conversations/{receiver['id']}/messages").json() == []


def test_a_conversation_cannot_send_to_itself(client):
    conversation = client.post("/api/conversations", json={}).json()

    response = client.post(
        f"/api/conversations/{conversation['id']}/send",
        json={"to_handle": conversation["handle"], "text": "echo"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "cannot_message_self"


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


def test_llm_provider_can_be_switched_to_ollama_and_back_with_no_key_set(client):
    """The no-restart half of the demo/Ollama toggle: no key means neither
    provider needs one, so this switches app.state.llm live. Ollama's own
    reachability isn't asserted here — a dead endpoint fails the next turn,
    named, same as it does for a provider picked at startup.
    """
    assert client.get("/api/health").json()["llm"]["provider"] == "demo"

    switched = client.put("/api/settings/llm-provider", json={"provider": "ollama"})
    assert switched.status_code == 200
    assert switched.json()["provider"] == "ollama"
    assert client.get("/api/health").json()["llm"]["provider"] == "ollama"

    back = client.put("/api/settings/llm-provider", json={"provider": "demo"})
    assert back.status_code == 200
    assert client.get("/api/health").json()["llm"]["provider"] == "demo"


def test_llm_provider_cannot_be_switched_once_a_key_is_configured(app_settings):
    """A real key already answers "which provider" (README: a present key
    always wins). Switching it out from under whoever set it would be a
    silent override of that decision, so this 400s instead.
    """
    app_settings.llm_api_key = "sk-test"
    with TestClient(create_app(app_settings)) as client:
        assert client.get("/api/health").json()["llm"]["has_api_key"] is True

        response = client.put("/api/settings/llm-provider", json={"provider": "demo"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "provider_switch_not_allowed"
