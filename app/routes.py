import json
from contextlib import suppress
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from app.chat import DEFAULT_TITLE, ChatService, TurnDelta, TurnDone, TurnStarted
from app.config import Settings
from app.errors import (
    AppError,
    ConversationNotFound,
    HeldMessageNotFound,
    ProviderSwitchNotAllowed,
    error_envelope,
)
from app.llm import build_llm
from app.llm.base import LLMClient
from app.llm.openai_compatible import OpenAICompatibleLLMClient
from app.models import Conversation, Delivery, HeldMessage, InboundPolicy, Message
from app.peers import deliver, run_exchange
from app.repository import Repository

router = APIRouter()

# 24 hex chars — a MongoDB ObjectId. Malformed ids are 422 at the edge,
# never a 500 from inside bson.
ConversationId = Annotated[str, Path(pattern=r"^[0-9a-fA-F]{24}$")]
HeldMessageId = Annotated[str, Path(pattern=r"^[0-9a-fA-F]{24}$")]


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def get_app_settings(request: Request) -> Settings:
    # The exact Settings this app was built with (app/main.py stores it on
    # app.state), not get_settings()'s process-wide singleton: a test app
    # built with its own throwaway Settings must never read or mutate some
    # other app's config, and set_llm_provider below does mutate this.
    return request.app.state.settings


RepositoryDep = Annotated[Repository, Depends(get_repository)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]


class CreateConversationBody(BaseModel):
    title: str | None = None


class RenameConversationBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SendMessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class SendPeerMessageBody(BaseModel):
    to_handle: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=8000)


class SetInboundPolicyBody(BaseModel):
    policy: InboundPolicy


class SetLlmProviderBody(BaseModel):
    # Deliberately narrower than Settings.llm_provider's four values: this
    # endpoint is the no-key toggle only. Picking "openrouter" from here
    # would need a key it doesn't have, and there is nothing to toggle once
    # a key is configured — see the has_api_key gate in health() below.
    provider: Literal["demo", "ollama"]


@router.get("/api/health")
async def health(repository: RepositoryDep, llm: LLMDep, settings: SettingsDep):
    mongo_ok = await repository.ping()
    # isinstance, not a name check: the "provider selected but no key"
    # placeholder client also names itself "openrouter" (see llm/__init__.py)
    # and would otherwise be misreported as fully functional here.
    #
    # "real" means a real endpoint is configured, not that it answered — this
    # never probes the provider. Doing so would spend a unit of a 50-per-day
    # quota on every Docker HEALTHCHECK. A dead endpoint (wrong key, Ollama not
    # running) surfaces on the next turn, named, rather than here.
    llm_is_real = isinstance(llm, OpenAICompatibleLLMClient)

    if not mongo_ok:
        degraded_reason = "mongo is unreachable"
    elif llm_is_real:
        degraded_reason = None
    elif settings.llm_provider == "demo":
        degraded_reason = "LLM_PROVIDER=demo — using the offline provider by explicit choice"
    elif settings.llm_provider == "openrouter":
        degraded_reason = (
            "LLM_PROVIDER=openrouter but no LLM_API_KEY is set — every message will fail"
        )
    else:  # "auto" with no key
        degraded_reason = (
            "no LLM_API_KEY configured — using the offline demo provider "
            "(set LLM_PROVIDER=ollama to use a local model instead)"
        )

    return {
        "status": "ok" if (mongo_ok and llm_is_real) else "degraded",
        "mongo": "ok" if mongo_ok else "unreachable",
        "llm": {
            "provider": llm.name,
            "model": llm.model,
            "configured": settings.llm_configured,
            "degraded_reason": degraded_reason,
            # Distinct from "configured": that's true for ollama even with no
            # key. This is what the UI's demo/Ollama toggle gates on, since a
            # real key already answers "which provider" and isn't something
            # this app switches out from under whoever configured it.
            "has_api_key": settings.llm_api_key is not None,
        },
    }


@router.put("/api/settings/llm-provider")
async def set_llm_provider(body: SetLlmProviderBody, request: Request, settings: SettingsDep):
    """The runtime half of the demo/Ollama toggle in the UI. No key means
    nothing here is really "configured" yet, so switching between the two
    keyless providers is safe to do live, no restart. Once LLM_API_KEY is
    set, this always 400s: a real key already answers "which provider", and
    silently overriding it would contradict the one-directional resolution
    documented in README (a present key always wins)."""
    if settings.llm_api_key is not None:
        raise ProviderSwitchNotAllowed()

    settings.llm_provider = body.provider
    llm = build_llm(settings)
    request.app.state.llm = llm
    request.app.state.chat_service = ChatService(request.app.state.repository, llm)
    return {"provider": llm.name, "model": llm.model}


@router.post("/api/conversations", status_code=201, response_model=Conversation)
async def create_conversation(body: CreateConversationBody, repository: RepositoryDep, llm: LLMDep):
    # The resolved client's model, not settings.llm_model — those differ for
    # every provider except OpenRouter, and this field should say what actually
    # answered the conversation.
    return await repository.create_conversation(title=body.title or DEFAULT_TITLE, model=llm.model)


@router.get("/api/conversations", response_model=list[Conversation])
async def list_conversations(repository: RepositoryDep):
    return await repository.list_conversations()


@router.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: ConversationId, repository: RepositoryDep):
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return conversation


@router.patch("/api/conversations/{conversation_id}", response_model=Conversation)
async def rename_conversation(
    conversation_id: ConversationId, body: RenameConversationBody, repository: RepositoryDep
):
    renamed = await repository.rename_conversation(conversation_id, body.title)
    if not renamed:
        raise ConversationNotFound(conversation_id)
    return await repository.get_conversation(conversation_id)


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: ConversationId, repository: RepositoryDep):
    deleted = await repository.delete_conversation(conversation_id)
    if not deleted:
        raise ConversationNotFound(conversation_id)


@router.get("/api/conversations/{conversation_id}/messages", response_model=list[Message])
async def list_messages(
    conversation_id: ConversationId, repository: RepositoryDep, chat_service: ChatServiceDep
):
    """The one read that mixes stored state with process state. A turn that is
    generating right now is `interrupted` with partial content on disk, which
    is the correct thing to have persisted (see chat.py) but reads, to anyone
    who wasn't the caller that started it, exactly like a turn that died. A
    peer exchange runs server-side with no client attached at all, so that is
    the normal case, not an edge one. `live` is that missing bit, filled in
    from the process rather than the record and never written back to it."""
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    messages = await repository.list_messages(conversation_id)
    return [m.model_copy(update={"live": chat_service.is_generating(m.id)}) for m in messages]


# --- peer messaging (see app/peers.py) --------------------------------------


@router.get("/api/agents", response_model=list[Conversation])
async def list_agents(repository: RepositoryDep):
    """The roster, mirroring Claude Code's `/list-agents`: every conversation
    this app can reach, by the handle it answers to."""
    return await repository.list_conversations()


@router.put("/api/conversations/{conversation_id}/inbound", response_model=Conversation)
async def set_inbound_policy(
    conversation_id: ConversationId, body: SetInboundPolicyBody, repository: RepositoryDep
):
    updated = await repository.set_inbound_policy(conversation_id, body.policy)
    if not updated:
        raise ConversationNotFound(conversation_id)
    return await repository.get_conversation(conversation_id)


@router.post(
    "/api/conversations/{conversation_id}/send", status_code=202, response_model=Delivery
)
async def send_peer_message(
    conversation_id: ConversationId,
    body: SendPeerMessageBody,
    background: BackgroundTasks,
    repository: RepositoryDep,
    chat_service: ChatServiceDep,
    settings: SettingsDep,
):
    """No SSE here: the interesting output of this call lands in *another*
    conversation, not this response, so there's nothing for this caller to
    stream. `delivered` starts the receiving turn as a background task —
    the response returns immediately with the outcome, before that turn (or
    the exchange it may lead to) has produced a single token. See
    app/peers.py's run_exchange for why that turn is already durable with no
    extra machinery: the assistant row it writes is committed as
    `interrupted` before its first token, exactly like every other turn."""
    from_conversation = await repository.get_conversation(conversation_id)
    if from_conversation is None:
        raise ConversationNotFound(conversation_id)

    delivery = await deliver(
        repository,
        to_handle=body.to_handle,
        text=body.text,
        from_conversation_id=conversation_id,
        from_handle=from_conversation.handle,
        hops=0,
    )
    if delivery.outcome == "delivered":
        background.add_task(
            run_exchange,
            repository,
            chat_service,
            to_conversation_id=delivery.to_conversation_id,
            text=body.text,
            from_conversation_id=conversation_id,
            from_handle=from_conversation.handle,
            hops=0,
            hop_limit=settings.peer_hop_limit,
        )
    return delivery


@router.get("/api/conversations/{conversation_id}/inbox", response_model=list[HeldMessage])
async def list_inbox(conversation_id: ConversationId, repository: RepositoryDep):
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return await repository.list_held_messages(conversation_id)


@router.post(
    "/api/conversations/{conversation_id}/inbox/{held_id}/approve",
    status_code=202,
    response_model=Delivery,
)
async def approve_held_message(
    conversation_id: ConversationId,
    held_id: HeldMessageId,
    background: BackgroundTasks,
    repository: RepositoryDep,
    chat_service: ChatServiceDep,
    settings: SettingsDep,
):
    conversation = await repository.get_conversation(conversation_id)
    held = await repository.get_held_message(conversation_id, held_id)
    if conversation is None or held is None:
        raise HeldMessageNotFound(held_id)
    await repository.delete_held_message(conversation_id, held_id)

    background.add_task(
        run_exchange,
        repository,
        chat_service,
        to_conversation_id=conversation_id,
        text=held.text,
        from_conversation_id=held.from_conversation_id,
        from_handle=held.from_handle,
        hops=held.hops,
        hop_limit=settings.peer_hop_limit,
    )
    return Delivery(outcome="delivered", to_handle=conversation.handle, to_conversation_id=conversation_id)


@router.delete("/api/conversations/{conversation_id}/inbox/{held_id}", status_code=204)
async def deny_held_message(
    conversation_id: ConversationId, held_id: HeldMessageId, repository: RepositoryDep
):
    deleted = await repository.delete_held_message(conversation_id, held_id)
    if not deleted:
        raise HeldMessageNotFound(held_id)


@router.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: ConversationId,
    body: SendMessageBody,
    request: Request,
    chat_service: ChatServiceDep,
):
    # Awaited here, so setup failures (an unknown conversation) raise before a
    # response type is chosen and come back as a real 404 on both renderers —
    # rather than, on the SSE path, a committed 200 carrying an error frame.
    turn = await chat_service.run_turn(conversation_id, body.content)
    if "text/event-stream" in request.headers.get("accept", ""):
        return EventSourceResponse(_render_sse(turn, request))
    return await _render_json(turn, success_status=201)


@router.post("/api/conversations/{conversation_id}/retry")
async def retry_message(
    conversation_id: ConversationId, request: Request, chat_service: ChatServiceDep
):
    turn = await chat_service.retry_turn(conversation_id)
    if "text/event-stream" in request.headers.get("accept", ""):
        return EventSourceResponse(_render_sse(turn, request))
    return await _render_json(turn, success_status=200)


async def _render_sse(turn, request: Request):
    """One generator, two renderers — this is the SSE half. Nothing here
    duplicates chat.py's lifecycle logic; it only translates events to wire
    format. `error` is a normal SSE event, not an HTTP status: by the time
    an AppError can be raised here, `start` (and possibly `delta`) events
    may already be on the wire, so the HTTP status is already committed —
    this is the one AppError path that can't just propagate to the global
    handler in main.py, which is why it's caught here instead.

    Nothing here has to run for the turn to be persisted honestly — see
    chat.py: the assistant row is already on disk, reading `interrupted`,
    before the first frame below is written. That is deliberate. An earlier
    version made this function's cleanup path load-bearing, and a disconnect
    would then race async-generator finalization and lose the message.
    """
    try:
        async for event in turn:
            if isinstance(event, TurnStarted):
                yield {
                    "event": "start",
                    "data": json.dumps(
                        {
                            "conversation_id": event.conversation_id,
                            "user_message_id": event.user_message_id,
                            "assistant_message_id": event.assistant_message_id,
                        }
                    ),
                }
            elif isinstance(event, TurnDelta):
                yield {"event": "delta", "data": json.dumps({"text": event.text})}
            elif isinstance(event, TurnDone):
                yield {
                    "event": "done",
                    "data": json.dumps(jsonable_encoder(event.message)),
                }
    except AppError as exc:
        request_id = getattr(request.state, "request_id", None)
        yield {"event": "error", "data": json.dumps(error_envelope(exc, request_id)["error"])}
    finally:
        # On a disconnect this generator is torn down while `turn` is still
        # mid-reply; closing it explicitly is what lets chat.py record the
        # partial text alongside the `interrupted` state it already wrote.
        # Strictly an improvement to the record, never what makes it correct —
        # so a RuntimeError here (raised when something else is already
        # closing the same generator, which then does this same work) is
        # swallowed rather than turned into a traceback in the logs.
        with suppress(RuntimeError):
            await turn.aclose()


async def _render_json(turn, success_status: int) -> JSONResponse:
    """No local AppError handling: nothing has been written to the response
    yet at this point (unlike SSE), so an AppError here can simply propagate
    to main.py's global handler like any other route-level exception — one
    envelope-rendering path, and it's the one that already attaches
    request_id."""
    final_message = None
    async for event in turn:
        if isinstance(event, TurnDone):
            final_message = event.message
    return JSONResponse(status_code=success_status, content=jsonable_encoder(final_message))
