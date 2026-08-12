import json
from contextlib import suppress
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from app.chat import DEFAULT_TITLE, ChatService, TurnDelta, TurnDone, TurnStarted
from app.config import Settings, get_settings
from app.errors import AppError, ConversationNotFound, error_envelope
from app.llm.base import LLMClient
from app.llm.openai_compatible import OpenAICompatibleLLMClient
from app.models import Conversation, Message
from app.repository import Repository

router = APIRouter()

# 24 hex chars — a MongoDB ObjectId. Malformed ids are 422 at the edge,
# never a 500 from inside bson.
ConversationId = Annotated[str, Path(pattern=r"^[0-9a-fA-F]{24}$")]


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


RepositoryDep = Annotated[Repository, Depends(get_repository)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]


class CreateConversationBody(BaseModel):
    title: str | None = None


class RenameConversationBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SendMessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


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
        },
    }


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
async def list_messages(conversation_id: ConversationId, repository: RepositoryDep):
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return await repository.list_messages(conversation_id)


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
