import json
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from app.chat import DEFAULT_TITLE, ChatService, TurnDelta, TurnDone, TurnStarted
from app.config import Settings, get_settings
from app.errors import AppError, ConversationNotFound, error_envelope
from app.llm.openrouter import OpenRouterLLMClient
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


def get_llm(request: Request):
    return request.app.state.llm


class CreateConversationBody(BaseModel):
    title: str | None = None


class RenameConversationBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SendMessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


@router.get("/api/health")
async def health(
    repository: Repository = Depends(get_repository),
    llm=Depends(get_llm),
    settings: Settings = Depends(get_settings),
):
    mongo_ok = await repository.ping()
    # isinstance, not llm.name == "openrouter": the "provider selected but no
    # key" placeholder client also names itself "openrouter" (see llm/__init__.py)
    # and would otherwise be misreported as fully functional here.
    llm_is_real = isinstance(llm, OpenRouterLLMClient)

    if not mongo_ok:
        degraded_reason = "mongo is unreachable"
    elif not llm_is_real:
        if settings.llm_provider == "demo":
            degraded_reason = "LLM_PROVIDER=demo — using the offline provider by explicit choice"
        elif not settings.llm_configured:
            degraded_reason = "no LLM_API_KEY configured — using the offline demo provider"
        else:
            degraded_reason = "LLM_PROVIDER=openrouter but no LLM_API_KEY is set"
    else:
        degraded_reason = None

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
async def create_conversation(
    body: CreateConversationBody,
    repository: Repository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
):
    return await repository.create_conversation(
        title=body.title or DEFAULT_TITLE, model=settings.llm_model
    )


@router.get("/api/conversations", response_model=list[Conversation])
async def list_conversations(repository: Repository = Depends(get_repository)):
    return await repository.list_conversations()


@router.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: ConversationId, repository: Repository = Depends(get_repository)
):
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return conversation


@router.patch("/api/conversations/{conversation_id}", response_model=Conversation)
async def rename_conversation(
    conversation_id: ConversationId,
    body: RenameConversationBody,
    repository: Repository = Depends(get_repository),
):
    renamed = await repository.rename_conversation(conversation_id, body.title)
    if not renamed:
        raise ConversationNotFound(conversation_id)
    return await repository.get_conversation(conversation_id)


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: ConversationId, repository: Repository = Depends(get_repository)
):
    deleted = await repository.delete_conversation(conversation_id)
    if not deleted:
        raise ConversationNotFound(conversation_id)


@router.get("/api/conversations/{conversation_id}/messages", response_model=list[Message])
async def list_messages(
    conversation_id: ConversationId, repository: Repository = Depends(get_repository)
):
    conversation = await repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return await repository.list_messages(conversation_id)


@router.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: ConversationId,
    body: SendMessageBody,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
):
    turn = chat_service.run_turn(conversation_id, body.content)
    if "text/event-stream" in request.headers.get("accept", ""):
        return EventSourceResponse(_render_sse(turn, request))
    return await _render_json(turn, success_status=201)


@router.post("/api/conversations/{conversation_id}/retry")
async def retry_message(
    conversation_id: ConversationId,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
):
    turn = chat_service.retry_turn(conversation_id)
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
    handler in main.py, which is why it's caught here instead."""
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
