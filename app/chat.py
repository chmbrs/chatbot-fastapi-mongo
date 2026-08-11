"""The turn lifecycle, as one async generator — the core of this submission.

Imports neither `fastapi` nor `openai` directly: only the LLMClient Protocol
and the Repository. That's what makes this module testable with a fake LLM
and no real database.

The user's message is committed before any provider call — a failed
completion must never eat the user's turn. Whatever the generator's exit
path is — normal completion, an AppError, or cancellation (a client
disconnect and the Stop button look identical here: the server genuinely
can't tell them apart, verified against sse-starlette's actual behavior,
not assumed) — the assistant message is written exactly once, in a
`finally`, shielded so a second cancellation can't cut the write short.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from bson import ObjectId

from app.errors import AppError, ConversationNotFound, NothingToRetry
from app.llm.base import LLMClient
from app.models import Message, MessageStatus
from app.repository import Repository

HISTORY_LIMIT = 20
DEFAULT_TITLE = "New conversation"
_TITLE_MAX_LENGTH = 60


@dataclass(frozen=True)
class TurnStarted:
    conversation_id: str
    user_message_id: str
    assistant_message_id: str


@dataclass(frozen=True)
class TurnDelta:
    text: str


@dataclass(frozen=True)
class TurnDone:
    message: Message


StreamEvent = TurnStarted | TurnDelta | TurnDone


class ChatService:
    def __init__(self, repository: Repository, llm: LLMClient):
        self._repository = repository
        self._llm = llm

    async def run_turn(self, conversation_id: str, content: str) -> AsyncIterator[StreamEvent]:
        # insert_message only validates id *format* — a well-formed but
        # nonexistent id would otherwise insert an orphaned message, so
        # existence is checked explicitly, once, up front.
        conversation = await self._repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)

        user_message = await self._repository.insert_message(
            conversation_id, role="user", content=content, status="complete"
        )

        history = await self._repository.list_messages(conversation_id)
        if len(history) == 1 and conversation.title == DEFAULT_TITLE:
            await self._repository.rename_conversation(conversation_id, _derive_title(content))

        async for event in self._generate(conversation_id, history, user_message.id):
            yield event

    async def retry_turn(self, conversation_id: str) -> AsyncIterator[StreamEvent]:
        conversation = await self._repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)

        history = await self._repository.list_messages(conversation_id)
        if not history or history[-1].role != "assistant" or history[-1].status == "complete":
            raise NothingToRetry(conversation_id)

        failed_reply = history[-1]
        await self._repository.delete_message(conversation_id, failed_reply.id)
        remaining_history = history[:-1]
        user_message_id = remaining_history[-1].id

        async for event in self._generate(conversation_id, remaining_history, user_message_id):
            yield event

    async def _generate(
        self, conversation_id: str, history: list[Message], user_message_id: str
    ) -> AsyncIterator[StreamEvent]:
        llm_messages = [
            {"role": m.role, "content": m.content} for m in history if m.status == "complete"
        ][-HISTORY_LIMIT:]

        assistant_id = ObjectId()
        yield TurnStarted(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=str(assistant_id),
        )

        accumulated = ""
        usage: dict | None = None
        ttft_ms: int | None = None
        status: MessageStatus = "complete"
        error: AppError | None = None
        started_at = time.monotonic()

        try:
            async for chunk in self._llm.stream(llm_messages):
                if chunk.text:
                    if ttft_ms is None:
                        ttft_ms = int((time.monotonic() - started_at) * 1000)
                    accumulated += chunk.text
                    yield TurnDelta(text=chunk.text)
                if chunk.usage is not None:
                    usage = chunk.usage
        except AppError as exc:
            status = "failed"
            error = exc
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        finally:
            total_ms = int((time.monotonic() - started_at) * 1000)
            # Shielded: this write must complete even if the task is
            # cancelled a second time while we're cleaning up from the first.
            assistant_message = await asyncio.shield(
                self._finalize(
                    conversation_id=conversation_id,
                    assistant_id=assistant_id,
                    content=accumulated,
                    status=status,
                    error=error,
                    usage=usage,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                )
            )

        if error is not None:
            raise error

        yield TurnDone(message=assistant_message)

    async def _finalize(
        self,
        *,
        conversation_id: str,
        assistant_id: ObjectId,
        content: str,
        status: MessageStatus,
        error: AppError | None,
        usage: dict | None,
        ttft_ms: int | None,
        total_ms: int | None,
    ) -> Message:
        error_payload = (
            {
                "code": error.code,
                "message": error.message,
                "retry_after_seconds": error.retry_after_seconds,
            }
            if error is not None
            else None
        )
        message = await self._repository.insert_message(
            conversation_id,
            role="assistant",
            content=content,
            status=status,
            message_id=assistant_id,
            error=error_payload,
            provider=self._llm.name,
            model=self._llm.model,
            usage=usage,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
        )
        await self._repository.touch_conversation(conversation_id)
        return message


def _derive_title(content: str) -> str:
    content = content.strip()
    if not content:
        return DEFAULT_TITLE
    if len(content) <= _TITLE_MAX_LENGTH:
        return content
    truncated = content[:_TITLE_MAX_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "…"
