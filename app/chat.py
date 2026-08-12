"""The turn lifecycle, the core of this submission.

Imports neither `fastapi` nor `openai` directly: only the LLMClient Protocol
and the Repository. That's what makes this module testable with a fake LLM
and no real database.

Two writes bracket every turn, and the order of them is the whole design:

1. The user's message is committed before any provider call. A failed
   completion must never eat the user's turn.
2. The assistant's row is committed *before the first token*, already reading
   `interrupted` with empty content: "if nothing further happens, this is
   what happened." Streaming then updates it to `complete`, or to `failed`
   with the error, or enriches the `interrupted` row with whatever text
   arrived.

The point of (2) is that the honest terminal state is persisted by
construction rather than by cleanup. An earlier version wrote the assistant
row once, at the end, from the generator's `finally`. It was elegant on
paper, and it lost the message outright on roughly two thirds of real
mid-stream disconnects, because durability then depended on async-generator
finalization, which under a real disconnect races itself
(`RuntimeError: aclose(): asynchronous generator is already running`). Now
no cleanup path has to run for the persisted state to be true; the `except`
handlers below only improve the record, they are not what makes it correct.

A client disconnect and the Stop button are the same event here: the server
genuinely cannot tell them apart, and both land on `interrupted`.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

from app.errors import AppError, ConversationNotFound, NothingToRetry
from app.llm.base import LLMClient
from app.models import Message, MessageStatus
from app.repository import Repository

HISTORY_LIMIT = 20
DEFAULT_TITLE = "New conversation"
_TITLE_MAX_LENGTH = 60
_TITLE_PROMPT = (
    "Reply with only a short title (3 to 6 words, no punctuation, no quotes) "
    "summarizing what this message is about:\n\n"
)


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
        # Assistant rows this process is generating *right now*. Deliberately
        # in memory and nowhere else: the database records what happened, and
        # a row that is mid-reply has already recorded the honest answer to
        # that question (`interrupted`, see the module docstring). Whether one
        # is still being written is a fact about a running process, not about
        # the record, and it dies correctly with the process -- after a crash
        # or a restart nothing is generating, and every such row then reads
        # exactly what it is.
        #
        # Single-process by construction (one uvicorn worker, see Dockerfile).
        # A second worker would need this shared, and switching providers
        # mid-turn rebuilds this service (routes.py's set_llm_provider), so
        # the turn already running loses its liveness and reads as
        # interrupted until it settles -- both fail toward "not live", which
        # is the direction that only ever understates what's happening.
        self._in_flight: set[str] = set()

    def is_generating(self, message_id: str) -> bool:
        """Whether that assistant row is being written at this instant. The
        one thing a reader cannot tell from the row itself: a turn in flight
        and a turn stopped halfway are the same three fields on disk."""
        return message_id in self._in_flight

    async def run_turn(self, conversation_id: str, content: str) -> AsyncIterator[StreamEvent]:
        """A coroutine that *returns* the stream, deliberately not an async
        generator that wraps one. Setup is awaited here and errors surface at
        the call site; only _generate below is a generator, so the whole
        request has exactly one of them.

        That is a correctness constraint, not a style preference. Closing an
        async generator does not close a generator it happens to be iterating:
        `async for` abandons its iterator when the loop exits by exception, so
        a wrapper here left _generate's finally, and with it the persist-what-
        happened guarantee, to the garbage collector. Nesting them and closing
        both in turn only traded that for `aclose(): asynchronous generator is
        already running` under a real disconnect. One generator has neither
        problem.
        """
        # insert_message only validates id *format*. A well-formed but
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
            title = await self._generate_title(content)
            await self._repository.rename_conversation(conversation_id, title)

        return self._generate(conversation_id, history, user_message.id)

    async def retry_turn(self, conversation_id: str) -> AsyncIterator[StreamEvent]:
        conversation = await self._repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)

        history = await self._repository.list_messages(conversation_id)
        last = history[-1] if history else None
        if last is None or (last.role == "assistant" and last.status == "complete"):
            raise NothingToRetry(conversation_id)

        if last.role == "assistant":
            # A failed or interrupted reply: drop it and regenerate in its place.
            await self._repository.delete_message(conversation_id, last.id)
            history = history[:-1]

        if not history:
            raise NothingToRetry(conversation_id)

        return self._generate(conversation_id, history, history[-1].id)

    async def run_peer_turn(
        self,
        conversation_id: str,
        text: str,
        *,
        from_handle: str,
        from_conversation_id: str,
        hops: int,
    ) -> AsyncIterator[StreamEvent]:
        """The peer twin of run_turn (see app/peers.py): what arrives before
        any provider call is a `peer`-role row instead of a `user`-role one,
        naming the sender it came from. Everything past that point is the
        same lifecycle, the same generator, and the same guarantee, including
        the `return`-not-`yield` correctness constraint documented on
        run_turn above.
        """
        conversation = await self._repository.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)

        peer_message = await self._repository.insert_message(
            conversation_id,
            role="peer",
            content=text,
            status="complete",
            peer={
                "from_handle": from_handle,
                "from_conversation_id": from_conversation_id,
                "hops": hops,
            },
        )

        history = await self._repository.list_messages(conversation_id)
        if len(history) == 1 and conversation.title == DEFAULT_TITLE:
            title = await self._generate_title(text)
            await self._repository.rename_conversation(conversation_id, title)

        return self._generate(conversation_id, history, peer_message.id)

    async def _generate(
        self, conversation_id: str, history: list[Message], user_message_id: str
    ) -> AsyncIterator[StreamEvent]:
        llm_messages = [_to_llm_message(m) for m in history if m.status == "complete"][
            -HISTORY_LIMIT:
        ]

        # Committed before the first token, and already reading `interrupted`:
        # the state that is true if this process is killed on the next line.
        # Everything below only ever improves on it.
        assistant = await self._repository.insert_message(
            conversation_id,
            role="assistant",
            content="",
            status="interrupted",
            provider=self._llm.name,
            model=self._llm.model,
        )
        # Bracketing every exit below, including the ones that don't run their
        # own cleanup: whatever happens, this row stops claiming to be live.
        self._in_flight.add(assistant.id)
        try:
            yield TurnStarted(
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant.id,
            )

            accumulated = ""
            usage: dict | None = None
            ttft_ms: int | None = None
            started_at = time.monotonic()

            async def settle(
                status: MessageStatus, error: AppError | None = None
            ) -> Message | None:
                # Closes over the accumulator deliberately: it reads whatever
                # has streamed by the moment it's called, which is the whole job.
                return await self._settle(
                    conversation_id=conversation_id,
                    assistant_id=assistant.id,
                    status=status,
                    content=accumulated,
                    started_at=started_at,
                    ttft_ms=ttft_ms,
                    usage=usage,
                    error=error,
                )

            # Three exits, three terminal states, and nothing else below this line.
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
                await settle("failed", error=exc)
                raise
            except (asyncio.CancelledError, GeneratorExit):
                # Stop, or a closed tab. Two spellings of one event: a cancelled
                # task raises CancelledError at the yield, a closed generator gets
                # GeneratorExit. Best effort by design: the row already says
                # `interrupted`, so this only adds the text that made it through,
                # and if the disconnect tears us down first we lose the partial
                # text, never the truth about what happened.
                with suppress(Exception):
                    await settle("interrupted")
                raise

            yield TurnDone(message=await settle("complete"))
        finally:
            self._in_flight.discard(assistant.id)

    async def _settle(
        self,
        *,
        conversation_id: str,
        assistant_id: str,
        status: MessageStatus,
        content: str,
        started_at: float,
        ttft_ms: int | None,
        usage: dict | None,
        error: AppError | None = None,
    ) -> Message | None:
        """The turn's second and final write: the placeholder row grows into
        what actually happened."""
        message = await self._repository.update_message(
            conversation_id,
            assistant_id,
            content=content,
            status=status,
            error=(
                {
                    "code": error.code,
                    "message": error.message,
                    "retry_after_seconds": error.retry_after_seconds,
                }
                if error is not None
                else None
            ),
            usage=usage,
            ttft_ms=ttft_ms,
            total_ms=int((time.monotonic() - started_at) * 1000),
        )
        await self._repository.touch_conversation(conversation_id)
        return message

    async def _generate_title(self, content: str) -> str:
        """The title is now the model's own summary of the opener, not a
        truncation of it: "In one short sentence, why use two collections
        instead of embedded messages?" makes a poor sidebar label, and a
        real summary doesn't. Demo mode is excluded on purpose: its one
        fixed reply text has nothing to do with what the user asked, so it
        is never a title source. Any other failure here (a rate limit, a
        cold Ollama load that times out, an SDK bug) falls back to the
        original heuristic, the same as demo mode does, on the same
        reasoning as the `suppress(Exception)` in `_generate` above: a bad
        title is never worth failing the turn over. `_derive_title` doubles
        as the format guardrail on whatever the model actually returns, in
        case it ignores the instruction.
        """
        if self._llm.name != "demo":
            try:
                chunks = [
                    chunk.text
                    async for chunk in self._llm.stream(
                        [{"role": "user", "content": _TITLE_PROMPT + content}]
                    )
                ]
            except Exception:
                pass
            else:
                generated = " ".join("".join(chunks).split()).strip("\"'“”‘’")
                if generated:
                    return _derive_title(generated)
        return _derive_title(content)


async def drain(turn: AsyncIterator[StreamEvent]) -> Message | None:
    """The non-HTTP twin of routes.py's _render_json: fully consumes a turn's
    generator and returns whatever it finally persisted. app/peers.py uses
    this to run a turn with no HTTP client attached, since the receiving
    half of a peer exchange has nothing to stream to."""
    final: Message | None = None
    async for event in turn:
        if isinstance(event, TurnDone):
            final = event.message
    return final


def _to_llm_message(message: Message) -> dict[str, str]:
    """`peer` is not an OpenAI role, so a peer row is rendered as the user
    turn it functionally is, prefixed with who it came from, the same way
    Claude Code tells a receiving session a message came from another
    session rather than from the person at the keyboard."""
    if message.role == "peer":
        handle = message.peer.from_handle if message.peer else "unknown"
        return {"role": "user", "content": f"[message from @{handle}]\n{message.content}"}
    return {"role": message.role, "content": message.content}


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
