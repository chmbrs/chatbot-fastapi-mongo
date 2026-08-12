"""These tests pin the thesis: every turn ends in a state the server chose
and persisted. Each test below maps directly to one of the README's
decisions; see the docstring on app/chat.py.
"""

import asyncio
import json

import pytest

from app.chat import DEFAULT_TITLE, ChatService, TurnDelta, TurnDone, TurnStarted
from app.errors import AppError, ConversationNotFound, NothingToRetry
from app.llm.base import LLMChunk
from app.llm.demo import DemoLLMClient
from app.routes import _render_sse
from tests.fakes import FakeLLMClient


async def test_user_message_persisted_even_when_the_llm_fails_immediately(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(error=AppError("boom"))
    service = ChatService(repository, llm)

    with pytest.raises(AppError):
        async for _ in await service.run_turn(conv.id, "hello"):
            pass

    messages = await repository.list_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "hello"
    assert messages[0].status == "complete"


async def test_successful_turn_persists_complete_with_provider_model_and_timing(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(
        chunks=[
            LLMChunk(text="Hello"),
            LLMChunk(text=" there"),
            LLMChunk(text="", usage={"prompt_tokens": 5, "completion_tokens": 2}),
        ]
    )
    service = ChatService(repository, llm)

    events = [event async for event in await service.run_turn(conv.id, "hi")]

    assert isinstance(events[0], TurnStarted)
    assert all(isinstance(e, TurnDelta) for e in events[1:-1])
    assert isinstance(events[-1], TurnDone)

    assistant = events[-1].message
    assert assistant.content == "Hello there"
    assert assistant.status == "complete"
    assert assistant.provider == "fake"
    assert assistant.model == "fake-model"
    assert assistant.usage == {"prompt_tokens": 5, "completion_tokens": 2}
    assert assistant.ttft_ms is not None
    assert assistant.total_ms is not None


async def test_llm_error_persists_failed_with_partial_content_and_error_payload(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(
        chunks=[LLMChunk(text="partial ")],
        error=AppError("provider exploded", retry_after_seconds=5),
    )
    service = ChatService(repository, llm)

    with pytest.raises(AppError):
        async for _ in await service.run_turn(conv.id, "hi"):
            pass

    messages = await repository.list_messages(conv.id)
    assistant = messages[-1]
    assert assistant.status == "failed"
    assert assistant.content == "partial "
    assert assistant.error.message == "provider exploded"
    assert assistant.error.retry_after_seconds == 5


async def test_cancellation_persists_interrupted_with_whatever_accumulated(repository):
    conv = await repository.create_conversation(title="t", model="fake")

    class SlowLLM:
        name = "fake"
        model = "fake-model"

        async def stream(self, messages):
            yield LLMChunk(text="partial ")
            await asyncio.sleep(10)  # never reached, the task is cancelled first
            yield LLMChunk(text="unreachable")

    service = ChatService(repository, SlowLLM())
    events = []

    async def consume():
        async for event in await service.run_turn(conv.id, "hi"):
            events.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let TurnStarted + the first delta happen
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    messages = await repository.list_messages(conv.id)
    assistant = messages[-1]
    assert assistant.status == "interrupted"
    assert assistant.content == "partial "


async def test_closing_the_sse_renderer_midstream_persists_the_partial(repository):
    """The disconnect path as it actually happens, which the cancellation test
    above does *not* reach: a real disconnect doesn't cancel the turn, it tears
    down the SSE renderer wrapped around it.

    A regression test for a bug that shipped. When the assistant row was
    written once, at the end, from the generator's `finally`, durability
    depended on async-generator finalization, and against the running
    container that lost the message outright on 10 of 16 mid-stream
    disconnects. The row is now written up front, so what this pins is the
    *enrichment*: the partial text that made it through is still recorded, on
    top of a state that was already correct.
    """
    conv = await repository.create_conversation(title="t", model="fake")

    class SlowLLM:
        name = "fake"
        model = "fake-model"

        async def stream(self, messages):
            yield LLMChunk(text="partial ")
            await asyncio.sleep(10)  # never reached
            yield LLMChunk(text="unreachable")

    service = ChatService(repository, SlowLLM())
    renderer = _render_sse(await service.run_turn(conv.id, "hi"), _StubRequest())

    start = await anext(renderer)
    assert start["event"] == "start"
    delta = await anext(renderer)
    assert json.loads(delta["data"])["text"] == "partial "

    # Exactly what sse-starlette does when the client goes away.
    await renderer.aclose()

    # No sleep, no grace period: the write has to already be durable here.
    messages = await repository.list_messages(conv.id)
    assert [(m.role, m.status) for m in messages] == [
        ("user", "complete"),
        ("assistant", "interrupted"),
    ]
    assert messages[-1].content == "partial "


class _StubRequest:
    """_render_sse only touches request.state.request_id, and only on the
    AppError branch. This keeps the test at the renderer level instead of
    dragging in a full ASGI scope."""

    class state:
        request_id = "test-request-id"


async def test_failed_turns_are_excluded_from_the_next_turns_history(repository):
    conv = await repository.create_conversation(title="t", model="fake")

    failing_llm = FakeLLMClient(error=AppError("boom"))
    with pytest.raises(AppError):
        async for _ in await ChatService(repository, failing_llm).run_turn(conv.id, "first"):
            pass

    succeeding_llm = FakeLLMClient(chunks=[LLMChunk(text="ok")])
    async for _ in await ChatService(repository, succeeding_llm).run_turn(conv.id, "second"):
        pass

    sent_contents = [m["content"] for m in succeeding_llm.calls[0]]
    assert sent_contents == ["first", "second"]  # the failed assistant reply is excluded


async def test_interrupted_turns_are_also_excluded_from_history(repository):
    """The filter is `status == "complete"`, so this follows from the same line
    as the `failed` case, but it's the half a reader is likelier to doubt: a
    half-finished sentence from a stopped stream is exactly the kind of thing
    that quietly poisons every later turn if it's fed back to the model.
    """
    conv = await repository.create_conversation(title="t", model="fake")
    await repository.insert_message(conv.id, "user", "first", "complete")
    await repository.insert_message(conv.id, "assistant", "half a sen", "interrupted")

    llm = FakeLLMClient(chunks=[LLMChunk(text="ok")])
    async for _ in await ChatService(repository, llm).run_turn(conv.id, "second"):
        pass

    assert [m["content"] for m in llm.calls[0]] == ["first", "second"]


async def test_missing_conversation_raises_before_calling_the_llm(repository):
    llm = FakeLLMClient(chunks=[LLMChunk(text="should never be sent")])
    service = ChatService(repository, llm)

    with pytest.raises(ConversationNotFound):
        async for _ in await service.run_turn("000000000000000000000000", "hi"):
            pass

    assert llm.calls == []


async def test_start_event_names_the_assistant_row_written_before_the_first_token(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="hi")])
    service = ChatService(repository, llm)

    events = [event async for event in await service.run_turn(conv.id, "hi")]

    start = events[0]
    done = events[-1]
    assert isinstance(start, TurnStarted)
    assert isinstance(done, TurnDone)
    assert start.assistant_message_id == done.message.id


async def test_the_assistant_row_is_on_disk_before_the_first_token(repository):
    """The thesis, made structural rather than aspirational. Mid-stream, with
    no reply text generated yet, no cleanup path run, nothing finalized, the
    database already holds an assistant row reading `interrupted`. Kill the
    process on the very next line and the transcript is still honest. Every
    write after this one only improves the record; none of them is what makes
    it true.
    """
    conv = await repository.create_conversation(title="t", model="fake")

    class BlockingLLM:
        name = "fake"
        model = "fake-model"

        async def stream(self, messages):
            yield LLMChunk(text="first ")
            await asyncio.sleep(10)  # parked here for the assertions below

    turn = await ChatService(repository, BlockingLLM()).run_turn(conv.id, "hi")
    started = await anext(turn)

    messages = await repository.list_messages(conv.id)
    assert [(m.role, m.status) for m in messages] == [
        ("user", "complete"),
        ("assistant", "interrupted"),
    ]
    assert messages[-1].id == started.assistant_message_id
    assert messages[-1].provider == "fake"

    await turn.aclose()


async def test_first_message_title_is_generated_by_the_llm(repository):
    """The title is the model's own summary now, not a truncation of the raw
    question: "In one short sentence, why use two collections instead of
    embedded messages?" made a poor sidebar label; a real summary doesn't.
    Two separate calls happen here, title first, and FakeLLMClient records
    each one in order, so the first call's prompt is checkable directly.
    """
    conv = await repository.create_conversation(title=DEFAULT_TITLE, model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="Quicksort basics")])

    async for _ in await ChatService(repository, llm).run_turn(conv.id, "explain quicksort please"):
        pass

    renamed = await repository.get_conversation(conv.id)
    assert renamed.title == "Quicksort basics"
    assert len(llm.calls) == 2
    assert "explain quicksort please" in llm.calls[0][0]["content"]


async def test_demo_provider_title_uses_the_heuristic_not_the_canned_reply(repository):
    """The demo client's one fixed reply text has nothing to do with what the
    user asked, so it is never a title source; titling falls back to the
    original heuristic exactly as it did before the LLM took over titling.
    """
    conv = await repository.create_conversation(title=DEFAULT_TITLE, model="demo")

    async for _ in await ChatService(repository, DemoLLMClient()).run_turn(
        conv.id, "explain quicksort please"
    ):
        pass

    renamed = await repository.get_conversation(conv.id)
    assert renamed.title == "explain quicksort please"


async def test_title_generation_falls_back_to_the_heuristic_on_llm_failure(repository):
    """A bad title is never worth failing the turn over. The title-generation
    call and the real reply are the same broken LLM here, so this also pins
    that the turn's own failure still surfaces normally afterward; the
    fallback swallows only the title attempt, nothing else.
    """
    conv = await repository.create_conversation(title=DEFAULT_TITLE, model="fake")
    llm = FakeLLMClient(error=AppError("boom"))

    with pytest.raises(AppError):
        async for _ in await ChatService(repository, llm).run_turn(
            conv.id, "explain quicksort please"
        ):
            pass

    renamed = await repository.get_conversation(conv.id)
    assert renamed.title == "explain quicksort please"


async def test_first_message_does_not_override_an_explicit_title(repository):
    conv = await repository.create_conversation(title="my custom title", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="hi")])

    async for _ in await ChatService(repository, llm).run_turn(conv.id, "something else entirely"):
        pass

    unchanged = await repository.get_conversation(conv.id)
    assert unchanged.title == "my custom title"


async def test_retry_regenerates_without_inserting_a_new_user_message(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    failing_llm = FakeLLMClient(error=AppError("boom"))
    with pytest.raises(AppError):
        async for _ in await ChatService(repository, failing_llm).run_turn(conv.id, "hello"):
            pass

    succeeding_llm = FakeLLMClient(chunks=[LLMChunk(text="fixed reply")])
    events = [
        event async for event in await ChatService(repository, succeeding_llm).retry_turn(conv.id)
    ]

    messages = await repository.list_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]  # no second user message
    assert messages[-1].content == "fixed reply"
    assert messages[-1].status == "complete"
    assert isinstance(events[-1], TurnDone)
    assert succeeding_llm.calls[0] == [{"role": "user", "content": "hello"}]


async def test_an_interrupted_reply_is_retryable_too(repository):
    """Stop and a closed tab both land on `interrupted`, and both are things a
    user wants to pick back up, so /retry keys off "not complete", not off
    "failed". Without this the Stop button would be a dead end.
    """
    conv = await repository.create_conversation(title="t", model="fake")
    await repository.insert_message(conv.id, "user", "hello", "complete")
    await repository.insert_message(conv.id, "assistant", "half a sen", "interrupted")

    llm = FakeLLMClient(chunks=[LLMChunk(text="a complete reply")])
    async for _ in await ChatService(repository, llm).retry_turn(conv.id):
        pass

    messages = await repository.list_messages(conv.id)
    assert [m.status for m in messages] == ["complete", "complete"]
    assert messages[-1].content == "a complete reply"
    # The abandoned partial is gone, not left behind next to its replacement.
    assert llm.calls[0] == [{"role": "user", "content": "hello"}]


async def test_a_completed_reply_is_not_retryable(repository):
    """The other side of that condition: /retry must not be a way to delete and
    regenerate a perfectly good answer."""
    conv = await repository.create_conversation(title="t", model="fake")
    async for _ in await ChatService(
        repository, FakeLLMClient(chunks=[LLMChunk(text="fine")])
    ).run_turn(conv.id, "hello"):
        pass

    llm = FakeLLMClient(chunks=[LLMChunk(text="should never be sent")])
    with pytest.raises(NothingToRetry):
        async for _ in await ChatService(repository, llm).retry_turn(conv.id):
            pass

    assert llm.calls == []
    assert (await repository.list_messages(conv.id))[-1].content == "fine"


async def test_an_overlong_generated_title_is_truncated_on_a_word_boundary(repository):
    """`_derive_title` doubles as the format guardrail on whatever the model
    actually returns for a title, for when it ignores the "3 to 6 words"
    instruction and just answers at length instead."""
    conv = await repository.create_conversation(title=DEFAULT_TITLE, model="fake")
    overlong_title = (
        "please explain in detail how mongodb index prefixes work and when a "
        "compound index can serve a query on only its leading field"
    )
    llm = FakeLLMClient(chunks=[LLMChunk(text=overlong_title)])

    async for _ in await ChatService(repository, llm).run_turn(conv.id, "hi"):
        pass

    title = (await repository.get_conversation(conv.id)).title
    assert title.endswith("…")
    assert len(title) <= 61  # 60 chars plus the ellipsis
    assert overlong_title.startswith(title[:-1])
    assert not title[:-1].endswith(" ")  # cut on a word boundary, not mid-word


async def test_retry_with_nothing_to_retry_raises_without_calling_the_llm(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="should never be sent")])

    with pytest.raises(NothingToRetry):
        async for _ in await ChatService(repository, llm).retry_turn(conv.id):
            pass

    assert llm.calls == []


async def test_retry_on_an_orphaned_assistant_message_does_not_crash(repository):
    """Dropping the trailing reply can empty the history. Nothing above this
    layer can produce that state today, but "raise NothingToRetry" and "500 on
    an IndexError" are one line apart, and only one of them is this app's
    promise."""
    conv = await repository.create_conversation(title="t", model="fake")
    await repository.insert_message(conv.id, "assistant", "orphan", "failed")

    llm = FakeLLMClient(chunks=[LLMChunk(text="should never be sent")])
    with pytest.raises(NothingToRetry):
        async for _ in await ChatService(repository, llm).retry_turn(conv.id):
            pass

    assert llm.calls == []
