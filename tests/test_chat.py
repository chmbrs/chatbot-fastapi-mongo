"""These tests pin the thesis: every turn ends in a state the server chose
and persisted. Each test below maps directly to one of the README's
decisions — see the docstring on app/chat.py.
"""

import asyncio

import pytest

from app.chat import DEFAULT_TITLE, ChatService, TurnDelta, TurnDone, TurnStarted
from app.errors import AppError, ConversationNotFound, NothingToRetry
from app.llm.base import LLMChunk
from tests.fakes import FakeLLMClient


async def test_user_message_persisted_even_when_the_llm_fails_immediately(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(error=AppError("boom"))
    service = ChatService(repository, llm)

    with pytest.raises(AppError):
        async for _ in service.run_turn(conv.id, "hello"):
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

    events = [event async for event in service.run_turn(conv.id, "hi")]

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
        async for _ in service.run_turn(conv.id, "hi"):
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
            await asyncio.sleep(10)  # never reached — the task is cancelled first
            yield LLMChunk(text="unreachable")

    service = ChatService(repository, SlowLLM())
    events = []

    async def consume():
        async for event in service.run_turn(conv.id, "hi"):
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


async def test_failed_turns_are_excluded_from_the_next_turns_history(repository):
    conv = await repository.create_conversation(title="t", model="fake")

    failing_llm = FakeLLMClient(error=AppError("boom"))
    with pytest.raises(AppError):
        async for _ in ChatService(repository, failing_llm).run_turn(conv.id, "first"):
            pass

    succeeding_llm = FakeLLMClient(chunks=[LLMChunk(text="ok")])
    async for _ in ChatService(repository, succeeding_llm).run_turn(conv.id, "second"):
        pass

    sent_contents = [m["content"] for m in succeeding_llm.calls[0]]
    assert sent_contents == ["first", "second"]  # the failed assistant reply is excluded


async def test_missing_conversation_raises_before_calling_the_llm(repository):
    llm = FakeLLMClient(chunks=[LLMChunk(text="should never be sent")])
    service = ChatService(repository, llm)

    with pytest.raises(ConversationNotFound):
        async for _ in service.run_turn("000000000000000000000000", "hi"):
            pass

    assert llm.calls == []


async def test_start_event_names_the_assistant_id_before_it_is_persisted(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="hi")])
    service = ChatService(repository, llm)

    events = [event async for event in service.run_turn(conv.id, "hi")]

    start = events[0]
    done = events[-1]
    assert isinstance(start, TurnStarted)
    assert isinstance(done, TurnDone)
    assert start.assistant_message_id == done.message.id


async def test_first_message_derives_a_title_from_its_content(repository):
    conv = await repository.create_conversation(title=DEFAULT_TITLE, model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="hi")])

    async for _ in ChatService(repository, llm).run_turn(conv.id, "explain quicksort please"):
        pass

    renamed = await repository.get_conversation(conv.id)
    assert renamed.title == "explain quicksort please"


async def test_first_message_does_not_override_an_explicit_title(repository):
    conv = await repository.create_conversation(title="my custom title", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="hi")])

    async for _ in ChatService(repository, llm).run_turn(conv.id, "something else entirely"):
        pass

    unchanged = await repository.get_conversation(conv.id)
    assert unchanged.title == "my custom title"


async def test_retry_regenerates_without_inserting_a_new_user_message(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    failing_llm = FakeLLMClient(error=AppError("boom"))
    with pytest.raises(AppError):
        async for _ in ChatService(repository, failing_llm).run_turn(conv.id, "hello"):
            pass

    succeeding_llm = FakeLLMClient(chunks=[LLMChunk(text="fixed reply")])
    events = [event async for event in ChatService(repository, succeeding_llm).retry_turn(conv.id)]

    messages = await repository.list_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]  # no second user message
    assert messages[-1].content == "fixed reply"
    assert messages[-1].status == "complete"
    assert isinstance(events[-1], TurnDone)
    assert succeeding_llm.calls[0] == [{"role": "user", "content": "hello"}]


async def test_retry_with_nothing_to_retry_raises_without_calling_the_llm(repository):
    conv = await repository.create_conversation(title="t", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="should never be sent")])

    with pytest.raises(NothingToRetry):
        async for _ in ChatService(repository, llm).retry_turn(conv.id):
            pass

    assert llm.calls == []
