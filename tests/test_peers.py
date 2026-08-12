"""Conversation-to-conversation messaging (app/peers.py). Same fixtures and
style as test_chat.py — real throwaway Mongo, FakeLLMClient, no network —
because deliver() and run_exchange() are exercised through the same
ChatService the HTTP layer uses, not a mock of it.
"""

import pytest

from app.chat import DEFAULT_TITLE, ChatService
from app.errors import AppError, CannotMessageSelf, PeerNotFound
from app.llm.base import LLMChunk
from app.peers import deliver, run_exchange
from tests.fakes import FakeLLMClient


async def test_a_delivered_message_persists_a_peer_row_naming_the_sender(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="pong")])
    service = ChatService(repository, llm)

    delivery = await deliver(
        repository,
        to_handle=receiver.handle,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
    )
    assert delivery.outcome == "delivered"

    await run_exchange(
        repository,
        service,
        to_conversation_id=delivery.to_conversation_id,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
        hop_limit=1,
    )

    messages = await repository.list_messages(receiver.id)
    assert [m.role for m in messages] == ["peer", "assistant"]
    assert messages[0].content == "ping"
    assert messages[0].peer.from_handle == sender.handle
    assert messages[0].peer.hops == 0
    assert messages[1].status == "complete"
    assert messages[1].content == "pong"


async def test_a_peer_message_reaches_the_model_naming_its_sender(repository):
    """peer is not an OpenAI role — the model must see it as a user turn that
    names who it came from, so a reply can plausibly respond to it."""
    sender = await repository.create_conversation(title="payments-api", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="pong")])
    service = ChatService(repository, llm)

    await run_exchange(
        repository,
        service,
        to_conversation_id=receiver.id,
        text="the migration finished",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
        hop_limit=1,
    )

    assert llm.calls[0] == [
        {"role": "user", "content": f"[message from @{sender.handle}]\nthe migration finished"}
    ]


async def test_a_first_peer_message_titles_the_conversation_via_the_llm(repository):
    """run_peer_turn shares run_turn's title-generation path (app/chat.py's
    _generate_title): a fresh conversation's first message being a peer
    message rather than a user one still gets an LLM-written title, not the
    raw truncation heuristic."""
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title=DEFAULT_TITLE, model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="Quicksort basics")])
    service = ChatService(repository, llm)

    await run_exchange(
        repository,
        service,
        to_conversation_id=receiver.id,
        text="explain quicksort please",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
        hop_limit=1,
    )

    renamed = await repository.get_conversation(receiver.id)
    assert renamed.title == "Quicksort basics"
    assert len(llm.calls) == 2
    assert "explain quicksort please" in llm.calls[0][0]["content"]


async def test_an_exchange_stops_at_the_hop_limit(repository):
    a = await repository.create_conversation(title="a", model="fake")
    b = await repository.create_conversation(title="b", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="pong")])
    service = ChatService(repository, llm)

    await run_exchange(
        repository,
        service,
        to_conversation_id=b.id,
        text="ping",
        from_conversation_id=a.id,
        from_handle=a.handle,
        hops=0,
        hop_limit=3,
    )

    # hop 0->1: b replies. hop 1->2: a replies (the forwarded reply lands as
    # a peer row in a, and a's own turn answers it). hop 2->3: b replies
    # again, but 3 >= hop_limit stops the exchange *before* forwarding that
    # third reply back to a — so a's transcript gains exactly one exchange.
    b_messages = await repository.list_messages(b.id)
    a_messages = await repository.list_messages(a.id)
    assert [m.role for m in b_messages] == ["peer", "assistant", "peer", "assistant"]
    assert [m.role for m in a_messages] == ["peer", "assistant"]


async def test_hop_limit_of_one_delivers_and_replies_but_never_forwards(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="pong")])
    service = ChatService(repository, llm)

    await run_exchange(
        repository,
        service,
        to_conversation_id=receiver.id,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
        hop_limit=1,
    )

    assert len(await repository.list_messages(receiver.id)) == 2  # peer + assistant
    assert await repository.list_messages(sender.id) == []  # nothing forwarded back


async def test_refuse_policy_writes_no_row_anywhere(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    await repository.set_inbound_policy(receiver.id, "refuse")

    delivery = await deliver(
        repository,
        to_handle=receiver.handle,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
    )

    assert delivery.outcome == "refused"
    assert await repository.list_messages(receiver.id) == []
    assert await repository.list_held_messages(receiver.id) == []


async def test_hold_policy_writes_to_the_inbox_and_runs_no_turn(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    await repository.set_inbound_policy(receiver.id, "hold")

    delivery = await deliver(
        repository,
        to_handle=receiver.handle,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
    )

    assert delivery.outcome == "held"
    held = await repository.list_held_messages(receiver.id)
    assert len(held) == 1
    assert held[0].text == "ping"
    assert held[0].from_handle == sender.handle
    assert await repository.list_messages(receiver.id) == []  # no turn ran


async def test_approving_a_held_message_delivers_it(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    await repository.set_inbound_policy(receiver.id, "hold")
    llm = FakeLLMClient(chunks=[LLMChunk(text="pong")])
    service = ChatService(repository, llm)

    await deliver(
        repository,
        to_handle=receiver.handle,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
    )
    held = (await repository.list_held_messages(receiver.id))[0]

    # What routes.approve_held_message does: drop the held row, then run the
    # exchange it had been waiting to run.
    await repository.delete_held_message(receiver.id, held.id)
    await run_exchange(
        repository,
        service,
        to_conversation_id=receiver.id,
        text=held.text,
        from_conversation_id=held.from_conversation_id,
        from_handle=held.from_handle,
        hops=held.hops,
        hop_limit=1,
    )

    assert await repository.list_held_messages(receiver.id) == []
    messages = await repository.list_messages(receiver.id)
    assert [m.role for m in messages] == ["peer", "assistant"]


async def test_denying_a_held_message_drops_it(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    await repository.set_inbound_policy(receiver.id, "hold")

    await deliver(
        repository,
        to_handle=receiver.handle,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
    )
    held = (await repository.list_held_messages(receiver.id))[0]

    await repository.delete_held_message(receiver.id, held.id)

    assert await repository.list_held_messages(receiver.id) == []
    assert await repository.list_messages(receiver.id) == []


async def test_a_conversation_cannot_message_itself(repository):
    conv = await repository.create_conversation(title="solo", model="fake")

    with pytest.raises(CannotMessageSelf):
        await deliver(
            repository,
            to_handle=conv.handle,
            text="echo",
            from_conversation_id=conv.id,
            from_handle=conv.handle,
            hops=0,
        )


async def test_an_unknown_handle_raises_peer_not_found(repository):
    sender = await repository.create_conversation(title="sender", model="fake")

    with pytest.raises(PeerNotFound):
        await deliver(
            repository,
            to_handle="nobody-home",
            text="hello?",
            from_conversation_id=sender.id,
            from_handle=sender.handle,
            hops=0,
        )


async def test_a_failed_peer_turn_persists_failed_and_is_not_forwarded(repository):
    sender = await repository.create_conversation(title="sender", model="fake")
    receiver = await repository.create_conversation(title="receiver", model="fake")
    llm = FakeLLMClient(error=AppError("boom"))
    service = ChatService(repository, llm)

    await run_exchange(
        repository,
        service,
        to_conversation_id=receiver.id,
        text="ping",
        from_conversation_id=sender.id,
        from_handle=sender.handle,
        hops=0,
        hop_limit=3,
    )

    messages = await repository.list_messages(receiver.id)
    assert messages[-1].status == "failed"
    assert await repository.list_messages(sender.id) == []  # never forwarded


async def test_an_exchange_never_pulls_in_a_third_decoy_conversation(repository):
    """run_exchange only ever swaps between the two original participants —
    it never reads a reply's text looking for another handle to route to.
    A third conversation existing in the system must be left untouched."""
    a = await repository.create_conversation(title="a", model="fake")
    b = await repository.create_conversation(title="b", model="fake")
    decoy = await repository.create_conversation(title="decoy", model="fake")
    llm = FakeLLMClient(chunks=[LLMChunk(text="pong")])
    service = ChatService(repository, llm)

    await run_exchange(
        repository,
        service,
        to_conversation_id=b.id,
        text="ping",
        from_conversation_id=a.id,
        from_handle=a.handle,
        hops=0,
        hop_limit=3,
    )

    assert await repository.list_messages(decoy.id) == []
