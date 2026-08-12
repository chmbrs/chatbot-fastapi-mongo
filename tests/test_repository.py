"""Direct tests for the persistence boundary. Most of app/repository.py is
already exercised through test_chat.py and test_api.py, which run against a
real Mongo; what's here is the handful of behaviors those never reach —
each one a claim the module's own comments make about itself.
"""

import json
from datetime import UTC, datetime

from bson import ObjectId

from app.models import Conversation


async def test_deleting_a_conversation_cascades_to_its_messages(repository):
    """No transactions and no replica set (see the README's data-model note),
    so the cascade is two writes, ordered deliberately: messages first, because
    a conversation with no messages is still readable while messages with no
    conversation are unreachable garbage.
    """
    conversation = await repository.create_conversation(title="doomed", model="fake")
    await repository.insert_message(conversation.id, "user", "hi", "complete")
    await repository.insert_message(conversation.id, "assistant", "hello", "complete")

    assert await repository.delete_conversation(conversation.id) is True

    assert await repository.get_conversation(conversation.id) is None
    assert await repository.messages.count_documents({}) == 0


async def test_deleting_a_message_cannot_cross_a_conversation_boundary(repository):
    """delete_message scopes its filter by both ids. The retry path is what
    calls it — a bug here would let one conversation's retry delete another's
    reply."""
    mine = await repository.create_conversation(title="mine", model="fake")
    theirs = await repository.create_conversation(title="theirs", model="fake")
    their_message = await repository.insert_message(theirs.id, "user", "hi", "complete")

    assert await repository.delete_message(mine.id, their_message.id) is False
    assert len(await repository.list_messages(theirs.id)) == 1


async def test_conversations_are_listed_most_recently_updated_first(repository):
    """The sidebar's order, and the reason for the {updated_at: -1} index.
    Timestamps are set explicitly rather than by calling touch_conversation:
    two creates and a touch can all land inside one millisecond, and at that
    point "most recently updated" is a question with no answer to test.
    """
    older = await repository.create_conversation(title="older", model="fake")
    newer = await repository.create_conversation(title="newer", model="fake")
    await _set_updated_at(repository, older.id, datetime(2026, 8, 11, 9, 0, tzinfo=UTC))
    await _set_updated_at(repository, newer.id, datetime(2026, 8, 11, 10, 0, tzinfo=UTC))

    listed = await repository.list_conversations()

    assert [c.title for c in listed] == ["newer", "older"]


async def test_touch_conversation_bumps_updated_at(repository):
    """What actually drives that order — called once per turn, so a conversation
    the user is chatting in rises to the top of the sidebar."""
    conversation = await repository.create_conversation(title="t", model="fake")
    stale = datetime(2020, 1, 1, tzinfo=UTC)
    await _set_updated_at(repository, conversation.id, stale)

    await repository.touch_conversation(conversation.id)

    refreshed = await repository.get_conversation(conversation.id)
    assert refreshed.updated_at > stale


async def _set_updated_at(repository, conversation_id: str, when: datetime) -> None:
    await repository.conversations.update_one(
        {"_id": ObjectId(conversation_id)}, {"$set": {"updated_at": when}}
    )


async def test_malformed_ids_return_empty_rather_than_raising(repository):
    """_parse_id swallows bson's InvalidId so a bad id can never surface as a
    500 from inside the driver. Routes reject these at the edge with a 422
    (test_api.py); this is the belt to that's suspenders."""
    assert await repository.get_conversation("not-an-object-id") is None
    assert await repository.list_messages("not-an-object-id") == []
    assert await repository.rename_conversation("not-an-object-id", "x") is False
    assert await repository.delete_conversation("not-an-object-id") is False
    assert await repository.insert_message("not-an-object-id", "user", "hi", "complete") is None


async def test_ensure_indexes_is_idempotent(repository):
    """Called on every boot, including against a database that already has
    them — a restart must not fail here."""
    await repository.ensure_indexes(retries=1, delay_seconds=0)
    await repository.ensure_indexes(retries=1, delay_seconds=0)

    conversation_indexes = await repository.conversations.index_information()
    message_indexes = await repository.messages.index_information()

    assert [("updated_at", -1), ("_id", -1)] in [v["key"] for v in conversation_indexes.values()]
    assert [("conversation_id", 1), ("created_at", 1), ("_id", 1)] in [
        v["key"] for v in message_indexes.values()
    ]


async def test_documents_round_trip_through_the_models_unchanged(repository):
    """Mongo hands back ObjectId and tz-aware datetimes; the models coerce ids
    to str at this boundary so nothing above repository.py ever sees a bson
    type. Timestamps are compared at millisecond resolution because that is
    BSON's: `datetime.now(UTC)` has microseconds, the stored value does not.
    """
    created = await repository.create_conversation(title="round trip", model="fake")
    fetched = await repository.get_conversation(created.id)

    assert isinstance(fetched, Conversation)
    assert isinstance(fetched.id, str)
    assert (fetched.id, fetched.title, fetched.model) == (created.id, created.title, created.model)
    assert abs((fetched.created_at - created.created_at).total_seconds()) < 0.001
    assert fetched.created_at.tzinfo is not None

    message = await repository.insert_message(created.id, "user", "hi", "complete")
    assert isinstance(message.id, str)
    assert message.conversation_id == created.id


async def test_messages_in_the_same_millisecond_are_ordered_by_id(repository):
    """The one that matters most in this file. BSON timestamps are
    millisecond-precision and a turn's two messages usually land inside the same
    one, so `sort(created_at)` alone leaves the transcript formally unordered —
    and an unordered transcript is not a display bug, it's the wrong
    conversation being replayed to the model on the next turn.
    """
    conversation = await repository.create_conversation(title="t", model="fake")
    same_instant = datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)

    # Inserted back-to-front on purpose, so "returned in _id order" can't be
    # satisfied by accident: without the tiebreak these come back in insertion
    # order, which here is the exact reverse of what's asserted below.
    ids = [ObjectId() for _ in range(5)]
    for position, message_id in reversed(list(enumerate(ids))):
        await repository.messages.insert_one(
            {
                "_id": message_id,
                "conversation_id": ObjectId(conversation.id),
                "role": "user",
                "content": str(position),
                "status": "complete",
                "created_at": same_instant,
            }
        )

    listed = await repository.list_messages(conversation.id)

    assert [m.content for m in listed] == ["0", "1", "2", "3", "4"]
    assert [m.id for m in listed] == [str(i) for i in ids]


async def test_the_message_sort_is_served_by_the_index_without_an_in_memory_sort(repository):
    """The trailing _id in the compound index exists so the tiebroken sort stays
    a pure index scan. If someone later trims the index back, this fails rather
    than the app silently growing a SORT stage on every history read."""
    conversation = await repository.create_conversation(title="t", model="fake")
    await repository.insert_message(conversation.id, "user", "hi", "complete")

    plan = (
        await repository.messages.find({"conversation_id": ObjectId(conversation.id)})
        .sort([("created_at", 1), ("_id", 1)])
        .explain()
    )

    stages = json.dumps(plan["queryPlanner"]["winningPlan"])
    assert "IXSCAN" in stages
    assert "SORT" not in stages
