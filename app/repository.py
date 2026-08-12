"""The only module in this app that imports pymongo. Everything else talks
to conversations and messages through the Repository below — that boundary
is what makes app/chat.py testable without a real database.

No abstract base class here: there is exactly one implementation and Mongo
isn't being swapped out, so an interface would be ceremony, not design.
"""

import asyncio
from datetime import UTC, datetime

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from app.models import Conversation, HeldMessage, InboundPolicy, Message, MessageStatus

# The _id tiebreak is not decoration. BSON datetimes are millisecond-precision,
# and a user message plus its reply routinely land in the *same* millisecond
# (measured: 32 of 40 turns against the demo provider) — at which point a sort
# on the timestamp alone is formally unordered, and the transcript can come
# back with the answer before the question. ObjectIds embed a timestamp and a
# per-process counter, so within the single process that writes any one turn
# they increase monotonically in exactly the order the rows were created.
_OLDEST_FIRST = [("created_at", 1), ("_id", 1)]
_NEWEST_FIRST = [("updated_at", -1), ("_id", -1)]
_HELD_OLDEST_FIRST = [("created_at", 1), ("_id", 1)]


def create_client(mongo_uri: str) -> AsyncMongoClient:
    return AsyncMongoClient(mongo_uri, tz_aware=True, serverSelectionTimeoutMS=5000)


class Repository:
    def __init__(self, client: AsyncMongoClient, db_name: str):
        self._client = client
        db = client[db_name]
        self.conversations = db["conversations"]
        self.messages = db["messages"]
        self.held_messages = db["held_messages"]

    async def ensure_indexes(self, retries: int = 10, delay_seconds: float = 1.0) -> None:
        """Idempotent, with a bounded retry: a cold `docker compose up` can
        start this container before Mongo finishes accepting connections
        even though the healthcheck says otherwise."""
        last_error: PyMongoError | None = None
        for _ in range(retries):
            try:
                await self.conversations.create_index([("updated_at", -1), ("_id", -1)])
                # {conversation_id: 1} and {conversation_id: 1, created_at: 1} are
                # both prefixes of this one, so the cascade delete and any
                # timestamp-only query are served by it too — no second index.
                # The trailing _id is what keeps the tiebroken sort below a pure
                # index scan instead of an in-memory SORT stage.
                await self.messages.create_index(
                    [("conversation_id", 1), ("created_at", 1), ("_id", 1)]
                )
                # Serves both "this conversation's inbox" and its cascade
                # delete's prefix — same rule as the messages index above.
                await self.held_messages.create_index(
                    [("to_conversation_id", 1), ("created_at", 1), ("_id", 1)]
                )
                return
            except PyMongoError as exc:
                last_error = exc
                await asyncio.sleep(delay_seconds)
        raise RuntimeError("mongo never became ready for index creation") from last_error

    async def close(self) -> None:
        await self._client.close()

    # --- conversations -----------------------------------------------------

    async def create_conversation(self, title: str, model: str) -> Conversation:
        now = datetime.now(UTC)
        doc = {"title": title, "created_at": now, "updated_at": now, "model": model}
        result = await self.conversations.insert_one(doc)
        doc["_id"] = result.inserted_id
        return Conversation.from_doc(doc)

    async def list_conversations(self, limit: int = 50) -> list[Conversation]:
        cursor = self.conversations.find().sort(_NEWEST_FIRST).limit(limit)
        return [Conversation.from_doc(doc) async for doc in cursor]

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        oid = _parse_id(conversation_id)
        if oid is None:
            return None
        doc = await self.conversations.find_one({"_id": oid})
        return Conversation.from_doc(doc) if doc else None

    async def rename_conversation(self, conversation_id: str, title: str) -> bool:
        oid = _parse_id(conversation_id)
        if oid is None:
            return False
        result = await self.conversations.update_one({"_id": oid}, {"$set": {"title": title}})
        return result.matched_count > 0

    async def set_inbound_policy(self, conversation_id: str, policy: InboundPolicy) -> bool:
        oid = _parse_id(conversation_id)
        if oid is None:
            return False
        result = await self.conversations.update_one({"_id": oid}, {"$set": {"inbound": policy}})
        return result.matched_count > 0

    async def touch_conversation(self, conversation_id: str) -> None:
        oid = _parse_id(conversation_id)
        if oid is None:
            return
        await self.conversations.update_one(
            {"_id": oid}, {"$set": {"updated_at": datetime.now(UTC)}}
        )

    async def delete_conversation(self, conversation_id: str) -> bool:
        oid = _parse_id(conversation_id)
        if oid is None:
            return False
        # Cascade first: an orphaned conversation with no messages is
        # recoverable to look at; orphaned messages with no conversation are not.
        await self.messages.delete_many({"conversation_id": oid})
        await self.held_messages.delete_many({"to_conversation_id": oid})
        result = await self.conversations.delete_one({"_id": oid})
        return result.deleted_count > 0

    # --- messages ------------------------------------------------------------

    async def list_messages(self, conversation_id: str) -> list[Message]:
        oid = _parse_id(conversation_id)
        if oid is None:
            return []
        cursor = self.messages.find({"conversation_id": oid}).sort(_OLDEST_FIRST)
        return [Message.from_doc(doc) async for doc in cursor]

    async def insert_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        status: MessageStatus,
        **fields: object,
    ) -> Message | None:
        oid = _parse_id(conversation_id)
        if oid is None:
            return None
        doc = {
            "conversation_id": oid,
            "role": role,
            "content": content,
            "status": status,
            "created_at": datetime.now(UTC),
            **fields,
        }
        result = await self.messages.insert_one(doc)
        doc["_id"] = result.inserted_id
        return Message.from_doc(doc)

    async def update_message(
        self, conversation_id: str, message_id: str, **fields: object
    ) -> Message | None:
        """Returns the updated document, so chat.py never has to re-read what
        it just wrote. Scoped by both ids for the same reason delete_message
        is: an update must not cross a conversation boundary."""
        conv_oid = _parse_id(conversation_id)
        msg_oid = _parse_id(message_id)
        if conv_oid is None or msg_oid is None:
            return None
        doc = await self.messages.find_one_and_update(
            {"_id": msg_oid, "conversation_id": conv_oid},
            {"$set": fields},
            return_document=ReturnDocument.AFTER,
        )
        return Message.from_doc(doc) if doc else None

    async def delete_message(self, conversation_id: str, message_id: str) -> bool:
        conv_oid = _parse_id(conversation_id)
        msg_oid = _parse_id(message_id)
        if conv_oid is None or msg_oid is None:
            return False
        # Scoped by both ids: a delete can never cross a conversation boundary.
        result = await self.messages.delete_one({"_id": msg_oid, "conversation_id": conv_oid})
        return result.deleted_count > 0

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    # --- held messages (see app/peers.py) -------------------------------------

    async def insert_held_message(
        self,
        to_conversation_id: str,
        from_conversation_id: str,
        from_handle: str,
        text: str,
        hops: int,
    ) -> HeldMessage | None:
        to_oid = _parse_id(to_conversation_id)
        if to_oid is None:
            return None
        doc = {
            "to_conversation_id": to_oid,
            "from_conversation_id": from_conversation_id,
            "from_handle": from_handle,
            "text": text,
            "hops": hops,
            "created_at": datetime.now(UTC),
        }
        result = await self.held_messages.insert_one(doc)
        doc["_id"] = result.inserted_id
        return HeldMessage.from_doc(doc)

    async def list_held_messages(self, to_conversation_id: str) -> list[HeldMessage]:
        oid = _parse_id(to_conversation_id)
        if oid is None:
            return []
        cursor = self.held_messages.find({"to_conversation_id": oid}).sort(_HELD_OLDEST_FIRST)
        return [HeldMessage.from_doc(doc) async for doc in cursor]

    async def get_held_message(
        self, to_conversation_id: str, held_id: str
    ) -> HeldMessage | None:
        to_oid = _parse_id(to_conversation_id)
        held_oid = _parse_id(held_id)
        if to_oid is None or held_oid is None:
            return None
        doc = await self.held_messages.find_one(
            {"_id": held_oid, "to_conversation_id": to_oid}
        )
        return HeldMessage.from_doc(doc) if doc else None

    async def delete_held_message(self, to_conversation_id: str, held_id: str) -> bool:
        to_oid = _parse_id(to_conversation_id)
        held_oid = _parse_id(held_id)
        if to_oid is None or held_oid is None:
            return False
        # Scoped by both ids, same rule as delete_message: a deny can never
        # cross a conversation boundary.
        result = await self.held_messages.delete_one(
            {"_id": held_oid, "to_conversation_id": to_oid}
        )
        return result.deleted_count > 0


def _parse_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except InvalidId:
        return None
