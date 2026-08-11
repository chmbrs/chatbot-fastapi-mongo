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
from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from app.models import Conversation, Message, MessageStatus


def create_client(mongo_uri: str) -> AsyncMongoClient:
    return AsyncMongoClient(mongo_uri, tz_aware=True, serverSelectionTimeoutMS=5000)


class Repository:
    def __init__(self, client: AsyncMongoClient, db_name: str):
        self._client = client
        db = client[db_name]
        self.conversations = db["conversations"]
        self.messages = db["messages"]

    async def ensure_indexes(self, retries: int = 10, delay_seconds: float = 1.0) -> None:
        """Idempotent, with a bounded retry: a cold `docker compose up` can
        start this container before Mongo finishes accepting connections
        even though the healthcheck says otherwise."""
        last_error: PyMongoError | None = None
        for _ in range(retries):
            try:
                await self.conversations.create_index([("updated_at", -1)])
                # This prefix also serves conversation_id-only queries (e.g. the
                # cascade delete), so no separate {conversation_id: 1} index.
                await self.messages.create_index([("conversation_id", 1), ("created_at", 1)])
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
        cursor = self.conversations.find().sort("updated_at", -1).limit(limit)
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
        result = await self.conversations.delete_one({"_id": oid})
        return result.deleted_count > 0

    # --- messages ------------------------------------------------------------

    async def list_messages(self, conversation_id: str) -> list[Message]:
        oid = _parse_id(conversation_id)
        if oid is None:
            return []
        cursor = self.messages.find({"conversation_id": oid}).sort("created_at", 1)
        return [Message.from_doc(doc) async for doc in cursor]

    async def insert_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        status: MessageStatus,
        message_id: ObjectId | None = None,
        **fields: object,
    ) -> Message | None:
        """`message_id`, if given, is used as-is — this is what lets chat.py
        mint the assistant message's id before a single token exists, so the
        SSE `start` event can name it."""
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
        if message_id is not None:
            doc["_id"] = message_id
        result = await self.messages.insert_one(doc)
        doc["_id"] = result.inserted_id
        return Message.from_doc(doc)

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


def _parse_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except InvalidId:
        return None
