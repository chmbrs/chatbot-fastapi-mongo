from datetime import datetime
from typing import Literal

from pydantic import BaseModel

MessageStatus = Literal["complete", "interrupted", "failed"]
MessageRole = Literal["user", "assistant"]


class MessageError(BaseModel):
    code: str
    message: str
    retry_after_seconds: int | None = None


def _without_mongo_id(doc: dict) -> dict:
    """No `Field(alias="_id")`: FastAPI's response_model serialization
    defaults to by_alias=True, which would otherwise leak `_id` straight
    into JSON responses instead of the clean `id` field these models
    declare — so the rename happens here, explicitly, once."""
    return {**{k: v for k, v in doc.items() if k != "_id"}, "id": str(doc["_id"])}


class Conversation(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model: str

    @classmethod
    def from_doc(cls, doc: dict) -> "Conversation":
        return cls.model_validate(_without_mongo_id(doc))


class Message(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    status: MessageStatus
    error: MessageError | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict | None = None
    ttft_ms: int | None = None
    total_ms: int | None = None
    created_at: datetime

    @classmethod
    def from_doc(cls, doc: dict) -> "Message":
        data = _without_mongo_id(doc)
        data["conversation_id"] = str(data["conversation_id"])
        return cls.model_validate(data)
