from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

MessageStatus = Literal["complete", "interrupted", "failed"]
MessageRole = Literal["user", "assistant"]

# Mongo gives back ObjectId, not str — coerce on the way in.
MongoId = Annotated[str, BeforeValidator(str)]


class MessageError(BaseModel):
    code: str
    message: str
    retry_after_seconds: int | None = None


class Conversation(BaseModel):
    id: MongoId = Field(validation_alias="_id", serialization_alias="id")
    title: str
    created_at: datetime
    updated_at: datetime
    model: str

    @classmethod
    def from_doc(cls, doc: dict) -> "Conversation":
        return cls.model_validate(doc)


class Message(BaseModel):
    id: MongoId = Field(validation_alias="_id", serialization_alias="id")
    conversation_id: MongoId
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
        return cls.model_validate(doc)
