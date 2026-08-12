import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, computed_field

MessageStatus = Literal["complete", "interrupted", "failed"]
MessageRole = Literal["user", "assistant", "peer"]
InboundPolicy = Literal["accept", "hold", "refuse"]

# Mongo gives back ObjectId, not str — coerce on the way in.
MongoId = Annotated[str, BeforeValidator(str)]

_SLUG_MAX_LENGTH = 24


def derive_handle(title: str, conversation_id: str) -> str:
    """The address a conversation answers to for peer messaging (app/peers.py)
    — derived from its title and id, never stored. That means no uniqueness
    constraint, no index, and no migration for conversations already on disk:
    every existing document already has both inputs. Mirrors how Claude Code
    names a session after its working directory's folder, e.g. `myapp-3f`.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:_SLUG_MAX_LENGTH]
    slug = slug.rstrip("-") or "chat"
    return f"{slug}-{conversation_id[-4:]}"


class MessageError(BaseModel):
    code: str
    message: str
    retry_after_seconds: int | None = None


class PeerInfo(BaseModel):
    """Who a `peer`-role message came from, and how many hops the exchange
    that produced it has already taken. Not a MongoId: this id is stored for
    display, never used to look anything up back through the Repository."""

    from_handle: str
    from_conversation_id: str
    hops: int


class Conversation(BaseModel):
    id: MongoId = Field(validation_alias="_id", serialization_alias="id")
    title: str
    created_at: datetime
    updated_at: datetime
    model: str
    inbound: InboundPolicy = "accept"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def handle(self) -> str:
        return derive_handle(self.title, self.id)

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
    peer: PeerInfo | None = None
    created_at: datetime

    @classmethod
    def from_doc(cls, doc: dict) -> "Message":
        return cls.model_validate(doc)


class Delivery(BaseModel):
    """The three outcomes a peer send can produce (app/peers.py). `held` and
    `refused` are ordinary results of a working system, not failures — so
    they're returned as data here rather than raised as an AppError."""

    outcome: Literal["delivered", "held", "refused"]
    to_handle: str
    to_conversation_id: str
    held_id: str | None = None


class HeldMessage(BaseModel):
    """A message a `hold`-policy conversation has not yet let in. Deliberately
    its own collection rather than a MessageStatus value: a held message was
    never delivered, so it has no place in a transcript, and giving it a
    fourth status would put a non-terminal value in the one field this app's
    whole design is about."""

    id: MongoId = Field(validation_alias="_id", serialization_alias="id")
    to_conversation_id: MongoId
    from_conversation_id: str
    from_handle: str
    text: str
    hops: int
    created_at: datetime

    @classmethod
    def from_doc(cls, doc: dict) -> "HeldMessage":
        return cls.model_validate(doc)
