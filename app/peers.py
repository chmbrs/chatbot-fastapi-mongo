"""Conversation-to-conversation messaging — a small, playable port of Claude
Code's cross-session messaging
(https://code.claude.com/docs/en/cross-session-messaging) onto this app's
own conversations.

Imports neither `fastapi` nor `pymongo`, for the same reason app/chat.py
doesn't: that boundary is what makes this testable with a fake LLM and no
HTTP layer.

Kept from the real feature, because each is cheap and characteristic here: a
handle every conversation answers to, an accept/hold/refuse inbound policy,
plain text only, and a reply that forwards back to the sender so an exchange
visibly runs somewhere rather than disappearing into one turn. Deliberately
not ported: TTL/dialogExpiry, permission-class inference, cross-machine
delivery, queue caps, per-sender rate limiting — see README.md for why.
"""

from app.chat import ChatService, drain
from app.errors import AppError, CannotMessageSelf, PeerNotFound
from app.models import Delivery
from app.repository import Repository


async def resolve_handle(repository: Repository, handle: str):
    """The roster *is* the sidebar: this scans the same list_conversations()
    the UI does, so a conversation that has scrolled off it is also not
    addressable. That's a property of the handle being derived
    (models.derive_handle) rather than stored, not a limitation worked
    around here."""
    for conversation in await repository.list_conversations():
        if conversation.handle == handle:
            return conversation
    return None


async def deliver(
    repository: Repository,
    *,
    to_handle: str,
    text: str,
    from_conversation_id: str,
    from_handle: str,
    hops: int,
) -> Delivery:
    """Looks up the addressee and applies its inbound policy. Never runs a
    turn itself — that's run_exchange's job, called only when this returns
    `delivered` — so a caller that only wants the outcome (e.g. re-checking
    before an inbox Approve) never pays for a turn it didn't ask for."""
    to_conversation = await resolve_handle(repository, to_handle)
    if to_conversation is None:
        raise PeerNotFound(to_handle)
    if to_conversation.id == from_conversation_id:
        raise CannotMessageSelf()

    if to_conversation.inbound == "refuse":
        return Delivery(
            outcome="refused", to_handle=to_handle, to_conversation_id=to_conversation.id
        )

    if to_conversation.inbound == "hold":
        held = await repository.insert_held_message(
            to_conversation.id, from_conversation_id, from_handle, text, hops
        )
        return Delivery(
            outcome="held",
            to_handle=to_handle,
            to_conversation_id=to_conversation.id,
            held_id=held.id,
        )

    return Delivery(
        outcome="delivered", to_handle=to_handle, to_conversation_id=to_conversation.id
    )


async def run_exchange(
    repository: Repository,
    chat_service: ChatService,
    *,
    to_conversation_id: str,
    text: str,
    from_conversation_id: str,
    from_handle: str,
    hops: int,
    hop_limit: int,
) -> None:
    """Runs the receiving turn and, if it produced a real reply, forwards
    that reply back — a bounded while loop, never recursion, so the stack
    doesn't grow with the exchange. Each leg swaps sender and receiver, so
    this is what makes two conversations visibly volley for a few turns and
    then stop: the loop condition, not a special case for "the reply".

    This oscillates strictly between the two original participants — it
    never routes to a third conversation on its own, because nothing here
    reads a reply's text looking for another handle to forward to. A longer
    relay across more than two conversations can only happen through a
    separate, human-initiated send for each additional hop, and each of
    those starts its own independent hop count from zero. So there is no
    path to an unsupervised, unbounded loop at all — not just a bounded one
    — and no identical-repeat dedupe is needed to stop one.

    Never raises. A failed peer turn is already persisted `failed` by
    ChatService._generate; there is no HTTP caller here to raise to, so an
    AppError anywhere in the chain just ends the exchange where it is.
    """
    to_id, sender_id, sender_handle, body, current_hops = (
        to_conversation_id,
        from_conversation_id,
        from_handle,
        text,
        hops,
    )

    while True:
        try:
            turn = await chat_service.run_peer_turn(
                to_id,
                body,
                from_handle=sender_handle,
                from_conversation_id=sender_id,
                hops=current_hops,
            )
            reply = await drain(turn)
        except AppError:
            return

        # Interrupted, failed, or an empty reply: nothing worth forwarding,
        # and forwarding an empty message would spend a hop on nothing.
        if reply is None or reply.status != "complete" or not reply.content:
            return

        current_hops += 1
        if current_hops >= hop_limit:
            return

        replier = await repository.get_conversation(to_id)
        if replier is None:
            return  # deleted mid-exchange

        try:
            forwarded = await deliver(
                repository,
                to_handle=sender_handle,
                text=reply.content,
                from_conversation_id=to_id,
                from_handle=replier.handle,
                hops=current_hops,
            )
        except AppError:
            return

        if forwarded.outcome != "delivered":
            return  # held or refused on the way back: a human breaks the loop here

        to_id, sender_id, sender_handle, body = sender_id, to_id, replier.handle, reply.content
