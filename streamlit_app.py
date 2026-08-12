"""Streamlit UI for chatbot-fastapi-mongo.

A pure HTTP client of the FastAPI backend: no import of `app/`, no shared
code, no database access. That boundary is what makes this swap possible
without touching the turn lifecycle at all: whatever this script does or
doesn't do on a disconnect, the backend already committed the assistant row
as `interrupted` before the first token (see app/chat.py) and only ever
improves on it from there. Talking over HTTP rather than in-process also
means no CORS change was needed anywhere: the browser only ever talks to
*this* server; the call to the API is server-to-server.

Streaming uses st.write_stream, which consumes a generator of text chunks.
stream_turn() below is that generator: it opens the same SSE endpoint the
old vanilla-JS frontend used and parses the same wire format, so the two
UIs are drawing on identical backend behavior, not a reimplementation of it.
"""

import json
import os
import re
from collections.abc import Iterator

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
# No provider-agnostic way to know the right timeout from here, so this
# matches the Ollama worst case in app/llm/__init__.py: a hosted-provider
# timeout would abort a local model that's still loading into RAM.
_STREAM_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

st.set_page_config(page_title="chatbot-fastapi-mongo", page_icon="💬")

# The one deliberate use of unsafe_allow_html in this file, and it is not the
# boundary this file otherwise cares about: nothing here renders user- or
# model-supplied content (that boundary is st.markdown(message["content"]),
# and it stays exactly as unsafe_allow_html=False as ever). This is a static
# layout fix with zero interpolated data, targeting one real, measured bug a
# widget parameter cannot reach: a popover trigger with no icon or label of
# its own -- the sidebar's ⌄-only "more actions" menu -- still renders
# Streamlit's built-in chevron in a second flex slot sitting next to an empty
# first one, and flex centering never lands it dead center.
#
# Measured with getBoundingClientRect in the browser, in order: zeroing the
# gap between the two slots only closed part of the offset (an empty slot
# still reserves width via its own span); hiding that empty slot outright
# (:has(>span:empty), so a popover that *does* have an icon -- the header's
# 🛰 send button -- is untouched) got the chevron down to one flex child, but
# the button's own content box (padding: 4px 12px, border-box, minus a 1px
# border) comes out *narrower than the 16px icon itself* -- so any flex
# centering here is centering an overflowing box, which browsers don't
# split evenly. `width: 100%` on the wrapper just inherited that same
# too-small box instead of fixing it.
#
# Absolute positioning sidesteps the whole problem: centering by
# `top/left: 50%` plus `translate(-50%, -50%)` is computed from the element's
# own size against its containing block, never from how flex distributes
# space among siblings, so it can't inherit an overflow asymmetry that isn't
# there in the first place. `position: relative` on the button gives that
# containing block.
#
# `stPopoverButton` is a testid Streamlit assigns intentionally, not one of
# its generated (and unstable across versions) emotion-hash classes, which is
# why the selectors below are keyed off that rather than a class name --
# but that same emotion CSS outranks a same-specificity, later-in-source
# override on a couple of these properties regardless, which is what the
# `!important`s are for: not a style choice, the actual fix for that.
st.markdown(
    "<style>"
    "button[data-testid='stPopoverButton'] { position: relative !important; }"
    "button[data-testid='stPopoverButton'] > div > div:has(> span:empty) "
    "{ display: none !important; }"
    "button[data-testid='stPopoverButton'] > div > div[aria-hidden='true'] {"
    "  position: absolute !important;"
    "  top: 50% !important;"
    "  left: 50% !important;"
    # Horizontal came out exact (measured 5px/5px either side); vertical
    # didn't (14px/10px) -- the icon glyph's own box isn't symmetric around
    # its visual center, a font-metric quirk translate(-50%,-50%) alone
    # doesn't know about. The extra -2px is that measured gap, halved,
    # applied once: not a value that needs to react to anything at runtime.
    "  transform: translate(-50%, calc(-50% - 2px)) !important;"
    "}"
    "</style>",
    unsafe_allow_html=True,
)


class TurnFailed(Exception):
    """One exception for both the SSE `error` event and a plain HTTP error,
    mirroring the single error envelope both backend renderers share."""


def api(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", timeout=30.0, **kwargs)
    except httpx.HTTPError as exc:
        # Connection refused, DNS failure, timeout: the api container isn't
        # up yet, or isn't reachable. Caught here, not just HTTP status
        # codes, so this never falls through to Streamlit's raw traceback UI.
        raise TurnFailed(f"Could not reach the API at {API_BASE_URL}: {exc}") from exc
    if response.status_code >= 400:
        raise TurnFailed(_error_message(response))
    return response.json() if response.content else None


def _error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return f"Request failed (HTTP {response.status_code})"


def _parse_sse(response: httpx.Response) -> Iterator[tuple[str, dict]]:
    """Same block-based parsing as the old frontend's parseEventBlock: lines
    accumulate until a blank line ends the event; `:`-prefixed lines are
    sse-starlette's keep-alive comments (sent while a slow provider is still
    loading) and are ignored rather than mistaken for a hang."""
    event, data_lines = "message", []
    for raw_line in response.iter_lines():
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield event, json.loads("".join(data_lines))
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        yield event, json.loads("".join(data_lines))


def stream_turn(path: str, json_body: dict | None = None) -> Iterator[str]:
    """Yields text deltas for st.write_stream. Raises TurnFailed on the SSE
    `error` event, or on a plain HTTP error status: the two failure shapes
    the backend's dual renderer can produce (see app/routes.py): a setup
    failure like an unknown conversation raises before any SSE frame is
    sent, while a provider failure mid-turn arrives as an `error` frame after
    the 200 is already committed.
    """
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}{path}",
            json=json_body if json_body is not None else {},
            headers={"accept": "text/event-stream"},
            timeout=_STREAM_TIMEOUT,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise TurnFailed(_error_message(response))
            for event, data in _parse_sse(response):
                if event == "delta":
                    yield data["text"]
                elif event == "error":
                    raise TurnFailed(data["message"])
    except httpx.HTTPError as exc:
        # Same as api() above: a dropped connection mid-stream (or one that
        # never connected) is a TurnFailed the caller already knows how to
        # show, not an unhandled exception.
        raise TurnFailed(f"Could not reach the API at {API_BASE_URL}: {exc}") from exc


def _with_thinking(slot, chunks: Iterator[str]) -> Iterator[str]:
    """The same "Generating…" the transcript shows for a turn someone else
    started, for the one case the transcript can't cover: this tab's own
    in-flight turn, which isn't in the transcript yet and shows nothing at all
    until the first token arrives. That gap is a whole model load with Ollama,
    and an empty bubble is the one thing it shouldn't look like.

    A wrapper around the chunk generator rather than anything st.write_stream
    knows about: the first chunk is the only signal that the wait is over, and
    it passes through here on its way to being rendered.
    """
    thinking = True
    for chunk in chunks:
        if thinking:
            slot.empty()
            thinking = False
        yield chunk


def _sidebar_label(title: str, limit: int = 28) -> str:
    """The backend's own title (up to 60 chars, see app/chat.py's
    _derive_title) is right for the header, where there's room, but wraps a
    sidebar row onto two or three lines, and a list where every row is a
    different height doesn't read as considered. This applies the same
    word-boundary truncation the backend already uses, just at the shorter
    length that actually fits one line, so every row in the list stands the
    same height. The full title is never lost; only this one label is
    shortened, and the header still shows it in full.

    28, not some rounder number, because it was measured, not guessed: with
    the sidebar's actual button padding and font, a handful of realistic
    titles (the backend's own 3-6-word style, and a few much longer ones)
    all stayed within the row's available label width at 28 and several
    started overflowing to a second line at 29. Rounding down from a
    measurement beats rounding up from a guess.
    """
    if len(title) <= limit:
        return title
    truncated = title[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "…"


def _addressable(conversations: list[dict], self_id: str | None) -> list[dict]:
    """Everything the current conversation can send to: the roster minus
    itself. Mirrors app/peers.py's resolve_handle, which scans the same
    list_conversations() this UI renders, so what looks addressable here is
    exactly what is addressable there."""
    return [c for c in conversations if c["id"] != self_id]


def _resolve_mention(token: str, conversations: list[dict], self_id: str | None):
    """Turns what someone actually typed after `@` into a full handle, or
    into a sentence explaining why it didn't resolve. Returns
    (handle, error) with exactly one of the two set.

    A handle is `slug(title)[:24]-<last 4 of the id>` (app/models.py), which
    nobody types from memory, so an exact match can't be the only thing that
    works. Prefix and then substring matching let `@second` reach
    `second-chat-3f9a` while the request sent to the backend is still the
    canonical handle: the resolution is a UI affordance, and /send stays
    strict about what it accepts.

    Ambiguity is an error, never a guess. Two chats titled alike differ only
    in the id suffix, and silently picking one of them would be the same
    failure mode this whole function exists to remove.
    """
    others = _addressable(conversations, self_id)
    if not others:
        return None, "There's no other conversation to message yet. Start a second chat first."

    needle = token.lower().lstrip("@")
    if not needle:
        return None, "Name a conversation after the @, for example @" + others[0]["handle"] + "."

    for candidate in others:
        if candidate["handle"] == needle:
            return candidate["handle"], None

    prefix = [c for c in others if c["handle"].startswith(needle)]
    matches = prefix or [c for c in others if needle in c["handle"] or needle in c["title"].lower()]

    if len(matches) == 1:
        return matches[0]["handle"], None
    if len(matches) > 1:
        listed = ", ".join(f"@{c['handle']}" for c in matches)
        return None, f"@{needle} matches more than one conversation: {listed}. Be more specific."

    # Self-addressing is worth naming on its own: it is a different mistake
    # from a typo, and the backend rejects it too (CannotMessageSelf).
    if self_id is not None:
        current = next((c for c in conversations if c["id"] == self_id), None)
        if current and (current["handle"].startswith(needle) or needle in current["title"].lower()):
            return None, "That's this conversation. Pick a different one to message."

    listed = ", ".join(f"@{c['handle']}" for c in others[:5])
    return None, f"No conversation matches @{needle}. You can reach: {listed}"


def _composer_hint(conversations: list[dict], self_id: str | None) -> str:
    """The placeholder is the only teaching surface the composer has, but it
    doesn't name a real, current handle any more (it used to show
    others[0]) -- a title-derived slug is only ever meant to be read once
    you already know which conversation you're looking at, so surfacing an
    arbitrary *other* one's here, out of context, just reads as noise (worse
    when that title is a sentence fragment, e.g. "something-is-coming-your-
    7f60"). The literal word "handle" was tried before that and read as an
    instruction to type "handle" -- this sidesteps both failure modes by not
    naming anything: the conversation you actually want is one click away
    (open it, its own handle is right under its title).

    "or" would describe the two @-mention forms as mutually exclusive with
    messaging this chat, which is only true of the explicit `@handle text`
    prefix (a send with no reply from this chat). A mention anywhere else in
    an ordinary message still replies here *and* forwards that reply, so the
    hint says "too" rather than presenting an either/or that no longer holds
    for that second form."""
    others = _addressable(conversations, self_id)
    if not others:
        return "Message this chat"
    return "Message this chat, or @mention another conversation to send there too"


def _set_conversation(conversation_id: str | None) -> None:
    st.session_state.conversation_id = conversation_id
    st.session_state.peer_notice = None
    st.session_state.peer_error = None
    if conversation_id is None:
        st.query_params.clear()
    else:
        st.query_params["c"] = conversation_id


def _delete_conversation(conversation_id: str) -> None:
    api("DELETE", f"/api/conversations/{conversation_id}")
    if st.session_state.conversation_id == conversation_id:
        _set_conversation(None)


@st.dialog("Rename conversation")
def _rename_dialog(conversation_id: str, current_title: str) -> None:
    # Takes the id explicitly rather than closing over the selected
    # conversation: this is opened from the sidebar's per-row menu too, for
    # rows that are not the open conversation, and a version that assumed
    # "the dialog's target is whatever's selected" would rename the wrong one.
    new_title = st.text_input("Title", value=current_title)
    if st.button("Save", icon=":material/check:", type="primary") and new_title.strip():
        # Every other write in this file goes through the same TurnFailed ->
        # st.error path (see api()'s docstring); this dialog was the one
        # place that let a plain 404 -- e.g. the conversation was deleted
        # elsewhere between opening this dialog and saving it -- fall through
        # to Streamlit's raw traceback box instead.
        try:
            api("PATCH", f"/api/conversations/{conversation_id}", json={"title": new_title})
        except TurnFailed as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _set_llm_provider() -> None:
    # on_change: Streamlit already wrote the new value to session_state under
    # this key before calling back, so it's read here rather than passed in.
    provider = "ollama" if st.session_state["use-ollama"] else "demo"
    api("PUT", "/api/settings/llm-provider", json={"provider": provider})


def _request_retry(conversation_id: str) -> None:
    # on_click callbacks run before the script body and can't render
    # anything (they execute as a prefix to the rerun, per Streamlit's
    # session_state docs). This only records intent; the main body below
    # does the actual streaming.
    st.session_state.pending_retry = conversation_id


def _send_peer_message(from_conversation_id: str, to_handle: str, text: str) -> dict:
    return api(
        "POST",
        f"/api/conversations/{from_conversation_id}/send",
        json={"to_handle": to_handle, "text": text},
    )


def _set_inbound_policy(conversation_id: str, policy: str) -> None:
    api("PUT", f"/api/conversations/{conversation_id}/inbound", json={"policy": policy})


def _approve_held_message(conversation_id: str, held_id: str) -> None:
    api("POST", f"/api/conversations/{conversation_id}/inbox/{held_id}/approve")


def _deny_held_message(conversation_id: str, held_id: str) -> None:
    api("DELETE", f"/api/conversations/{conversation_id}/inbox/{held_id}")


def _peer_popover(conversation: dict, conversations: list[dict]) -> None:
    """The whole peer-messaging control surface, in one popover so it
    doesn't compete for room in a header that has none to spare (see
    _sidebar_label's docstring on why: this file has no layout="wide")."""
    policy = st.segmented_control(
        "When another chat messages this one",
        options=["accept", "hold", "refuse"],
        # The stored values are the backend's contract (app/models.py's
        # InboundPolicy); only the labels are softened.
        format_func={"accept": "Reply", "hold": "Ask me", "refuse": "Block"}.get,
        default=conversation["inbound"],
        key=f"inbound-{conversation['id']}",
    )
    if policy and policy != conversation["inbound"]:
        _set_inbound_policy(conversation["id"], policy)
        st.rerun()

    st.divider()
    others = _addressable(conversations, conversation["id"])
    if not others:
        st.caption("Create a second conversation to message it.")
    else:
        # Chosen by title, sent by handle: nobody recognizes a chat by its
        # slug-plus-id-suffix, but that is what /send needs.
        by_handle = {c["handle"]: c for c in others}
        to_handle = st.selectbox(
            "Send to",
            options=list(by_handle),
            format_func=lambda h: f"{_sidebar_label(by_handle[h]['title'])}  ·  @{h}",
            key=f"peer-to-{conversation['id']}",
        )
        text = st.text_area("Message", key=f"peer-text-{conversation['id']}")
        if st.button("Send", icon=":material/send:", key=f"peer-send-{conversation['id']}"):
            if text.strip():
                try:
                    delivery = _send_peer_message(conversation["id"], to_handle, text.strip())
                    st.toast(f"{delivery['outcome']} → @{to_handle}", icon="🛰")
                except TurnFailed as exc:
                    st.error(str(exc))

    inbox = api("GET", f"/api/conversations/{conversation['id']}/inbox")
    if inbox:
        st.divider()
        st.caption("Held messages")
        for held in inbox:
            st.text(f"@{held['from_handle']}: {held['text']}")
            approve_col, deny_col = st.columns(2)
            if approve_col.button("Approve", key=f"approve-{held['id']}", width="stretch"):
                _approve_held_message(conversation["id"], held["id"])
                st.rerun()
            if deny_col.button("Deny", key=f"deny-{held['id']}", width="stretch"):
                _deny_held_message(conversation["id"], held["id"])
                st.rerun()


@st.fragment(run_every="2s")
def _watch_for_peer_activity(conversation_id: str) -> None:
    """The receiving half of a peer exchange runs server-side (see
    app/peers.py): nothing streams it to this tab, so this is what notices a
    reply landed and repaints. Never repaints over an in-flight
    st.write_stream: that block sets st.session_state.streaming for exactly
    its own duration, and this returns early while it's set."""
    if st.session_state.streaming:
        return
    try:
        messages = api("GET", f"/api/conversations/{conversation_id}/messages")
    except TurnFailed:
        return  # a poll that fails is not an error the user caused
    # Every row, not just the last one, and the whole GET was already being
    # paid for either way. A hop-limited exchange can have an earlier reply
    # finish while a later one is still being written, and watching only the
    # tail left the finished one drawn as "Generating…" until the tail
    # happened to change too.
    #
    # `live` sits in here next to status because the two are one fact: a turn
    # that stops halfway settles from live-and-`interrupted` to plain
    # `interrupted`, which is a status that did not change and a meaning that
    # did.
    seen = tuple((m["id"], m["status"], m["live"]) for m in messages)
    # Record what was seen *before* rerunning, not after: st.rerun() raises
    # immediately, so an assignment placed after it never runs, last_seen
    # stays stale, and the next pass sees the same change and reruns again --
    # a hot loop that repaints forever. Worse, this fragment is reached
    # before st.chat_input below, so every one of those aborted passes skips
    # the composer: the symptom is not a busy app but an app that silently
    # swallows everything typed into it once a peer reply lands.
    previous, st.session_state.last_seen = st.session_state.last_seen, seen
    if previous not in (None, seen):
        st.rerun()  # app scope: the sidebar's ordering can have changed too


if "conversation_id" not in st.session_state:
    # Resume across a reload the same way the old localStorage did, via the
    # one thing that actually survives a reload in Streamlit's model: the URL.
    st.session_state.conversation_id = st.query_params.get("c")
st.session_state.setdefault("pending_retry", None)
st.session_state.setdefault("streaming", False)
st.session_state.setdefault("last_seen", None)
st.session_state.setdefault("peer_notice", None)
st.session_state.setdefault("peer_error", None)

try:
    health = api("GET", "/api/health")
    conversations = api("GET", "/api/conversations")
except TurnFailed as exc:
    # str(exc) is already a complete sentence: api() names the cause,
    # whether that's an HTTP error or a connection that never happened.
    st.error(str(exc))
    st.stop()

# A conversation named by a stale URL or deleted from another tab: the same
# authoritative-server-state check the old frontend did before trusting
# localStorage.
if st.session_state.conversation_id not in {c["id"] for c in conversations}:
    _set_conversation(None)

conversation_id = st.session_state.conversation_id


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    # Only when no key is set: a real key already answers "which provider",
    # and this app never silently overrides one (see /api/health's
    # has_api_key and the guard in set_llm_provider, app/routes.py).
    if not health["llm"]["has_api_key"]:
        st.toggle(
            "Use local Ollama model",
            value=health["llm"]["provider"] == "ollama",
            key="use-ollama",
            on_change=_set_llm_provider,
            help="No LLM_API_KEY is configured, so this app is running either "
            "the offline demo provider or a local Ollama model. Off is the "
            "demo, on is Ollama, no restart needed either way.",
        )

    if health["status"] != "ok":
        st.warning(health["llm"]["degraded_reason"])

    # A fixed-height container rather than a plain button: st.divider()'s own
    # margin used to leave a stretch of dead space between the button and the
    # list below it. Giving that height to the button's own container -- and
    # dropping the divider that was creating it -- turns that space into part
    # of the button instead of padding next to it.
    with st.container(height=80, border=False, vertical_alignment="center"):
        st.button(
            "New chat",
            icon=":material/add:",
            on_click=_set_conversation,
            args=(None,),
            width="stretch",
        )

    for conversation in conversations:
        # A single flat row per conversation, closer to a chat client's own
        # list: the title button's rounded pill is the only surface -- no
        # outer card stacked around it -- with the ⋮ menu beside it instead
        # of a third icon of equal weight competing with the title.
        #
        # The handle used to live here too, always visible, then moved to a
        # hover tooltip on the title button, then off the row entirely -- it
        # only lives in the ⋮ menu now. Nothing that needed it lost it: the
        # peer popover's "Send to" list already spells out every reachable
        # handle at the one place this app actually asks anyone to type one.
        # gap=0: the closest native approximation of "one unified row" without
        # the custom CSS this file otherwise avoids (see _peer_popover's note
        # on st.code's built-in copy icon for the same reasoning) -- the title
        # and the ⋮ trigger are still two separate widgets, but they now sit
        # flush against each other instead of visibly separated.
        row = st.columns([5, 1], vertical_alignment="center", gap=0)
        row[0].button(
            _sidebar_label(conversation["title"]),
            icon=":material/chat_bubble:",
            key=f"select-{conversation['id']}",
            type="primary" if conversation["id"] == conversation_id else "secondary",
            width="stretch",
            on_click=_set_conversation,
            args=(conversation["id"],),
        )
        # st.popover, not a session_state toggle: a toggle's opened content is
        # just more elements in the sidebar's own vertical flow, so opening
        # one row's menu pushed every row below it down the list -- worse
        # than the thing it was trying to fix. A popover floats above the
        # list instead and costs nothing else on the page while closed.
        #
        # No icon on the trigger: a popover always draws its own trailing
        # chevron regardless (baked into the widget, no parameter turns it
        # off), so leaving `icon` unset renders that chevron alone instead of
        # stacking a kebab icon in front of it.
        #
        # `key=` still matters for the reason it always did in a loop like
        # this: every row's trigger has byte-identical visible args (empty
        # label, no icon), so without an id-derived key Streamlit can only
        # tell rows apart by position, and a list reorder (a delete, a
        # resort) would hand row N's popover identity to whatever
        # conversation shifted into that slot next.
        #
        # width="stretch": a popover defaults to sizing itself to its content
        # ("content" is st.popover's own default, unlike st.button elsewhere
        # in this file), so with no icon or label left to size around, it
        # rendered narrower than its column and sat short of the column's own
        # right edge -- the row stopped short of where "New chat" reaches,
        # instead of the two lining up.
        with row[1].popover("", key=f"more-{conversation['id']}", width="stretch"):
            st.caption(f"@{conversation['handle']}")
            st.divider()
            if st.button(
                "Rename",
                icon=":material/edit:",
                key=f"rename-{conversation['id']}",
                width="stretch",
            ):
                _rename_dialog(conversation["id"], conversation["title"])
            st.button(
                "Delete conversation",
                icon=":material/delete:",
                key=f"delete-{conversation['id']}",
                on_click=_delete_conversation,
                args=(conversation["id"],),
                width="stretch",
            )


# --- main pane ---------------------------------------------------------------

# Both are read by the composer block at the bottom, which also runs on the
# pass where no conversation is selected yet and nothing below was rendered.
messages: list[dict] = []
live_turn = None

if conversation_id is None:
    st.info("Send a message to start a new conversation.")
else:
    conversation = api("GET", f"/api/conversations/{conversation_id}")
    # The header shows the title in full: this is the one place with room
    # for it; the sidebar list gets the shortened version (_sidebar_label).
    header = st.columns([7, 1, 1], vertical_alignment="center")
    header[0].subheader(conversation["title"])
    st.caption(f"@{conversation['handle']}")

    # on_change="rerun" (lazy content, computed only while open) was tried
    # first and dropped: it doubled the header row's buttons in the DOM on
    # the very first script pass, before the popover was ever opened. The
    # default (content computed every rerun, popover or not) costs one
    # extra GET .../inbox per rerun -- cheap, and correct.
    with header[1].popover("", icon=":material/send:", help="Message another conversation"):
        _peer_popover(conversation, conversations)

    if header[2].button("", icon=":material/edit:", help="Rename conversation"):
        _rename_dialog(conversation_id, conversation["title"])

    # A fixed height keeps the header, the 🛰 popover and the handle above
    # always visible and always reachable, with only the transcript itself
    # scrolling. st.chat_message children get autoscroll-to-latest for
    # free (see st.container's docs) -- which matters more here than in a
    # single-user chat, since a peer exchange can add several rows in the
    # time it takes to glance away and back.
    #
    # border=False: a fixed-height container draws a box by default, and that
    # box read as a divider cutting the conversation in two -- especially
    # while a turn was streaming, since the live turn used to render below it
    # (see live_turn). The scroll behavior is what's wanted here, not the
    # frame around it.
    with st.container(height=450, border=False):
        messages = api("GET", f"/api/conversations/{conversation_id}/messages")

        # A send to a peer is recorded in the *receiving* conversation only
        # (app/routes.py's send_peer_message), so without this the sender's
        # own transcript never shows what it sent, and the peer's reply --
        # which forwards back here -- arrives with no visible question above
        # it. Session state, not the database: no schema change, and it stays
        # honest about being a client-side trace by not surviving a reload.
        notice = st.session_state.peer_notice
        if notice and notice["conversation_id"] != conversation_id:
            notice = None

        def _render_notice() -> None:
            with st.chat_message("user", avatar="🛰"):
                st.caption(f"sent to @{notice['to_handle']} · {notice['outcome']}")
                st.markdown(notice["text"])

        # Anchored to whatever the last message was when the send happened, so
        # it sits above the reply it prompted instead of below it. Appending
        # never shifts an earlier message's id, so the anchor stays put as the
        # exchange volleys.
        if notice and notice["after_message_id"] is None:
            _render_notice()

        for index, message in enumerate(messages):
            avatar = "🛰" if message["role"] == "peer" else None
            with st.chat_message(message["role"], avatar=avatar):
                if message["role"] == "peer":
                    st.caption(f"from @{message['peer']['from_handle']}")
                # Streamlit's markdown renderer has unsafe_allow_html off by
                # default, so this holds the same XSS boundary the old
                # textContent-only frontend documented as a deliberate choice,
                # just with markdown formatting now rendering, which that one
                # didn't have.
                st.markdown(message["content"] or ("…" if message["status"] != "failed" else ""))
                if message["live"]:
                    # Being written right now (see list_messages in
                    # app/routes.py). The row underneath says `interrupted`
                    # and will keep saying it until the last token lands, so
                    # without this flag a reply nobody in this tab started --
                    # the whole receiving half of an @-send, which runs
                    # server-side -- rendered as a dead turn, under a Retry
                    # button, while the model was still typing it.
                    st.caption("Generating…")
                elif message["role"] == "assistant" and message["status"] != "complete":
                    st.caption(
                        message["error"]["message"] if message["error"] else "(interrupted)"
                    )
                    # Only the last message: /retry always regenerates the
                    # trailing reply, so the button would be misleading
                    # anywhere else in the transcript.
                    if index == len(messages) - 1:
                        st.button(
                            "Retry",
                            icon=":material/refresh:",
                            key=f"retry-{message['id']}",
                            on_click=_request_retry,
                            args=(conversation_id,),
                        )

            if notice and notice["after_message_id"] == message["id"]:
                _render_notice()
                notice = None  # placed; don't fall through to the tail below

        # The anchor is gone (the message it named was deleted) or the send
        # predates nothing in this list: showing it last beats dropping it.
        if notice and notice["after_message_id"] is not None:
            _render_notice()

        if st.session_state.pending_retry == conversation_id:
            st.session_state.pending_retry = None
            st.session_state.streaming = True
            try:
                with st.chat_message("assistant"):
                    thinking = st.empty()
                    thinking.caption("Generating…")
                    st.write_stream(
                        _with_thinking(
                            thinking, stream_turn(f"/api/conversations/{conversation_id}/retry")
                        )
                    )
            except TurnFailed as exc:
                st.error(str(exc))
            finally:
                st.session_state.streaming = False
            st.rerun()

        # Filled in below, after st.chat_input has been read. Reserving the
        # slot here is what keeps a streaming turn inside the transcript
        # instead of below it: the composer has to be read after the
        # transcript renders, but its output belongs above it.
        live_turn = st.empty()

    # Directly above the composer, so a correction sits next to the box you
    # retype in -- and, critically, *before* the polling fragment below,
    # which calls st.rerun() the moment a peer reply lands and so ends the
    # script pass early. Anything written after that call is not reliably
    # reached on the pass that should have cleared it, which is exactly how a
    # corrected send ended up still showing the previous send's complaint.
    #
    # An always-created st.empty() rather than a bare `if`: leaving a slot
    # empty is what clears it; simply not drawing an element leaves the last
    # one in place.
    error_slot = st.empty()
    if st.session_state.peer_error:
        error_slot.warning(st.session_state.peer_error, icon="🛰")

    _watch_for_peer_activity(conversation_id)

prompt = st.chat_input(_composer_hint(conversations, conversation_id))
if prompt:
    target_id = conversation_id
    if target_id is None:
        target_id = api("POST", "/api/conversations", json={})["id"]
        _set_conversation(target_id)
        live_turn = None  # brand-new conversation: no transcript rendered yet

    # Anything starting with @ is a send attempt and is never quietly
    # re-routed to the LLM. Falling through on a typo was the single biggest
    # reason this feature looked broken: `@typo hello` came back as a chatbot
    # answer about handles, with nothing to say a delivery had failed.
    if prompt.startswith("@"):
        token, _, text = prompt.partition(" ")
        to_handle, error = _resolve_mention(token, conversations, target_id)
        if error:
            st.session_state.peer_error = error
        elif not text.strip():
            st.session_state.peer_error = f"Add a message after @{to_handle}, then send."
        else:
            st.session_state.peer_error = None
            try:
                delivery = _send_peer_message(target_id, to_handle, text.strip())
                st.session_state.peer_notice = {
                    "conversation_id": target_id,
                    "to_handle": to_handle,
                    "text": text.strip(),
                    "outcome": delivery["outcome"],
                    # `messages` is the transcript as it stood a moment ago,
                    # before this send could have provoked a reply into it.
                    "after_message_id": messages[-1]["id"] if messages else None,
                }
                st.toast(f"{delivery['outcome']} → @{to_handle}", icon="🛰")
            except TurnFailed as exc:
                st.session_state.peer_error = str(exc)
    else:
        # A mention anywhere in an ordinary message — as opposed to the
        # explicit "@handle text" send above, which is never a chat turn —
        # doesn't send by itself. It marks this turn's own reply, once
        # generated, for forwarding: "generate a number, then send it to
        # @x" works as one natural sentence instead of two turns.
        mention = re.search(r"@[\w-]+", prompt)
        forward_handle, forward_error = (
            _resolve_mention(mention.group(0), conversations, target_id)
            if mention
            else (None, None)
        )
        st.session_state.peer_error = forward_error
        st.session_state.peer_notice = None
        target = live_turn.container() if live_turn is not None else st.container()

        with target:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.streaming = True
            reply_text = None
            try:
                with st.chat_message("assistant"):
                    thinking = st.empty()
                    thinking.caption("Generating…")
                    reply_text = st.write_stream(
                        _with_thinking(
                            thinking,
                            stream_turn(
                                f"/api/conversations/{target_id}/messages", {"content": prompt}
                            ),
                        )
                    )
            except TurnFailed as exc:
                st.error(str(exc))
            finally:
                st.session_state.streaming = False

            if forward_handle and reply_text and reply_text.strip():
                # Unlike the explicit "@handle text" send above, this turn
                # already rendered a real, persisted user message and assistant
                # reply above — setting peer_notice here would draw a second,
                # fake "sent to @X" bubble repeating the reply already on screen.
                try:
                    delivery = _send_peer_message(target_id, forward_handle, reply_text.strip())
                    st.toast(f"{delivery['outcome']} → @{forward_handle}", icon="🛰")
                except TurnFailed as exc:
                    st.session_state.peer_error = str(exc)
    st.rerun()
