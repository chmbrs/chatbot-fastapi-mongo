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
from collections.abc import Iterator

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
# No provider-agnostic way to know the right timeout from here, so this
# matches the Ollama worst case in app/llm/__init__.py: a hosted-provider
# timeout would abort a local model that's still loading into RAM.
_STREAM_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

st.set_page_config(page_title="chatbot-fastapi-mongo", page_icon="💬")


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


def _sidebar_label(title: str, limit: int = 28) -> str:
    """The backend's own title (up to 60 chars, see app/chat.py's
    _derive_title) is right for the header, where there's room, but wraps a
    sidebar row onto two or three lines, and a list where every row is a
    different height doesn't read as considered. This applies the same
    word-boundary truncation the backend already uses, just at the shorter
    length that actually fits one line, so every row in the list stands the
    same height. The full title is never lost; only this one label is
    shortened, and the header still shows it in full.
    """
    if len(title) <= limit:
        return title
    truncated = title[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "…"


def _set_conversation(conversation_id: str | None) -> None:
    st.session_state.conversation_id = conversation_id
    if conversation_id is None:
        st.query_params.clear()
    else:
        st.query_params["c"] = conversation_id


def _delete_conversation(conversation_id: str) -> None:
    api("DELETE", f"/api/conversations/{conversation_id}")
    if st.session_state.conversation_id == conversation_id:
        _set_conversation(None)


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
        "Inbound",
        options=["accept", "hold", "refuse"],
        default=conversation["inbound"],
        key=f"inbound-{conversation['id']}",
    )
    if policy and policy != conversation["inbound"]:
        _set_inbound_policy(conversation["id"], policy)
        st.rerun()

    st.divider()
    others = [c for c in conversations if c["id"] != conversation["id"]]
    if not others:
        st.caption("Create a second conversation to message it.")
    else:
        to_handle = st.selectbox(
            "Send to",
            options=[c["handle"] for c in others],
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
        latest = api("GET", f"/api/conversations/{conversation_id}/messages")[-1]
    except (TurnFailed, IndexError):
        return  # a poll that fails is not an error the user caused
    seen = (latest["id"], latest["status"])
    if st.session_state.last_seen not in (None, seen):
        st.rerun()  # app scope: the sidebar's ordering can have changed too
    st.session_state.last_seen = seen


if "conversation_id" not in st.session_state:
    # Resume across a reload the same way the old localStorage did, via the
    # one thing that actually survives a reload in Streamlit's model: the URL.
    st.session_state.conversation_id = st.query_params.get("c")
st.session_state.setdefault("pending_retry", None)
st.session_state.setdefault("streaming", False)
st.session_state.setdefault("last_seen", None)

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
    if health["status"] != "ok":
        st.warning(health["llm"]["degraded_reason"])

    st.button(
        "New chat",
        icon=":material/add:",
        on_click=_set_conversation,
        args=(None,),
        width="stretch",
    )
    st.divider()

    for conversation in conversations:
        row = st.columns([5, 1], vertical_alignment="center")
        row[0].button(
            _sidebar_label(conversation["title"]),
            key=f"select-{conversation['id']}",
            type="primary" if conversation["id"] == conversation_id else "secondary",
            width="stretch",
            on_click=_set_conversation,
            args=(conversation["id"],),
        )
        row[1].button(
            "",
            icon=":material/delete:",
            help="Delete conversation",
            key=f"delete-{conversation['id']}",
            on_click=_delete_conversation,
            args=(conversation["id"],),
        )


# --- main pane ---------------------------------------------------------------

if conversation_id is None:
    st.info("Send a message to start a new conversation.")
else:
    conversation = api("GET", f"/api/conversations/{conversation_id}")
    # The header shows the title in full: this is the one place with room
    # for it; the sidebar list gets the shortened version (_sidebar_label).
    header = st.columns([7, 1, 1], vertical_alignment="center")
    header[0].subheader(conversation["title"])
    st.caption(f"@{conversation['handle']}")

    @st.dialog("Rename conversation")
    def _rename_dialog(current_title: str) -> None:
        new_title = st.text_input("Title", value=current_title)
        if st.button("Save", icon=":material/check:", type="primary") and new_title.strip():
            api("PATCH", f"/api/conversations/{conversation_id}", json={"title": new_title})
            st.rerun()

    # on_change="rerun" (lazy content, computed only while open) was tried
    # first and dropped: it doubled the header row's buttons in the DOM on
    # the very first script pass, before the popover was ever opened. The
    # default (content computed every rerun, popover or not) costs one
    # extra GET .../inbox per rerun -- cheap, and correct.
    with header[1].popover("", icon=":material/send:", help="Message another conversation"):
        _peer_popover(conversation, conversations)

    if header[2].button("", icon=":material/edit:", help="Rename conversation"):
        _rename_dialog(conversation["title"])

    messages = api("GET", f"/api/conversations/{conversation_id}/messages")
    for index, message in enumerate(messages):
        avatar = "🛰" if message["role"] == "peer" else None
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "peer":
                st.caption(f"from @{message['peer']['from_handle']} · hop {message['peer']['hops']}")
            # Streamlit's markdown renderer has unsafe_allow_html off by
            # default, so this holds the same XSS boundary the old
            # textContent-only frontend documented as a deliberate choice,
            # just with markdown formatting now rendering, which that one
            # didn't have.
            st.markdown(message["content"] or ("…" if message["status"] != "failed" else ""))
            if message["role"] == "assistant" and message["status"] != "complete":
                st.caption(message["error"]["message"] if message["error"] else "(interrupted)")
                # Only the last message: /retry always regenerates the
                # trailing reply, so the button would be misleading anywhere
                # else in the transcript.
                if index == len(messages) - 1:
                    st.button(
                        "Retry",
                        icon=":material/refresh:",
                        key=f"retry-{message['id']}",
                        on_click=_request_retry,
                        args=(conversation_id,),
                    )

    if st.session_state.pending_retry == conversation_id:
        st.session_state.pending_retry = None
        st.session_state.streaming = True
        try:
            with st.chat_message("assistant"):
                st.write_stream(stream_turn(f"/api/conversations/{conversation_id}/retry"))
        except TurnFailed as exc:
            st.error(str(exc))
        finally:
            st.session_state.streaming = False
        st.rerun()

    _watch_for_peer_activity(conversation_id)

prompt = st.chat_input("Message, or @handle to reach another conversation")
if prompt:
    target_id = conversation_id
    if target_id is None:
        target_id = api("POST", "/api/conversations", json={})["id"]
        _set_conversation(target_id)

    handle, _, text = prompt.partition(" ")
    known_handles = {c["handle"] for c in conversations}
    if prompt.startswith("@") and handle[1:] in known_handles and text.strip():
        try:
            delivery = _send_peer_message(target_id, handle[1:], text.strip())
            st.toast(f"{delivery['outcome']} → {handle}", icon="🛰")
        except TurnFailed as exc:
            st.error(str(exc))
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.streaming = True
        try:
            with st.chat_message("assistant"):
                st.write_stream(
                    stream_turn(f"/api/conversations/{target_id}/messages", {"content": prompt})
                )
        except TurnFailed as exc:
            st.error(str(exc))
        finally:
            st.session_state.streaming = False
    st.rerun()
