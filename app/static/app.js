"use strict";

const els = {
  banner: document.getElementById("banner"),
  conversationList: document.getElementById("conversation-list"),
  newChat: document.getElementById("new-chat"),
  emptyState: document.getElementById("empty-state"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
};

const state = {
  conversations: [],
  currentId: null,
  abortController: null,
};

// --- SSE ---------------------------------------------------------------

function parseEventBlock(block) {
  let event = "message";
  let dataText = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataText += line.slice(5).trim();
  }
  if (!dataText) return null;
  return { event, data: JSON.parse(dataText) };
}

async function* postSSE(url, body, signal) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.error?.message || `Request failed (HTTP ${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse-starlette writes CRLF line endings — normalize to LF so a plain
    // "\n\n" separator search below works regardless.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let separator;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const parsed = parseEventBlock(raw);
      if (parsed) yield parsed;
    }
  }
}

// --- API -----------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error?.message || `Request failed (HTTP ${response.status})`);
  }
  if (response.status === 204) return null;
  return response.json();
}

// --- Rendering -------------------------------------------------------------

function renderBanner(health) {
  const degraded = health.status !== "ok";
  els.banner.classList.toggle("hidden", !degraded);
  if (degraded) {
    els.banner.textContent = `Offline demo mode — ${health.llm.degraded_reason}`;
  }
}

function renderSidebar() {
  els.conversationList.textContent = "";
  for (const conversation of state.conversations) {
    const li = document.createElement("li");
    li.className = conversation.id === state.currentId ? "active" : "";
    li.addEventListener("click", () => selectConversation(conversation.id));

    const title = document.createElement("span");
    title.className = "title";
    title.textContent = conversation.title;
    title.title = "Double-click to rename";
    title.addEventListener("dblclick", (event) => {
      event.stopPropagation();
      renameConversation(conversation.id, conversation.title);
    });

    const deleteButton = document.createElement("button");
    deleteButton.textContent = "×";
    deleteButton.title = "Delete conversation";
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });

    li.append(title, deleteButton);
    els.conversationList.appendChild(li);
  }
}

function bubbleFor(message) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${message.role === "user" ? "user" : message.status}`;
  bubble.textContent = message.content || (message.status === "failed" ? "" : "…");

  if (message.status === "failed" && message.error) {
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = message.error.message;
    bubble.appendChild(meta);

    const retry = document.createElement("span");
    retry.className = "retry";
    retry.textContent = "Retry";
    retry.addEventListener("click", retryLast);
    bubble.appendChild(retry);
  } else if (message.status === "interrupted") {
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = "(interrupted)";
    bubble.appendChild(meta);
  }

  return bubble;
}

async function renderMessagesFromServer() {
  if (!state.currentId) return;
  let messages;
  try {
    messages = await api(`/api/conversations/${state.currentId}/messages`);
  } catch {
    messages = []; // e.g. the conversation was deleted from another tab
  }
  els.messages.textContent = "";
  for (const message of messages) {
    els.messages.appendChild(bubbleFor(message));
  }
  // Authoritative visibility, independent of whether the "start" SSE event
  // ever arrived — e.g. a send to a since-deleted conversation fails before
  // any event is sent, and this is still correct.
  els.emptyState.classList.toggle("hidden", messages.length > 0);
  els.messages.classList.toggle("hidden", messages.length === 0);
  els.messages.scrollTop = els.messages.scrollHeight;
}

// --- Actions -------------------------------------------------------------

async function refreshHealth() {
  try {
    renderBanner(await api("/api/health"));
  } catch {
    // A failed health check is not worth surfacing over whatever the user is doing.
  }
}

async function loadConversations() {
  state.conversations = await api("/api/conversations");
  renderSidebar();
}

async function selectConversation(id) {
  state.currentId = id;
  localStorage.setItem("lastConversationId", id);
  els.emptyState.classList.add("hidden");
  els.messages.classList.remove("hidden");
  renderSidebar();
  await renderMessagesFromServer();
}

async function createConversation() {
  const conversation = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({}),
  });
  state.conversations.unshift(conversation);
  await selectConversation(conversation.id);
}

async function deleteConversation(id) {
  if (!confirm("Delete this conversation?")) return;
  await api(`/api/conversations/${id}`, { method: "DELETE" });
  state.conversations = state.conversations.filter((c) => c.id !== id);
  if (state.currentId === id) {
    state.currentId = null;
    localStorage.removeItem("lastConversationId");
    els.messages.textContent = "";
    els.messages.classList.add("hidden");
    els.emptyState.classList.remove("hidden");
  }
  renderSidebar();
}

async function renameConversation(id, currentTitle) {
  const title = prompt("Rename conversation", currentTitle);
  if (!title || title === currentTitle) return;
  const updated = await api(`/api/conversations/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
  const conversation = state.conversations.find((c) => c.id === id);
  if (conversation) conversation.title = updated.title;
  renderSidebar();
}

function setStreaming(isStreaming) {
  els.send.classList.toggle("hidden", isStreaming);
  els.stop.classList.toggle("hidden", !isStreaming);
  els.input.disabled = isStreaming;
}

async function runStream(iterator) {
  let liveBubble = null;
  try {
    for await (const { event, data } of iterator) {
      if (event === "start") {
        els.emptyState.classList.add("hidden");
        els.messages.classList.remove("hidden");
        liveBubble = document.createElement("div");
        liveBubble.className = "bubble assistant";
        els.messages.appendChild(liveBubble);
        els.messages.scrollTop = els.messages.scrollHeight;
      } else if (event === "delta" && liveBubble) {
        liveBubble.textContent += data.text;
        els.messages.scrollTop = els.messages.scrollHeight;
      }
      // "done" and "error" are both resolved by re-fetching server truth below —
      // the server's persisted state is authoritative, not whatever the client
      // accumulated while streaming.
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      console.error(error);
    }
  } finally {
    await renderMessagesFromServer();
    await refreshHealth();
  }
}

async function sendMessage(content) {
  if (!state.currentId) {
    const conversation = await api("/api/conversations", {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.conversations.unshift(conversation);
    state.currentId = conversation.id;
    localStorage.setItem("lastConversationId", conversation.id);
    renderSidebar();
  }

  state.abortController = new AbortController();
  setStreaming(true);
  try {
    await runStream(
      postSSE(
        `/api/conversations/${state.currentId}/messages`,
        { content },
        state.abortController.signal
      )
    );
  } finally {
    setStreaming(false);
    state.abortController = null;
    // Re-fetch, not just re-render: heuristic titling happens server-side,
    // and the client's cached conversation list has no way to know about it.
    await loadConversations();
  }
}

async function retryLast() {
  if (!state.currentId) return;
  state.abortController = new AbortController();
  setStreaming(true);
  try {
    await runStream(
      postSSE(
        `/api/conversations/${state.currentId}/retry`,
        undefined,
        state.abortController.signal
      )
    );
  } finally {
    setStreaming(false);
    state.abortController = null;
  }
}

// --- Wiring --------------------------------------------------------------

els.newChat.addEventListener("click", createConversation);

els.stop.addEventListener("click", () => {
  state.abortController?.abort();
});

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = els.input.value.trim();
  if (!content) return;
  els.input.value = "";
  sendMessage(content);
});

async function init() {
  await refreshHealth();
  await loadConversations();

  const lastId = localStorage.getItem("lastConversationId");
  const stillExists = state.conversations.some((c) => c.id === lastId);
  if (stillExists) {
    await selectConversation(lastId);
  } else {
    els.emptyState.classList.remove("hidden");
    els.messages.classList.add("hidden");
  }
}

init();
