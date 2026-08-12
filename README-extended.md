# chatbot-fastapi-mongo

A chatbot: FastAPI backend, MongoDB persistence, a Streamlit frontend, and an
OpenAI-compatible model behind a one-method interface, OpenRouter's free tier by
default, or a local Ollama model if you'd rather not use a key at all.

**Every turn ends in a state the server chose and persisted: `complete`, `interrupted`,
or `failed`.** There is no configuration of this stack, including no API key at all, in
which it answers you with a stack trace.

Everything below is downstream of that sentence, including the things I removed to keep it
true.

---

## Quickstart (no API key needed)

```bash
docker compose up
```

Open <http://localhost:8501>. It works immediately, with no `.env`, no key, and no signup.
(The API itself is on `:8000`, for curl and Swagger; see [API](#api).)

You'll be talking to the built-in **offline demo provider**, and the app says so in five
places rather than letting you find out on your own: an amber banner in the UI, a startup
warning in the container logs, `degraded_reason` in `/api/health`, a `provider: "demo"`
field on every message it writes, and the reply text itself.

That is the deliberate answer to "what should happen if someone brings the project up
without configuring the key". A missing key is a configuration that resolves to a
different provider, not a failure mode, and not a crash.

## 🛰 Conversations that message each other

The standout feature, and the fastest way to see this app do something a chat demo
usually doesn't: open two browser tabs, each on a different conversation, and have one
message the other. It works in demo mode, with no API key, no signup, nothing.

This is a small, playable port of Claude Code's own
[cross-session messaging](https://code.claude.com/docs/en/cross-session-messaging) onto
this app's conversations. Every conversation gets a **handle**, derived from its title
and id, e.g. `payments-api-3f0a`, listed beside every chat in the sidebar, and can message
another conversation by that handle. Click the 🛰 icon next to a conversation's title, or
type `@payments your message` into the chat box.

The `@` shortcut takes any unambiguous prefix or fragment of a handle or title, so
`@payments` reaches `payments-api-3f0a` without anyone having to retype an id suffix; two
candidates is an error listing both rather than a guess at which one was meant. Anything
starting with `@` is treated as a send attempt and never quietly re-routed to the model:
a handle that doesn't resolve, or a mention with no message after it, says so and names
the conversations that *are* reachable. Silently answering `@typo hello` as an ordinary
prompt was the single thing that made this feature look broken rather than misaddressed.

```bash
A=$(curl -s -X POST localhost:8000/api/conversations -d '{"title":"a"}' -H 'content-type: application/json' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
B=$(curl -s -X POST localhost:8000/api/conversations -d '{"title":"b"}' -H 'content-type: application/json' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
B_HANDLE=$(curl -s localhost:8000/api/conversations/$B | python3 -c "import sys,json;print(json.load(sys.stdin)['handle'])")

curl -s -X POST "localhost:8000/api/conversations/$A/send" \
  -H 'content-type: application/json' -d "{\"to_handle\": \"$B_HANDLE\", \"text\": \"ping\"}"
# {"outcome": "delivered", "to_handle": "b-...", "to_conversation_id": "...", "held_id": null}

curl -s localhost:8000/api/conversations/$B/messages   # a peer row, then B's own reply
```

What actually happens on `delivered`: B runs a turn server-side, with no browser
attached, the same `ChatService` lifecycle as any other turn, so the reply is durable the
same way: the assistant row is written as `interrupted` before the first token, same as
every other turn in this app. If that reply is real, it forwards back to A, whose own
turn answers it, and so on: the two conversations visibly volley for a few turns, then
stop on their own, because `PEER_HOP_LIMIT` (default `3`) bounds it. Set it to `1` to
deliver and reply once but never forward.

Each conversation also has an **inbound policy**, `accept` (the default), `hold`, or
`refuse`, settable from the same 🛰 popover. `hold` queues an incoming message in that
conversation's inbox instead of running a turn, for you to Approve or Deny by hand.
`refuse` drops it silently. Neither is an error: they're ordinary outcomes of a working
system, the same way the real feature treats them.

**What this deliberately does *not* port**, because the brief here is fun and playable,
not a real feature with real implications:

- **No TTL / `dialogExpiry`.** A held message waits until you act on it, no timers.
- **No permission-class inference.** There's one process and no bypass-permissions
  concept to infer from; `hold`/`refuse` are set explicitly, per conversation.
- **No cross-machine delivery.** Every "session" here is a conversation in the same
  database; there is no second machine to reach.
- **No queue caps or per-sender rate limiting.** Every automatic exchange strictly
  alternates between the two conversations that started it; nothing here ever reads a
  reply's text looking for a third handle to forward to, so `PEER_HOP_LIMIT` is already
  a stronger guarantee than a queue cap: there's no path to an *unsupervised* loop at
  all, not just a bounded one. A longer relay across more conversations only happens
  through a separate, human-initiated send for each hop.
- **No streaming on `/send`.** The interesting output lands in the *other* conversation,
  not in the sender's response, so there's nothing here for the sender to stream.

## Talking to a real model

The default model is free, but OpenRouter's free tier is **20 requests per minute and 50
per day** across all free models. That is a real constraint on a demo: a reviewer clicking
around will hit it, so the app treats 429 as a first-class outcome rather than an
afterthought (see [Failure modes](#failure-modes)).

1. Get a key at <https://openrouter.ai/keys>.
2. `cp .env.example .env` and set `LLM_API_KEY=sk-or-...`
3. `docker compose up`

The banner disappears, `/api/health` reports `"status": "ok"`, and replies stream from the
real model. `.env` is gitignored and no key is ever logged or included in an error message.

## Or run the model locally, with no key at all

If you'd rather not sign up for anything, or you've burned through the 50/day, point the
app at [Ollama](https://ollama.com) instead. No key, no quota, no data leaving the machine:

```bash
ollama pull gemma4:latest
```

With no `LLM_API_KEY` set, the sidebar shows a **"Use local Ollama model" toggle** — flip
it and the next message goes to Ollama, no `.env` edit and no restart, because there's
nothing yet to override: `PUT /api/settings/llm-provider` just swaps the app's in-memory
client. It's a runtime convenience, not a persisted setting: it reverts to whatever
`LLM_PROVIDER` says on the next restart, and it 400s the moment a key *is* configured,
since a real key already answers "which provider" (see [Configuration](#configuration)).

To make Ollama the default instead, set `LLM_PROVIDER=ollama` in `.env` and rebuild (see
[Rebuilding after a change](#rebuilding-after-a-change) below). Ollama speaks the same
OpenAI-compatible protocol as OpenRouter, so it reuses the same client; providers here are
configuration, not classes.

Three things worth knowing:

- **`auto` will never pick Ollama.** It is opt-in only. Silently routing a conversation to
  a different model than the one configured is the same failure as silently falling back to
  the fake, and this app doesn't do either.
- **Prefer a non-reasoning instruct model.** Reasoning models (`qwen3.5`, `deepseek-r1`)
  stream their thinking into a separate `reasoning` field and leave `content` empty until
  it finishes, and over an OpenAI-compatible stream that is indistinguishable from a hang,
  for however long the model thinks. I hit this with `qwen3.5:4b`, which is why the default
  is a plain instruct model instead.
- **The first message is slow**, because Ollama loads the model into RAM on demand. The
  client allows 300s for Ollama against 60s for hosted providers, for exactly that reason.

If Ollama isn't running, or the model isn't pulled, you get told which; see
[Failure modes](#failure-modes).

## Verify it works

Twelve checks, in the order I'd run them:

| # | Do this | Expect |
|---|---|---|
| 1 | `docker compose up --build` from a clean clone, **no `.env` at all** | Chat works at `:8501`, amber banner explains why, no traceback in the logs |
| 2 | `curl -s localhost:8000/api/health` | `200`, `"status": "degraded"`, `"configured": false`, and a `degraded_reason` naming the fix |
| 3 | With no key set, flip the sidebar's "Use local Ollama model" toggle | `/api/health` flips `"provider": "demo"` ↔ `"ollama"` immediately, no restart |
| 4 | Add a real key, restart | Banner gone, `"status": "ok"`, replies stream from the model, and the sidebar toggle is gone (`has_api_key: true`) |
| 5 | `docker compose down && docker compose up -d` | Conversations still there. (`down`, never `down -v`: the data lives in the `mongo_data` volume) |
| 6 | `make test` | Passing, **with no API key set**; that green is the proof the suite is hermetic |
| 7 | Set `LLM_API_KEY=garbage`, send a message | A clear error message: "The configured LLM_API_KEY was rejected…", one upstream call, no retry storm |
| 8 | Send a message, then click Streamlit's own Stop control mid-stream | The reply is there on reload, marked `interrupted`, with a Retry button |
| 9 | `docker compose stop mongo`, then `curl localhost:8000/api/health` | Still `200`, `"mongo": "unreachable"`: degraded, not down |
| 10 | Two tabs, two conversations: click 🛰 on one and send to the other's handle | The reply appears in the other tab within ~2s, unprompted; the exchange volleys a few turns, then stops |
| 11 | Set one conversation's inbound policy to `hold`, send to it | The message sits in that conversation's inbox until Approved or Denied; no turn runs until then |
| 12 | While a reply nobody in this tab started is being written — the receiving side of check 10, or a reload mid-turn — watch that row | "Generating…", no Retry button, and it turns into the reply on its own. Compare `live` in `GET .../messages`: `true` only while it's being written. Check 8's genuinely interrupted row still shows `(interrupted)` and a Retry |

Check 6 in full, if you'd rather not use `make`:

```bash
docker compose --profile test run --rm --build tests
```

## Architecture

```
streamlit_app.py (HTTP, separate process)
        ↓
routes.py  →  chat.py (ChatService)  →  repository.py  →  MongoDB
                     ↓        ↑
              llm/base.py (Protocol)     peers.py (deliver, run_exchange)
                     ↓
           openai_compatible.py | demo.py
              (openrouter, ollama)
```

Twelve Python modules, about 1,400 lines including the comments, plus `streamlit_app.py`
(~250 lines) as a separate process with no shared code. Small enough to read in one
sitting, which is the point.

- **`routes.py`** owns HTTP and nothing else.
- **`chat.py`** owns the turn lifecycle and imports neither `fastapi` nor `openai`. That
  boundary is what makes the core testable with a fake LLM and no HTTP layer.
- **`repository.py`** is the only module that imports `pymongo`. No abstract base class:
  there is exactly one implementation and Mongo isn't being swapped, so an interface would
  be ceremony rather than design.
- **`peers.py`** is conversation-to-conversation messaging (see
  [above](#-conversations-that-message-each-other)). Same rule as `chat.py`: it imports
  neither `fastapi` nor `pymongo`, so it's testable the same way, with a fake LLM and a
  real throwaway database, no HTTP layer.
- **`llm/base.py`** is the one Protocol in the codebase, and it earns its ~20 lines three
  times over: it's the test seam, the zero-key degradation seam, and the thing that shows
  dependency inversion at a glance.
- **`llm/openai_compatible.py`** is one class serving both OpenRouter and Ollama, because
  there is nothing to differentiate: same wire protocol, and only a base URL, a model name,
  a timeout, and whether a key is required actually vary. Providers are configuration, not
  subclasses; adding the third one was a `build_llm` branch, not a file.

**`streamlit_app.py` is a separate process and a pure HTTP client of the API**, not a
second implementation of anything above. It imports nothing from `app/`, touches no
database, and talks to the API the same way `curl` does, over the compose network at
`http://api:8000`. That's also why it needs no CORS entry on the API: the call is
server-to-server, from Streamlit's own Python backend, never from the browser. See
[what changed in the swap](#the-streamlit-swap) for the trade-offs that came with it.

### The turn lifecycle

This is the core of the submission, and the ordering of its two writes is the whole design:

1. **The user's message is committed before any provider call.** A failed completion must
   never eat the user's turn.
2. **The assistant's row is committed before the first token**, already reading
   `interrupted` with empty content: *"if nothing further happens, this is what
   happened."* Streaming then updates it to `complete`, or to `failed` with the error
   attached, or enriches the `interrupted` row with whatever text arrived.

The point of (2) is that the honest terminal state is persisted **by construction rather
than by cleanup**. No teardown path has to run correctly for the transcript to be true.
That is not the design I started with, and [the reason I changed it](#4-a-turn-lifecycle-that-passed-its-tests-and-lost-data)
is the most useful thing in this README.

Stop and a closed tab both land on `interrupted`, because the server genuinely cannot tell
them apart. Saying so is more honest than inventing a distinction.

**The cost of (2), and what it took to pay it.** A reply being written *right now* is on
disk as `interrupted` too — that is the point — so anyone reading the transcript who isn't
the caller that started the turn sees a row that says "cut short" about a model still
typing. That is not an edge case: the receiving half of an [@-send](#-conversations-that-message-each-other)
runs server-side with no client attached, so the UI polling for it hit this on every peer
reply, drawing a Retry button under a live turn. The fix keeps the terminal states at
three and adds nothing to the database: `GET .../messages` marks the rows this process is
currently generating with `live: true`, from an in-memory set held by `ChatService`.
Liveness is a fact about a running process, not about the record, and it dies correctly
with the process — after a crash or a restart nothing is generating, and every such row
then reads exactly what it is.

## Data model

Three collections. Conversations and messages are **not** an embedded messages array:
that means unbounded growth against a 16 MB document ceiling and a full-document rewrite
on every single turn.

```
conversations  { _id, title, created_at, updated_at, model,
                 inbound: "accept" | "hold" | "refuse" }
messages       { _id, conversation_id, role: "user" | "assistant" | "peer", content,
                 status: "complete" | "interrupted" | "failed",
                 error: null | {code, message, retry_after_seconds},
                 provider, model, usage, ttft_ms, total_ms, created_at,
                 peer: null | {from_handle, from_conversation_id, hops} }
held_messages  { _id, to_conversation_id, from_conversation_id, from_handle, text,
                 hops, created_at }
```

A message comes back over the API with one field that is not in the document above:
`live`, filled in from process state at read time. Nothing writes it anywhere.

**`handle` is computed, never stored**: `slugify(title) + "-" + id[-4:]`, e.g.
`payments-api-3f0a`, the same way Claude Code names a session after its folder. That's
why the peer-messaging fields above are additive with model defaults and needed no
migration: every conversation and message already written before `app/peers.py` existed
loads unchanged, `inbound` defaults to `accept`, `peer` defaults to `None`, and a
`handle` falls out of fields that were already there. It also means renaming a
conversation changes the handle it answers to, exactly like Claude Code's own `/rename`.

**`held_messages` is its own collection, not a fourth `MessageStatus`.** A held message
was never delivered, so it has no place in a transcript, and giving it a `"held"` status
would put a non-terminal value in the one field this app's whole design is about.

The cost of the original choice, stated plainly: **no cross-turn atomicity.** A
conversation and its messages are written separately. That is also precisely why this
needs no replica set and no transactions, and why `delete_conversation` removes messages
(and any held messages addressed to it) *first*, since a conversation with no messages is
still readable while messages with no conversation are unreachable garbage.

Indexes, created idempotently on every startup:

- `conversations {updated_at: -1, _id: -1}`: the sidebar's sort order.
- `messages {conversation_id: 1, created_at: 1, _id: 1}`: the history read. Its prefixes
  `{conversation_id: 1}` and `{conversation_id: 1, created_at: 1}` also serve the cascade
  delete, so there is no second index.
- `held_messages {to_conversation_id: 1, created_at: 1, _id: 1}`: the inbox read, same
  prefix rule, so it also serves that collection's own cascade delete.

**Why `_id` is in both of those.** BSON datetimes are millisecond-precision, and a user's
message and its reply routinely land inside the *same* millisecond, measured here at 32
of 40 turns. A sort on the timestamp alone is then formally unordered, and MongoDB is free
to return the answer before the question, which is not a display bug: it is the wrong
conversation being replayed to the model on the next turn. ObjectIds embed a timestamp and
a per-process counter, so within the single process that writes any one turn they increase
in exactly the order the rows were created. The trailing `_id` on the index keeps the
tiebroken sort a pure index scan instead of an in-memory `SORT` stage, and there's a test
asserting exactly that against `explain()`.

Restart persistence is the named volume `mongo_data:/data/db`.

## API

| Route | Notes |
|---|---|
| `GET /api/health` | **Always 200.** Reports `status`, `mongo`, and `llm: {provider, configured, has_api_key, model, degraded_reason}`. Three consumers: Docker's HEALTHCHECK, the UI banner, and a reviewer with one curl. A 503 here would turn a keyless clone into a flapping container, the opposite of what the brief asks for. |
| `PUT /api/settings/llm-provider` | `{provider: "demo" \| "ollama"}`. The keyless demo/Ollama toggle the sidebar shows when `has_api_key` is `false`; swaps the app's in-memory client, no restart. `400 provider_switch_not_allowed` once a real key is configured (see [Configuration](#configuration)). |
| `POST /api/conversations` | 201. |
| `GET /api/conversations` | Sorted by `updated_at` desc; the projection never drags whole transcripts along. |
| `GET/PATCH/DELETE /api/conversations/{id}` | `PATCH` renames: it exists so auto-generated titling (see [Decisions](#decisions-and-trade-offs)) reads as a choice rather than a limitation. `DELETE` cascades to messages first, then 204. |
| `GET /api/conversations/{id}/messages` | Includes `failed` and `interrupted` messages. History is honest about what happened. Each row also carries `live`: `true` only while this process is generating it, which is the one thing the stored row cannot say (see [the turn lifecycle](#the-turn-lifecycle)). Never stored, never `true` after a restart. |
| `POST /api/conversations/{id}/messages` | **One generator, two renderers.** `Accept: text/event-stream` gets SSE; anything else gets JSON with a real HTTP status. Streaming for the product, plain JSON for Swagger and curl, zero duplicated lifecycle logic. |
| `POST /api/conversations/{id}/retry` | Drops a trailing `failed` or `interrupted` reply and re-enters the same generator. The working answer to "the reviewer's key hit 50/day mid-demo". |
| `GET /api/agents` | The roster, mirroring Claude Code's `/list-agents`: every conversation, by handle. |
| `PUT /api/conversations/{id}/inbound` | Sets `accept` \| `hold` \| `refuse`. |
| `POST /api/conversations/{id}/send` | `{to_handle, text}` → `202` with `{outcome: "delivered" \| "held" \| "refused", ...}`. On `delivered`, the receiving turn runs as a background task; see [Conversations that message each other](#-conversations-that-message-each-other). No SSE: the interesting output lands in the *other* conversation, not this response. |
| `GET /api/conversations/{id}/inbox` | Messages held by that conversation's `hold` policy, oldest first. |
| `POST .../inbox/{held_id}/approve`, `DELETE .../inbox/{held_id}` | Approve delivers (runs the turn); deny drops it. Both remove the held row either way. |
| `GET /docs` | Generated API docs (Swagger). `/` 404s: the UI is `streamlit_app.py`, a separate service on `:8501`. |

The SSE stream is `start` → many `delta` → exactly one `done` **or** one `error`:

```bash
CONV=$(curl -s -X POST localhost:8000/api/conversations \
  -H 'content-type: application/json' -d '{}' | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

curl -N -X POST "localhost:8000/api/conversations/$CONV/messages" \
  -H 'content-type: application/json' -H 'accept: text/event-stream' \
  -d '{"content":"hello"}'
```

```
event: start
data: {"conversation_id": "6a7b…0ed", "user_message_id": "6a7b…0ee", "assistant_message_id": "6a7b…0ef"}

event: delta
data: {"text": "This "}
```

Drop the `accept` header and the same turn comes back as one JSON object with a real
status code: same lifecycle, same persistence, different renderer.

## Configuration

Every variable has a working default except the API key. `tests/test_config.py` introspects
`Settings.model_fields` and fails the build if a new setting is added without one, or if it
isn't documented in `.env.example`, about fifteen lines that prevent the hour-nine
variable with no default.

| Variable | Default | Notes |
|---|---|---|
| `MONGO_URI` | `mongodb://mongo:27017` | |
| `MONGO_DB` | `chatbot` | |
| `LLM_PROVIDER` | `auto` | `auto` \| `openrouter` \| `ollama` \| `demo` |
| `LLM_API_KEY` | *(none)* | The only setting with no default. Blank is treated as unset. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `google/gemma-4-26b-a4b-it:free` | |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Only read when `LLM_PROVIDER=ollama` |
| `OLLAMA_MODEL` | `gemma4:latest` | A non-reasoning instruct model; see the note above |
| `PEER_HOP_LIMIT` | `3` | How many replies a peer exchange forwards before it stops. `1` delivers and replies but never forwards. |

Ollama gets its own base-URL and model rather than reusing `LLM_BASE_URL`/`LLM_MODEL`, so
that switching providers is one variable instead of three, and so an Ollama run can't
default to an OpenRouter model id that no local install has.

`LLM_PROVIDER` resolves **one-directionally: a present key always wins.** `auto` with a key
uses OpenRouter; `auto` without one uses the demo provider; `demo` always uses the demo
provider even if a key is present; `openrouter` always uses OpenRouter and fails per-turn
with a named error if no key is set; `ollama` always uses the local endpoint and needs no
key at all. The app never silently prefers the fake, and `auto` never reaches for a local
model on its own: a reviewer who suspects the integration was faked has already failed the
submission.

**`/api/health`'s `has_api_key` is a different question from `configured`.** `configured`
answers "does the *selected* provider have what it needs" — true for Ollama even with no
key, which is the whole point of offering it. `has_api_key` answers "is `LLM_API_KEY` set
at all," regardless of provider, and it's what the sidebar's demo/Ollama toggle and
`PUT /api/settings/llm-provider` gate on: switching providers at runtime is only safe to
offer when nothing has already answered "which provider" via a real key.

## Failure modes

Every row here has a test.

| Trigger | Code | HTTP | Persisted as |
|---|---|---|---|
| No key, `LLM_PROVIDER=auto` | — | 200 | `complete` (demo provider) |
| No key, `LLM_PROVIDER=openrouter` | `llm_not_configured` | 503 | `failed` |
| Rejected key | `invalid_key` | 502 | `failed` |
| Free-tier limit reached | `rate_limited` | 429 | `failed` + `retry_after_seconds` |
| Model not pulled / unknown model id | `model_not_available` | 502 | `failed` |
| Ollama not running, or provider unreachable | `provider_unreachable` | 502 | `failed` |
| No credits (402), upstream 5xx, mid-stream provider error | `upstream_error` | 502 | `failed` |
| Timeout mid-stream | `upstream_error` | 502 | `failed` |
| Stop button, or closed tab | — | *(connection closed)* | `interrupted` |
| Unknown conversation id | `conversation_not_found` | 404 | — |
| Malformed conversation id | `validation_error` | 422 | — |
| Retry with nothing to retry | `nothing_to_retry` | 400 | — |
| Toggle demo/Ollama while a real key is configured | `provider_switch_not_allowed` | 400 | — |
| MongoDB unreachable | — | 200 on `/api/health` | — |
| Send to an unknown handle | `peer_not_found` | 404 | — |
| A conversation sends to itself | `cannot_message_self` | 400 | — |
| Held or refused inbound policy on send | — | 202, `outcome` names it | — (queued or dropped, not an error) |

Every error response carries the same envelope, including a `request_id` that is echoed in
the `X-Request-Id` header, so an error message in the browser can be grepped straight out
of `docker compose logs`:

```json
{"error": {"code": "rate_limited",
           "message": "The AI provider's free-tier rate limit was reached. Retry after 17s.",
           "request_id": "b1e4…", "retry_after_seconds": 17}}
```

On the SSE path a failure arrives as an `error` **event**, not an HTTP status: by then
`start` and possibly several `delta` frames are already on the wire and the 200 is
committed. Same envelope, different transport.

## Decisions and trade-offs

### What I deliberately left out

The omissions are the argument, so they go first.

- **Model fallback chains on 429.** Silently switching to a different model mid-conversation
  changes the voice without the user's consent and masks real bugs. The right answer to an
  exhausted free tier is to say so and offer Retry.
- **Auto-falling back to the demo provider after a live failure.** One line away from a good
  idea, and it inverts the entire story: a stack that quietly starts faking replies when the
  real thing breaks is worse than one that admits it broke.
- **Our own retry policy, and the SDK's.** `max_retries=0` on the client is a decision, not
  a default. On a 50-request-per-day quota, a blind backoff spends a second unit of that
  quota to produce the same error the user sees anyway, and retrying mid-stream would
  duplicate content. The 429 test asserts the upstream was called exactly once.
- **A cost ledger, `/stats` percentiles, an eval harness, detachable streams that survive the
  HTTP request, a ULID scheme, cursor pagination, an ADR directory.** All considered, all cut
  as ceremony at this size. Detachable streams in particular are correct at exactly one
  worker and are genuinely a multi-day feature, not an afternoon's.
- **The rest of cross-session messaging**, listed with the feature itself: no TTL, no
  permission-class inference, no cross-machine delivery, no queue caps or rate limiting, no
  streaming on `/send`. See [Conversations that message each
  other](#-conversations-that-message-each-other).

### What I kept, and why it cost something

- **Two collections, no transactions.** Bought bounded documents and cheap appends; paid
  with no cross-turn atomicity. Stated above rather than hidden.
- **A placeholder row written before the first token.** Bought a guarantee that holds without
  any cleanup path running; paid with one extra write per turn and a brief window where a row
  exists with empty content. For a chat app that is not a real cost.
- **Partial text on interruption is best-effort.** The *status* is always correct; the text
  that had streamed before a disconnect is recorded only if the teardown gets that far. I
  chose a correct state with possibly-missing text over the reverse.
- **`/api/health` never returns non-200.** Makes the endpoint useless as a load-balancer
  liveness probe in the conventional way, and it is the right call here: the container is
  genuinely up and serving when Mongo is down, and it should say what's wrong rather than
  disappear.
- **LLM-generated conversation titles, reversed from the original call.** The first draft
  titled every conversation with a truncation of the raw first message, on the reasoning
  that spending one of a reviewer's 50 daily requests on a sidebar label was the wrong
  trade. It held up fine for short openers and read badly for real ones: "In one short
  sentence, why use two collections instead of embedded messages?" makes a poor list item.
  Now the model writes its own 3-to-6-word summary, one extra round trip before the reply
  starts streaming on a brand-new conversation only. Demo mode is excluded, since its one
  fixed reply text has nothing to do with what was asked, and any failure (rate limit,
  Ollama still loading, an SDK error) falls back to the original truncation heuristic,
  which now doubles as the format guardrail on whatever the model actually returns.
- **A live demo/Ollama toggle, mutable global settings, one process.** `PUT
  /api/settings/llm-provider` mutates the app's single cached `Settings` instance and
  rebuilds `app.state.llm` in place, so flipping providers needs no restart. That is safe
  precisely because there is one worker process and no key yet: the moment `LLM_API_KEY`
  is set, a real key has already answered "which provider" (the one-directional
  resolution above), and the route 400s rather than silently overriding it. A design with
  multiple workers or a real key present would need this to be a request-scoped decision
  instead of a mutable global; at this size, for a toggle that only exists to make the
  keyless demo more honest to compare against a real local model, it isn't.

## How I used AI

I used Claude Code throughout, for planning, implementation, and adversarial review. The
useful thing to report is not that I used it, but the five places where I overrode it, or
where it corrected itself, since each is checkable.

### 1. Motor → `pymongo.AsyncMongoClient`

Every model I asked reached for `motor` as the async MongoDB driver. Motor reached
**end-of-life on 14 May 2026**; the supported async driver is `AsyncMongoClient`, built into
PyMongo 4.9+. This repo uses it. MongoDB's own migration guide confirms it in about twenty
seconds, worth checking because it's the clearest example of confidently-stated,
out-of-date training data in this stack.

### 2. Embedded messages → two collections

The first-draft schema embedded messages as an array inside each conversation. It's the
common shape in tutorials and it's wrong here for two specific reasons: a 16 MB per-document
ceiling that a long conversation will eventually hit, and a full-document rewrite on every
turn. Two collections, as above.

### 3. Model fallback chains on 429 → an honest refusal

Suggested repeatedly and rejected for the reason in the decisions section. This one is a
judgment call rather than a factual correction, which is exactly why it needed a human.

### 4. A turn lifecycle that passed its tests and lost data

The most instructive one, and the reason I trust "measured against the running container"
over "the tests are green".

The original design wrote the assistant message **once, at the end**, from the turn
generator's `finally` block, protected with `asyncio.shield` so a cancellation couldn't cut
the write short. It reads beautifully. It had a unit test that cancelled the consuming task
and asserted the partial was persisted as `interrupted`. That test passed.

It was also wrong. The unit test cancelled a task with `task.cancel()`; a real browser
disconnect doesn't do that: it tears down the SSE renderer *wrapped around* the turn
generator, and `async for` abandons its iterator when the loop exits by exception, leaving
the `finally` to whenever the garbage collector gets around to finalizing an orphaned async
generator. Driving 16 real mid-stream disconnects against the running container:

- **10–13 of 16 lost the assistant message entirely.** No row, no error, no log line.
- The survivors landed hundreds of milliseconds late.
- Closing the generators explicitly fixed some of it and introduced
  `RuntimeError: aclose(): asynchronous generator is already running`: two teardown paths
  racing to close the same generator.

Three successive patches at the cleanup layer each moved the number without fixing it, which
was the signal that the layer was wrong. Durability cannot depend on async-generator
finalization, so it no longer does: the row is written up front, already reading
`interrupted`, and every later write only improves the record. **0 of 24 disconnects lose the
message now**, and the regression test drives the real teardown path, closing the SSE
renderer, rather than the cancellation shape that gave a false pass.

I'd rather submit this with the history described than with the elegant version that silently
dropped a message in half the cases the README's own headline sentence is about.

### 5. A claim about the peer-messaging loop that was wrong, caught by writing the test

While building [conversations that message each other](#-conversations-that-message-each-other),
an early draft of this README claimed an A→B→C→A cycle across three conversations
"terminates on the same shared hop counter." That's a natural-sounding claim, and it's
false: `run_exchange` only ever swaps between the *two* conversations that started an
exchange; nothing in it reads a reply's text looking for a third handle to route to, so
a genuine three-way cycle can't occur inside one automatic exchange at all, not
"eventually, once the counter runs out."

The claim didn't survive contact with a test. Asked to write
`test_an_exchange_never_pulls_in_a_third_decoy_conversation`, the honest version of the
guarantee fell out on its own: a longer relay across more than two conversations only
happens through a separate, human-initiated send for each extra hop, which is a stronger
property than "cycles are bounded," and it's what the feature section and the code
comments say now. Left in as the clearest example here of a plausible-sounding claim
that needed a test before it went in a README, not after.

---

## Development

```bash
make up      # docker compose up --build
make test    # the suite, hermetic, no API key required
make lint    # ruff check
make fmt     # ruff format
make down    # stop (keeps the volume)
```

### Rebuilding after a change

Two different commands, because the two things that change need different treatment:

```bash
docker compose up -d              # a .env value changed: LLM_PROVIDER, OLLAMA_MODEL, a key
docker compose up -d --build      # any .py file, the Dockerfile, or .streamlit/config.toml changed
```

Settings like `LLM_PROVIDER` are read from the environment at container start, not baked
into the image, so switching to Ollama is genuinely just editing `.env` and running
`up -d`: Compose notices the config changed and recreates only the `api` container, no
rebuild needed. Verified directly, not assumed: `up -d` alone was enough to move a running
container from the demo provider to `LLM_PROVIDER=ollama` and back.

Code is the opposite. `up -d` with no `--build` silently keeps serving the *previous*
image, so a real change looks like nothing happened. This bit the `.streamlit/config.toml`
theme once already during development: it lives in the repo but has to be copied into the
`web` image explicitly (see the Dockerfile's `ui` stage), so a `--build`-less rebuild kept
running the old image with the old theme, and the missing `COPY` only became obvious by
checking the running container directly (`docker compose exec web ls .streamlit/`), not by
staring at the diff.

If something looks stale after a change, that check generalizes: `docker compose exec
<service> <command>` to look inside the container that's actually running, rather than
trusting that a rebuild happened.

### Testing

The suite runs against a real MongoDB, a throwaway database per test dropped on teardown,
because the Repository boundary is exactly where a fake would stop proving anything. The LLM
is faked at the `LLMClient` Protocol, and `tests/test_llm_openai_compatible.py` goes one better by
faking only the HTTP transport, so the exception mapping is verified against the real
`openai` SDK's parsing rather than an assumption about it. An autouse fixture makes any real
outbound HTTP request raise, and a test asserts that guard actually fires, so "green with no
API key" means the suite genuinely never left the machine.
