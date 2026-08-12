# chatbot-fastapi-mongo

A chatbot: FastAPI backend, MongoDB persistence, a vanilla HTML/JS frontend, and an
OpenAI-compatible model behind a one-method interface — OpenRouter's free tier by
default, a local Ollama model if you'd rather not use a key at all.

**Every turn ends in a state the server chose and persisted — `complete`, `interrupted`,
or `failed`.** There is no configuration of this stack, including no API key at all, in
which it answers you with a stack trace.

Everything below is downstream of that sentence, including the things I removed to keep it
true.

---

## Quickstart (no API key needed)

```bash
docker compose up
```

Open <http://localhost:8000>. It works immediately — with no `.env`, no key, and no signup.

You'll be talking to the built-in **offline demo provider**, and the app says so in five
places rather than letting you find out on your own: an amber banner in the UI, a startup
warning in the container logs, `degraded_reason` in `/api/health`, a `provider: "demo"`
field on every message it writes, and the reply text itself.

That is the deliberate answer to "what should happen if someone brings the project up
without configuring the key". A missing key is a configuration that resolves to a
different provider — not a failure mode, and not a crash.

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

If you'd rather not sign up for anything — or you've burned through the 50/day — point the
app at [Ollama](https://ollama.com) instead. No key, no quota, no data leaving the machine:

```bash
ollama pull llama3.2
```

Then set `LLM_PROVIDER=ollama` in `.env` and `docker compose up`. That's the whole change:
Ollama speaks the same OpenAI-compatible protocol, so it reuses the same client — providers
here are configuration, not classes.

Three things worth knowing:

- **`auto` will never pick Ollama.** It is opt-in only. Silently routing a conversation to
  a different model than the one configured is the same failure as silently falling back to
  the fake, and this app doesn't do either.
- **Prefer a non-reasoning instruct model.** Reasoning models (`qwen3.5`, `deepseek-r1`)
  stream their thinking into a separate `reasoning` field and leave `content` empty until
  it finishes — over an OpenAI-compatible stream that is indistinguishable from a hang, for
  however long the model thinks. I hit this with `qwen3.5:4b`, which is why the default is
  `llama3.2`.
- **The first message is slow**, because Ollama loads the model into RAM on demand. The
  client allows 300s for Ollama against 60s for hosted providers, for exactly that reason.

If Ollama isn't running, or the model isn't pulled, you get told which — see
[Failure modes](#failure-modes).

## Verify it works

Eight checks, in the order I'd run them:

| # | Do this | Expect |
|---|---|---|
| 1 | `docker compose up` from a clean clone, **no `.env` at all** | Chat works at `:8000`, amber banner explains why, no traceback in the logs |
| 2 | `curl -s localhost:8000/api/health` | `200`, `"status": "degraded"`, `"configured": false`, and a `degraded_reason` naming the fix |
| 3 | Add a real key, restart | Banner gone, `"status": "ok"`, replies stream from the model |
| 4 | `docker compose down && docker compose up -d` | Conversations still there. (`down`, never `down -v` — the data lives in the `mongo_data` volume) |
| 5 | `make test` | 67 passing, **with no API key set** — that green is the proof the suite is hermetic |
| 6 | Set `LLM_API_KEY=garbage`, send a message | Red bubble: "The configured LLM_API_KEY was rejected…", one upstream call, no retry storm |
| 7 | Close the tab mid-stream, reopen it | The reply is there, marked `interrupted`, with a Retry button |
| 8 | `docker compose stop mongo`, then `curl localhost:8000/api/health` | Still `200`, `"mongo": "unreachable"` — degraded, not down |

Check 5 in full, if you'd rather not use `make`:

```bash
docker compose --profile test run --rm --build tests
```

## Architecture

```
routes.py  →  chat.py (ChatService)  →  repository.py  →  MongoDB
                     ↓
              llm/base.py (Protocol)  →  openai_compatible.py | demo.py
                                       (openrouter, ollama)
```

Eleven Python modules, about 1,200 lines including the comments — plus ~550 lines of
dependency-free HTML, CSS and JavaScript. Small enough to read in one sitting, which is
the point.

- **`routes.py`** owns HTTP and nothing else.
- **`chat.py`** owns the turn lifecycle and imports neither `fastapi` nor `openai`. That
  boundary is what makes the core testable with a fake LLM and no HTTP layer.
- **`repository.py`** is the only module that imports `pymongo`. No abstract base class:
  there is exactly one implementation and Mongo isn't being swapped, so an interface would
  be ceremony rather than design.
- **`llm/base.py`** is the one Protocol in the codebase, and it earns its ~20 lines three
  times over: it's the test seam, the zero-key degradation seam, and the thing that shows
  dependency inversion at a glance.
- **`llm/openai_compatible.py`** is one class serving both OpenRouter and Ollama, because
  there is nothing to differentiate: same wire protocol, and only a base URL, a model name,
  a timeout, and whether a key is required actually vary. Providers are configuration, not
  subclasses — adding the third one was a `build_llm` branch, not a file.

The frontend is served by the same container at the same origin, so there is no CORS
configuration and no second service to run.

### The turn lifecycle

This is the core of the submission, and the ordering of its two writes is the whole design:

1. **The user's message is committed before any provider call.** A failed completion must
   never eat the user's turn.
2. **The assistant's row is committed before the first token**, already reading
   `interrupted` with empty content — *"if nothing further happens, this is what
   happened."* Streaming then updates it to `complete`, or to `failed` with the error
   attached, or enriches the `interrupted` row with whatever text arrived.

The point of (2) is that the honest terminal state is persisted **by construction rather
than by cleanup**. No teardown path has to run correctly for the transcript to be true.
That is not the design I started with, and [the reason I changed it](#4-a-turn-lifecycle-that-passed-its-tests-and-lost-data)
is the most useful thing in this README.

Stop and a closed tab both land on `interrupted`, because the server genuinely cannot tell
them apart. Saying so is more honest than inventing a distinction.

## Data model

Two collections. **Not** an embedded messages array: that means unbounded growth against a
16 MB document ceiling and a full-document rewrite on every single turn.

```
conversations  { _id, title, created_at, updated_at, model }
messages       { _id, conversation_id, role, content,
                 status: "complete" | "interrupted" | "failed",
                 error: null | {code, message, retry_after_seconds},
                 provider, model, usage, ttft_ms, total_ms, created_at }
```

The cost of that choice, stated plainly: **no cross-turn atomicity.** A conversation and
its messages are written separately. That is also precisely why this needs no replica set
and no transactions — and why `delete_conversation` removes messages *first*, since a
conversation with no messages is still readable while messages with no conversation are
unreachable garbage.

Indexes, created idempotently on every startup:

- `conversations {updated_at: -1, _id: -1}` — the sidebar's sort order.
- `messages {conversation_id: 1, created_at: 1, _id: 1}` — the history read. Its prefixes
  `{conversation_id: 1}` and `{conversation_id: 1, created_at: 1}` also serve the cascade
  delete, so there is no second index.

**Why `_id` is in both of those.** BSON datetimes are millisecond-precision, and a user's
message and its reply routinely land inside the *same* millisecond — measured here at 32
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
| `GET /api/health` | **Always 200.** Reports `status`, `mongo`, and `llm: {provider, configured, model, degraded_reason}`. Three consumers: Docker's HEALTHCHECK, the UI banner, and a reviewer with one curl. A 503 here would turn a keyless clone into a flapping container — the opposite of what the brief asks for. |
| `POST /api/conversations` | 201. |
| `GET /api/conversations` | Sorted by `updated_at` desc; the projection never drags whole transcripts along. |
| `GET/PATCH/DELETE /api/conversations/{id}` | `PATCH` renames — it exists so that heuristic titling reads as a choice rather than a limitation. `DELETE` cascades to messages first, then 204. |
| `GET /api/conversations/{id}/messages` | Includes `failed` and `interrupted` messages. History is honest about what happened. |
| `POST /api/conversations/{id}/messages` | **One generator, two renderers.** `Accept: text/event-stream` gets SSE; anything else gets JSON with a real HTTP status. Streaming for the product, plain JSON for Swagger and curl, zero duplicated lifecycle logic. |
| `POST /api/conversations/{id}/retry` | Drops a trailing `failed` or `interrupted` reply and re-enters the same generator. The working answer to "the reviewer's key hit 50/day mid-demo". |
| `GET /`, `/docs` | The UI and generated API docs. |

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
status code — same lifecycle, same persistence, different renderer.

## Configuration

Every variable has a working default except the API key. `tests/test_config.py` introspects
`Settings.model_fields` and fails the build if a new setting is added without one, or if it
isn't documented in `.env.example` — about fifteen lines that prevent the hour-nine
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
| `OLLAMA_MODEL` | `llama3.2` | |

Ollama gets its own base-URL and model rather than reusing `LLM_BASE_URL`/`LLM_MODEL`, so
that switching providers is one variable instead of three — and so an Ollama run can't
default to an OpenRouter model id that no local install has.

`LLM_PROVIDER` resolves **one-directionally: a present key always wins.** `auto` with a key
uses OpenRouter; `auto` without one uses the demo provider; `demo` always uses the demo
provider even if a key is present; `openrouter` always uses OpenRouter and fails per-turn
with a named error if no key is set; `ollama` always uses the local endpoint and needs no
key at all. The app never silently prefers the fake, and `auto` never reaches for a local
model on its own — a reviewer who suspects the integration was faked has already failed the
submission.

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
| MongoDB unreachable | — | 200 on `/api/health` | — |

Every error response carries the same envelope, including a `request_id` that is echoed in
the `X-Request-Id` header — so a red bubble in the browser can be grepped straight out of
`docker compose logs`:

```json
{"error": {"code": "rate_limited",
           "message": "The AI provider's free-tier rate limit was reached. Retry after 17s.",
           "request_id": "b1e4…", "retry_after_seconds": 17}}
```

On the SSE path a failure arrives as an `error` **event**, not an HTTP status — by then
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
  quota to produce the same error the user sees anyway — and retrying mid-stream would
  duplicate content. The 429 test asserts the upstream was called exactly once.
- **LLM-generated conversation titles.** Spending one of a reviewer's 50 daily requests on a
  sidebar label is the wrong trade. Titles are the first message, truncated on a word
  boundary.
- **Markdown rendering in the UI.** The frontend sets `textContent`, never `innerHTML`. That
  is an XSS decision about the one component nobody is grading, not an omission.
- **A cost ledger, `/stats` percentiles, an eval harness, detachable streams that survive the
  HTTP request, a ULID scheme, cursor pagination, an ADR directory.** All considered, all cut
  as ceremony at this size. Detachable streams in particular are correct at exactly one
  worker and are genuinely a multi-day feature, not an afternoon's.

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
  liveness probe in the conventional way, and it is the right call here — the container is
  genuinely up and serving when Mongo is down, and it should say what's wrong rather than
  disappear.

## How I used AI

I used Claude Code throughout — for planning, implementation, and adversarial review. The
useful thing to report is not that I used it, but the four places where I overrode it, since
each is checkable.

### 1. Motor → `pymongo.AsyncMongoClient`

Every model I asked reached for `motor` as the async MongoDB driver. Motor reached
**end-of-life on 14 May 2026**; the supported async driver is `AsyncMongoClient`, built into
PyMongo 4.9+. This repo uses it. MongoDB's own migration guide confirms it in about twenty
seconds — worth checking, because it's the clearest example of confidently-stated,
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
disconnect doesn't do that — it tears down the SSE renderer *wrapped around* the turn
generator, and `async for` abandons its iterator when the loop exits by exception, leaving
the `finally` to whenever the garbage collector gets around to finalizing an orphaned async
generator. Driving 16 real mid-stream disconnects against the running container:

- **10–13 of 16 lost the assistant message entirely.** No row, no error, no log line.
- The survivors landed hundreds of milliseconds late.
- Closing the generators explicitly fixed some of it and introduced
  `RuntimeError: aclose(): asynchronous generator is already running` — two teardown paths
  racing to close the same generator.

Three successive patches at the cleanup layer each moved the number without fixing it, which
was the signal that the layer was wrong. Durability cannot depend on async-generator
finalization, so it no longer does: the row is written up front, already reading
`interrupted`, and every later write only improves the record. **0 of 24 disconnects lose the
message now**, and the regression test drives the real teardown path — closing the SSE
renderer — rather than the cancellation shape that gave a false pass.

I'd rather submit this with the history described than with the elegant version that silently
dropped a message in half the cases the README's own headline sentence is about.

---

## Development

```bash
make up      # docker compose up --build
make test    # the suite, hermetic, no API key required
make lint    # ruff check
make fmt     # ruff format
make down    # stop (keeps the volume)
```

The suite runs against a real MongoDB — a throwaway database per test, dropped on teardown —
because the Repository boundary is exactly where a fake would stop proving anything. The LLM
is faked at the `LLMClient` Protocol, and `tests/test_llm_openai_compatible.py` goes one better by
faking only the HTTP transport, so the exception mapping is verified against the real
`openai` SDK's parsing rather than an assumption about it. An autouse fixture makes any real
outbound HTTP request raise, and a test asserts that guard actually fires — so "green with no
API key" means the suite genuinely never left the machine.
