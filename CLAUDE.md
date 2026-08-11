# CLAUDE.md

Instructions for any AI assistant (or human) working in this repo.

## What this is

A chatbot: FastAPI backend, MongoDB persistence, vanilla HTML/JS frontend, an LLM
behind an OpenAI-compatible interface (OpenRouter free tier by default). Built as a
take-home challenge. The full design rationale lives in `README.md` — read it before
changing architecture, not just before changing code.

## The one rule everything else serves

Every turn ends in a state the server chose and persisted: `complete`, `interrupted`,
or `failed`. Nothing FastAPI does should ever surface a raw stack trace to the client —
map it to `AppError` and render it through the single error envelope. If you're adding
code and you can't say which of the three terminal states it produces, that's a sign
the change needs more thought, not less.

## Conventions

- Python 3.13, managed with `uv`. Don't add a dependency without a one-line note in
  the README's decisions section on why.
- `app/repository.py` is the *only* module that imports `pymongo`. Keep it that way —
  it's what makes the rest of the app testable without a real database.
- `app/chat.py` must not import `fastapi` or `openai` directly — it depends only on
  the `LLMClient` Protocol in `app/llm/base.py`. That boundary is the test seam.
- Every setting in `app/config.py` needs a working default except the API key. If you
  add one without a default, `tests/test_config.py` will fail on purpose — that's the
  point, not a bug to silence.
- Never log message content or the API key. Structured, one line per request.
- Commit messages: conventional style (`feat:`, `fix:`, `test:`, `docs:`, `chore:`),
  one logical change per commit. This repo's history is meant to be read.
- Never commit `.env`. `.env.example` documents every variable with a comment.

## Running things

- `docker compose up` — full stack, no API key required (falls back to a demo LLM
  provider).
- `docker compose --profile test run --rm tests` — test suite, hermetic, no network.
- See `README.md` for the full quickstart and verification checklist.
