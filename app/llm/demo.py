"""The offline provider: no network calls, fully deterministic. Demo mode
announces itself in five places, because a reviewer who suspects the
integration was faked has already failed the submission: this reply text,
a startup warning in the container logs (main.py), the `provider` field
stored on every message document, /api/health's degraded_reason, and the
amber banner in the UI. Deliberately boring: no keyword matching, no jokes.
The moment this is clever, it's a gimmick and the repo reads as a toy.
"""

import asyncio
from collections.abc import AsyncIterator

from app.llm.base import LLMChunk

_REPLY = (
    "This is the offline demo provider — no LLM_API_KEY is configured, so there's no "
    "real model behind this reply. Your last message had {length} characters. "
    "See the README for how to get a free OpenRouter key and talk to a real model."
)


class DemoLLMClient:
    name = "demo"
    model = "demo"

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[LLMChunk]:
        last_user_length = next(
            (len(m["content"]) for m in reversed(messages) if m["role"] == "user"), 0
        )
        reply = _REPLY.format(length=last_user_length)
        for word in reply.split(" "):
            yield LLMChunk(text=word + " ")
            await asyncio.sleep(0)  # yield control, not a typing-speed simulation
