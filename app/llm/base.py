"""The one seam between app/chat.py and any real network call. It earns its
few lines three times: it's the test seam (fakes implement this instead of
mocking `openai`), the zero-key degradation seam (demo.py implements it too),
and the thing that shows dependency inversion at a glance.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMChunk:
    text: str
    # Only ever set on the final chunk, and only if the provider reports it.
    usage: dict[str, int] | None = None


class LLMClient(Protocol):
    name: str
    model: str

    def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[LLMChunk]:
        """`messages` is OpenAI-style: [{"role": "user"|"assistant", "content": str}, ...]."""
        ...
