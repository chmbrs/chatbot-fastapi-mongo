"""A fake LLM for testing app/chat.py without any network calls. This is
the one seam meant for fakes — see app/llm/base.py's docstring on why the
Repository boundary is tested against a real Mongo instead.
"""

from collections.abc import AsyncIterator

from app.llm.base import LLMChunk


class FakeLLMClient:
    name = "fake"
    model = "fake-model"

    def __init__(self, chunks: list[LLMChunk] | None = None, error: Exception | None = None):
        self._chunks = chunks or []
        self._error = error
        self.calls: list[list[dict[str, str]]] = []

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[LLMChunk]:
        self.calls.append(messages)
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error
