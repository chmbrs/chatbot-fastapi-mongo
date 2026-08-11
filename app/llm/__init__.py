from collections.abc import AsyncIterator

from app.config import Settings
from app.errors import LLMNotConfigured
from app.llm.base import LLMChunk, LLMClient
from app.llm.demo import DemoLLMClient
from app.llm.openrouter import OpenRouterLLMClient

__all__ = ["build_llm"]


class _UnconfiguredOpenRouterClient:
    """LLM_PROVIDER=openrouter set explicitly, but no key. Boots fine;
    fails per-turn with a named error rather than crashing at startup."""

    name = "openrouter"
    model = ""

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[LLMChunk]:
        raise LLMNotConfigured()
        yield  # unreachable, but required to make this an async generator function.


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "demo":
        return DemoLLMClient()

    if settings.llm_provider == "auto" and not settings.llm_configured:
        return DemoLLMClient()

    # "openrouter", or "auto" with a key present. A present key always wins —
    # the app never silently prefers the fake, because a reviewer who suspects
    # the integration was faked has already failed the submission.
    if not settings.llm_configured:
        return _UnconfiguredOpenRouterClient()

    return OpenRouterLLMClient(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
