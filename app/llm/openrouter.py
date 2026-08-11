"""The real LLM client. Its job beyond the network call is mapping every
`openai` SDK exception into one of app/errors.py's AppError types — chat.py
never sees an `openai` exception directly.

One thing worth writing down: OpenRouter's mid-stream error extension (a
chunk with a top-level "error" key, e.g. after a 200 OK once the free-tier
quota runs out) looks like it needs manual inspection of
`chunk.choices[0].finish_reason == "error"`. It doesn't — the openai SDK's
own stream reader (`AsyncStream.__stream__`) already special-cases any
chunk with a truthy top-level "error" key and raises a plain `openai.APIError`
*before* it even tries to validate the chunk against its schema. That
validation would fail anyway: `finish_reason` is a strict Literal that does
not include "error". So the only exception handling this needs is one
`except openai.APIError` around the iteration loop — verified against the
real SDK source and an httpx.MockTransport, not assumed from the docs.
"""

from collections.abc import AsyncIterator

import httpx
import openai

from app.errors import InvalidApiKey, RateLimited, UpstreamError
from app.llm.base import LLMChunk


class OpenRouterLLMClient:
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
            # We own the retry policy (see chat.py / README decisions): the
            # SDK's blind backoff on 429 would spend a second unit of a
            # 50-per-day quota to produce an error the user sees anyway.
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://github.com/chmbrs/chatbot-fastapi-mongo",
                "X-Title": "chatbot-fastapi-mongo",
            },
            # Testing seam: tests inject a MockTransport-backed client here
            # instead of mocking `chat.completions.create` directly, so the
            # exception mapping below is verified against the real SDK.
            http_client=http_client,
        )

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[LLMChunk]:
        try:
            response_stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                # OpenRouter now includes usage in every response automatically;
                # this is redundant but harmless back-compat, kept explicit.
                stream_options={"include_usage": True},
            )
        except openai.AuthenticationError as exc:
            raise InvalidApiKey() from exc
        except openai.RateLimitError as exc:
            raise RateLimited(retry_after_seconds=_retry_after_seconds(exc)) from exc
        except openai.APIStatusError as exc:
            raise UpstreamError(detail=f"HTTP {exc.status_code}") from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise UpstreamError(detail="could not reach the provider") from exc

        try:
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                usage = chunk.usage.model_dump() if chunk.usage else None
                if delta or usage:
                    yield LLMChunk(text=delta, usage=usage)
        except openai.RateLimitError as exc:
            raise RateLimited(retry_after_seconds=_retry_after_seconds(exc)) from exc
        except openai.APIStatusError as exc:
            raise UpstreamError(detail=f"HTTP {exc.status_code}") from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise UpstreamError(detail="the connection was interrupted") from exc
        except openai.APIError as exc:
            # The mid-stream data-level error case described above.
            raise UpstreamError(detail=exc.message) from exc


def _retry_after_seconds(exc: openai.RateLimitError) -> int | None:
    if exc.response is None:
        return None
    value = exc.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
