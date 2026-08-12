"""These tests hit the real openai SDK's streaming/error-handling code —
only the HTTP transport is faked, via httpx.MockTransport. That's a
deliberately higher bar than mocking `client.chat.completions.create`
directly: it proves the exception mapping in openrouter.py works against
how the SDK actually parses OpenRouter's wire format, not against an
assumption of it. See openrouter.py's module docstring for what that caught.
"""

import httpx
import pytest

from app.errors import InvalidApiKey, RateLimited, UpstreamError
from app.llm.openrouter import OpenRouterLLMClient


def _client(handler) -> OpenRouterLLMClient:
    return OpenRouterLLMClient(
        api_key="test-key",
        base_url="https://example.com",
        model="m",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _sse(*lines: str) -> httpx.Response:
    body = "".join(f"data: {line}\n\n" for line in lines)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


async def _collect(client, messages):
    return [chunk async for chunk in client.stream(messages)]


async def test_successful_stream_yields_text_and_final_usage():
    def handler(request):
        return _sse(
            '{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":""},'
            '"finish_reason":null}]}',
            '{"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            '{"id":"1","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}',
            '{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}',
            "[DONE]",
        )

    chunks = await _collect(_client(handler), [{"role": "user", "content": "hi"}])

    assert "".join(c.text for c in chunks) == "Hello there"
    assert chunks[-1].usage == {
        "completion_tokens": 2,
        "prompt_tokens": 10,
        "total_tokens": 12,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    }


async def test_midstream_error_chunk_raises_upstream_error_after_partial_text():
    def handler(request):
        return _sse(
            '{"id":"1","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}',
            '{"id":"1","choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}],'
            '"error":{"code":"server_error","message":"Provider disconnected unexpectedly"}}',
        )

    client = _client(handler)
    received = []
    with pytest.raises(UpstreamError) as exc_info:
        async for chunk in client.stream([{"role": "user", "content": "hi"}]):
            received.append(chunk.text)

    assert received == ["partial"]
    assert "Provider disconnected unexpectedly" in exc_info.value.message


async def test_401_raises_invalid_api_key():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    with pytest.raises(InvalidApiKey):
        await _collect(_client(handler), [{"role": "user", "content": "hi"}])


async def test_429_raises_rate_limited_with_retry_after_seconds_and_does_not_retry():
    """`max_retries=0` is a decision, not a default (see openrouter.py): on a
    50-requests-per-day free tier, the SDK's blind backoff would spend a second
    unit of the quota to produce the same error the user sees anyway. The call
    count is the assertion that pins it — without it this test passes just as
    happily if the SDK quietly retries twice more.
    """
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            429, headers={"retry-after": "17"}, json={"error": {"message": "slow down"}}
        )

    with pytest.raises(RateLimited) as exc_info:
        await _collect(_client(handler), [{"role": "user", "content": "hi"}])

    assert exc_info.value.retry_after_seconds == 17
    assert attempts == 1


async def test_402_no_credits_raises_upstream_error_naming_the_status():
    def handler(request):
        return httpx.Response(402, json={"error": {"message": "insufficient credits"}})

    with pytest.raises(UpstreamError) as exc_info:
        await _collect(_client(handler), [{"role": "user", "content": "hi"}])

    assert "402" in exc_info.value.message


async def test_connection_failure_raises_upstream_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(UpstreamError):
        await _collect(_client(handler), [{"role": "user", "content": "hi"}])


async def test_the_suite_cannot_reach_the_network():
    """Guards the guard. Every test above fakes only the transport, so the one
    way this file could start silently spending a reviewer's free-tier quota is
    an OpenRouterLLMClient built without a MockTransport. conftest.py's autouse
    fixture makes that raise instead — this is the test that fails if that
    fixture is ever removed or renamed, rather than the suite quietly going
    live.
    """
    async with httpx.AsyncClient() as unguarded:
        with pytest.raises(RuntimeError, match="real outbound HTTP request"):
            await unguarded.get("https://openrouter.ai/api/v1/models")
