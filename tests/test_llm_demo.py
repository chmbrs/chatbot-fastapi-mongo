from app.llm.demo import DemoLLMClient


async def _collect(chunks):
    return "".join([chunk.text async for chunk in chunks])


async def test_demo_streams_multiple_chunks():
    client = DemoLLMClient()
    chunks = [c async for c in client.stream([{"role": "user", "content": "hi"}])]

    assert len(chunks) > 1
    assert all(c.usage is None for c in chunks)


async def test_demo_reply_mentions_no_api_key_configured():
    client = DemoLLMClient()
    reply = await _collect(client.stream([{"role": "user", "content": "hello there"}]))

    assert "LLM_API_KEY" in reply
    assert "demo" in reply.lower()


async def test_demo_is_deterministic_for_the_same_input():
    client = DemoLLMClient()
    messages = [{"role": "user", "content": "same input twice"}]

    first = await _collect(client.stream(messages))
    second = await _collect(client.stream(messages))

    assert first == second


async def test_demo_ignores_assistant_messages_when_finding_last_user_message():
    client = DemoLLMClient()
    messages = [
        {"role": "user", "content": "12345"},
        {"role": "assistant", "content": "a much longer assistant reply than the user's"},
    ]

    reply = await _collect(client.stream(messages))

    assert "had 5 characters" in reply
