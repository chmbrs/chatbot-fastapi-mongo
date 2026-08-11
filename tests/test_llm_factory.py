import pytest

from app.config import Settings
from app.errors import LLMNotConfigured
from app.llm import build_llm
from app.llm.demo import DemoLLMClient
from app.llm.openrouter import OpenRouterLLMClient


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_auto_with_no_key_resolves_to_demo():
    client = build_llm(_settings(llm_provider="auto"))
    assert isinstance(client, DemoLLMClient)


def test_auto_with_a_key_resolves_to_openrouter():
    client = build_llm(_settings(llm_provider="auto", llm_api_key="sk-or-test"))
    assert isinstance(client, OpenRouterLLMClient)


def test_demo_wins_even_if_a_key_is_present():
    client = build_llm(_settings(llm_provider="demo", llm_api_key="sk-or-test"))
    assert isinstance(client, DemoLLMClient)


async def test_openrouter_explicit_with_no_key_fails_per_turn_not_at_boot():
    client = build_llm(_settings(llm_provider="openrouter"))  # must not raise here

    with pytest.raises(LLMNotConfigured):
        async for _ in client.stream([{"role": "user", "content": "hi"}]):
            pass


def test_openrouter_explicit_with_a_key_resolves_to_openrouter():
    client = build_llm(_settings(llm_provider="openrouter", llm_api_key="sk-or-test"))
    assert isinstance(client, OpenRouterLLMClient)
