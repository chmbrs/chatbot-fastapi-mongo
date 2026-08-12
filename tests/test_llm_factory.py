import pytest

from app.config import Settings
from app.errors import LLMNotConfigured
from app.llm import build_llm
from app.llm.demo import DemoLLMClient
from app.llm.openai_compatible import OpenAICompatibleLLMClient


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_auto_with_no_key_resolves_to_demo():
    client = build_llm(_settings(llm_provider="auto"))
    assert isinstance(client, DemoLLMClient)


def test_auto_with_a_key_resolves_to_openrouter():
    client = build_llm(_settings(llm_provider="auto", llm_api_key="sk-or-test"))
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.name == "openrouter"


def test_ollama_needs_no_key_and_is_not_degraded():
    """The one provider where "no API key" is a complete configuration rather
    than a missing one — that's the whole reason it's worth offering."""
    settings = _settings(llm_provider="ollama")
    client = build_llm(settings)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.name == "ollama"
    assert client.model == settings.ollama_model
    assert settings.llm_api_key is None
    assert settings.llm_configured is True


def test_ollama_is_never_selected_implicitly():
    """`auto` resolves to OpenRouter-or-demo and never probes for a local
    Ollama. Silently sending a conversation to a different model than the one
    configured is the same failure as silently falling back to the fake — the
    user has to ask for it."""
    assert build_llm(_settings(llm_provider="auto")).name == "demo"
    assert build_llm(_settings(llm_provider="auto", llm_api_key="sk-or-test")).name == "openrouter"


def test_ollama_does_not_borrow_the_openrouter_model_id():
    """Separate settings, so switching providers is one variable. Sharing
    LLM_MODEL would point an Ollama run at `google/gemma-4-26b-a4b-it:free`,
    which no local install has."""
    client = build_llm(_settings(llm_provider="ollama", llm_model="google/some-remote-model:free"))
    assert client.model == _settings().ollama_model


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
    assert isinstance(client, OpenAICompatibleLLMClient)
