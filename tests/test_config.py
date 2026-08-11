"""The config contract: a clean clone with no .env must boot. If a setting
is added later with no real default, this fails on purpose — that's the
whole point.
"""

from pathlib import Path

from app.config import Settings

ENV_EXAMPLE = (Path(__file__).parent.parent / ".env.example").read_text()

# The one field allowed to default to "not configured" rather than a real value.
ALLOWED_NONE_DEFAULT = {"llm_api_key"}


def test_every_setting_has_a_real_default_except_the_api_key():
    for name, field in Settings.model_fields.items():
        assert not field.is_required(), (
            f"{name} has no default at all — a clean clone with no .env would crash on startup"
        )
        if name in ALLOWED_NONE_DEFAULT:
            continue
        assert field.default is not None, (
            f"{name} defaults to None — a clean clone with no .env "
            "would be missing this value at runtime"
        )


def test_settings_boots_with_no_environment_configured(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.llm_configured is False


def test_blank_api_key_is_treated_as_not_configured(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.llm_configured is False


def test_present_api_key_is_configured(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test-key")

    settings = Settings(_env_file=None)

    assert settings.llm_configured is True
    assert settings.llm_api_key.get_secret_value() == "sk-or-test-key"


def test_every_setting_is_documented_in_env_example():
    for name in Settings.model_fields:
        assert name.upper() in ENV_EXAMPLE, f"{name.upper()} is missing from .env.example"
