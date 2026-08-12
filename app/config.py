from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # LLM_API_KEY= (set but empty) is how compose renders an unset host
        # variable — this falls back to the field's own default (None) for
        # every setting, same as if the variable were never set at all.
        env_ignore_empty=True,
    )

    mongo_uri: str = "mongodb://mongo:27017"
    mongo_db: str = "chatbot"

    llm_provider: Literal["auto", "openrouter", "ollama", "demo"] = "auto"
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemma-4-26b-a4b-it:free"

    # Ollama gets its own pair rather than reusing LLM_BASE_URL/LLM_MODEL:
    # switching providers should be one variable, not three, and a shared
    # LLM_MODEL would default an Ollama run to an OpenRouter model id.
    # host.docker.internal, not localhost — this app runs in a container and
    # Ollama runs on the host (compose maps the name on Linux too).
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    # A small, non-reasoning instruct model on purpose. Reasoning models
    # (qwen3.5, deepseek-r1) put their thinking in a separate `reasoning` field
    # and leave `content` empty until it finishes — over an OpenAI-compatible
    # stream that is indistinguishable from a hang, for as long as it thinks.
    ollama_model: str = "llama3.2"

    @property
    def llm_configured(self) -> bool:
        """Whether the *selected* provider has what it needs. Ollama runs
        locally and authenticates nothing, so for it the answer is always yes —
        which is the point of offering it: a fully working stack, no signup,
        no quota, no key on disk."""
        return self.llm_provider == "ollama" or self.llm_api_key is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
