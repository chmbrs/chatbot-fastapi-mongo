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

    llm_provider: Literal["auto", "openrouter", "demo"] = "auto"
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemma-4-26b-a4b-it:free"

    @property
    def llm_configured(self) -> bool:
        return self.llm_api_key is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
