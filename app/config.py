from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongo_uri: str = "mongodb://mongo:27017"
    mongo_db: str = "chatbot"

    llm_provider: Literal["auto", "openrouter", "demo"] = "auto"
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemma-4-26b-a4b-it:free"

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def blank_key_means_not_configured(cls, value: object) -> object:
        # LLM_API_KEY= (set but empty) is how compose renders an unset host
        # variable — treat it the same as not setting the variable at all.
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @property
    def llm_configured(self) -> bool:
        return self.llm_api_key is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
