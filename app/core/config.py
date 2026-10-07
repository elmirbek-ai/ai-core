from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Core"
    app_version: str = "0.1.0"

    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"
    groq_fast_model: str = "openai/gpt-oss-20b"
    groq_reasoning_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = Field(default=30.0, gt=0)
    groq_max_retries: int = Field(default=2, ge=0)

    llm_primary_provider: str = Field(default="groq", min_length=1)
    llm_fallback_provider: str | None = "openrouter"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"
    openrouter_timeout_seconds: float = Field(default=30.0, gt=0)
    openrouter_max_retries: int = Field(default=2, ge=0)

    gemini_api_key: str | None = None
    gemini_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = Field(default=30.0, gt=0)
    gemini_max_retries: int = Field(default=2, ge=0)

    cloudflare_api_token: str | None = None
    cloudflare_account_id: str | None = None
    cloudflare_model: str = "cf/openai/gpt-oss-120b"
    cloudflare_timeout_seconds: float = Field(default=30.0, gt=0)
    cloudflare_max_retries: int = Field(default=2, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("llm_primary_provider", mode="before")
    @classmethod
    def normalize_primary_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("llm_fallback_provider", mode="before")
    @classmethod
    def normalize_fallback_provider(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized or None
        return value

    @field_validator(
        "openrouter_api_key",
        "gemini_api_key",
        "cloudflare_api_token",
        "cloudflare_account_id",
        mode="before",
    )
    @classmethod
    def normalize_optional_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
