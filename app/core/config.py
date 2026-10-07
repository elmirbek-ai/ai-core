from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Core"
    app_version: str = "0.1.0"

    ai_core_auth_enabled: bool = True
    ai_core_api_key: SecretStr | None = None

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
    openrouter_supports_images: bool = False
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

    ollama_api_key: str | None = None
    ollama_base_url: str = Field(default="https://ollama.com", min_length=1)
    ollama_model: str = "gpt-oss:120b"
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)
    ollama_max_retries: int = Field(default=2, ge=0)

    # Kilo is an anonymous shared gateway, so explicit opt-in is required.
    kilo_enabled: bool = False
    kilo_base_url: str = Field(
        default="https://api.kilo.ai/api/gateway",
        min_length=1,
    )
    kilo_general_model: str = Field(
        default="stepfun/step-3.7-flash:free",
        min_length=1,
    )
    kilo_code_model: str = Field(
        default="cohere/north-mini-code:free",
        min_length=1,
    )
    kilo_long_context_model: str = Field(
        default="dots-studio/dots-3-note-preview:free",
        min_length=1,
    )
    kilo_timeout_seconds: float = Field(default=60.0, gt=0)
    kilo_max_retries: int = Field(default=1, ge=0)

    llm7_enabled: bool = False
    llm7_api_key: str | None = None
    llm7_base_url: str = Field(default="https://api.llm7.io/v1", min_length=1)
    llm7_general_model: str = Field(default="gpt-oss:20b", min_length=1)
    llm7_reasoning_model: str = Field(default="gpt-oss:20b", min_length=1)
    llm7_code_model: str = Field(default="gpt-oss:20b", min_length=1)
    llm7_timeout_seconds: float = Field(default=60.0, gt=0)
    llm7_max_retries: int = Field(default=1, ge=0)

    llm_circuit_breaker_enabled: bool = True
    llm_circuit_failure_threshold: int = Field(default=3, ge=1)
    llm_circuit_cooldown_seconds: float = Field(default=60.0, ge=0)

    llm_request_budget_enabled: bool = True
    llm_standard_budget_seconds: float = Field(default=30.0, gt=0)
    llm_reasoning_budget_seconds: float = Field(default=60.0, gt=0)
    llm_code_budget_seconds: float = Field(default=60.0, gt=0)
    llm_long_context_budget_seconds: float = Field(default=90.0, gt=0)
    llm_multimodal_budget_seconds: float = Field(default=90.0, gt=0)

    llm_telemetry_enabled: bool = True

    llm_provider_concurrency_enabled: bool = True
    llm_provider_default_max_concurrency: int = Field(default=4, ge=1)
    llm_groq_max_concurrency: int = Field(default=4, ge=1)
    llm_openrouter_max_concurrency: int = Field(default=2, ge=1)
    llm_gemini_max_concurrency: int = Field(default=3, ge=1)
    llm_cloudflare_max_concurrency: int = Field(default=3, ge=1)
    llm_ollama_max_concurrency: int = Field(default=2, ge=1)
    llm_kilo_max_concurrency: int = Field(default=1, ge=1)
    llm_llm7_max_concurrency: int = Field(default=1, ge=1)

    llm_auto_long_context_chars: int = Field(default=12_000, gt=0)

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
        "ollama_api_key",
        "llm7_api_key",
        "ai_core_api_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator(
        "ollama_base_url",
        "kilo_base_url",
        "kilo_general_model",
        "kilo_code_model",
        "kilo_long_context_model",
        "llm7_base_url",
        "llm7_general_model",
        "llm7_reasoning_model",
        "llm7_code_model",
        mode="before",
    )
    @classmethod
    def normalize_non_empty_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
