import asyncio
import logging

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.cost_policy import configuration_allowed
from app.llm.providers.cloudflare import CloudflareProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.groq import GroqProvider
from app.llm.providers.kilo import KiloProvider
from app.llm.providers.llm7 import LLM7Provider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)

VALID_PROVIDER_NAMES = frozenset(
    {
        "groq",
        "openrouter",
        "gemini",
        "cloudflare",
        "ollama",
        "kilo",
        "llm7",
        "openai",
    }
)


class ProviderRegistry:
    def __init__(self, providers: dict[str, BaseLLMProvider]) -> None:
        self._providers = providers

    def require_enabled(self, name: str) -> BaseLLMProvider:
        self._validate_name(name)
        provider = self._providers.get(name)
        if provider is None:
            raise ValueError(
                f"Configured primary provider '{name}' is disabled",
            )
        return provider

    def get_enabled(self, name: str | None) -> BaseLLMProvider | None:
        if name is None:
            return None
        self._validate_name(name)
        return self._providers.get(name)

    def is_enabled(self, name: str) -> bool:
        self._validate_name(name)
        return name in self._providers

    async def close(self) -> None:
        providers = list(self._providers.values())
        results = await asyncio.gather(
            *(provider.close() for provider in providers),
            return_exceptions=True,
        )
        for provider, result in zip(providers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Failed to close provider %s",
                    provider.name,
                )

    @staticmethod
    def _validate_name(name: str) -> None:
        if name not in VALID_PROVIDER_NAMES:
            valid_names = ", ".join(sorted(VALID_PROVIDER_NAMES))
            raise ValueError(
                f"Unknown LLM provider '{name}'. Valid providers: {valid_names}",
            )


async def create_provider_registry(settings: Settings) -> ProviderRegistry:
    ProviderRegistry._validate_name(settings.llm_primary_provider)
    if settings.llm_fallback_provider is not None:
        ProviderRegistry._validate_name(settings.llm_fallback_provider)
    if "openai" in {settings.llm_primary_provider, settings.llm_fallback_provider}:
        raise ValueError("OpenAI direct is rejected: ZERO-COST GATE FAILED")
    if settings.llm_primary_provider == "gemini":
        raise ValueError("Gemini cannot be configured as the generic primary")
    if settings.llm_fallback_provider == "gemini":
        raise ValueError("Gemini cannot be configured as the generic fallback")
    if settings.llm_primary_provider == "cloudflare":
        raise ValueError("Cloudflare cannot be configured as the generic primary")
    if settings.llm_fallback_provider == "cloudflare":
        raise ValueError("Cloudflare cannot be configured as the generic fallback")
    if settings.llm_primary_provider == "ollama":
        raise ValueError("Ollama cannot be configured as the generic primary")
    if settings.llm_fallback_provider == "ollama":
        raise ValueError("Ollama cannot be configured as the generic fallback")
    if settings.llm_primary_provider == "kilo":
        raise ValueError("Kilo cannot be configured as the generic primary")
    if settings.llm_fallback_provider == "kilo":
        raise ValueError("Kilo cannot be configured as the generic fallback")
    if settings.llm_primary_provider == "llm7":
        raise ValueError("LLM7 cannot be configured as the generic primary")
    if settings.llm_fallback_provider == "llm7":
        raise ValueError("LLM7 cannot be configured as the generic fallback")

    providers: dict[str, BaseLLMProvider] = {}
    try:
        candidates = (
            ("groq", GroqProvider, True),
            ("openrouter", OpenRouterProvider, bool(settings.openrouter_api_key)),
            ("gemini", GeminiProvider, bool(settings.gemini_api_key)),
            (
                "cloudflare",
                CloudflareProvider,
                bool(settings.cloudflare_api_token and settings.cloudflare_account_id),
            ),
            ("ollama", OllamaProvider, bool(settings.ollama_api_key)),
            ("kilo", KiloProvider, settings.kilo_enabled),
            (
                "llm7",
                LLM7Provider,
                bool(settings.llm7_enabled and settings.llm7_api_key),
            ),
            ("openai", OpenAIProvider, settings.openai_api_key is not None),
        )
        for name, factory, enabled in candidates:
            if not enabled:
                continue
            if not configuration_allowed(settings, name):
                if name == settings.llm_primary_provider:
                    raise ValueError(
                        "Primary provider failed the mandatory zero-cost gate"
                    )
                continue
            providers[name] = factory(settings=settings)
    except Exception:
        await ProviderRegistry(providers).close()
        raise

    return ProviderRegistry(providers)
