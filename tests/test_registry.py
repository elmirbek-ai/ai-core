import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.llm.registry import ProviderRegistry, create_provider_registry


def test_unknown_provider_is_configuration_error() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        llm_primary_provider="mistral",
    )

    with pytest.raises(ValueError, match="Unknown LLM provider 'mistral'"):
        asyncio.run(create_provider_registry(settings))


def test_openrouter_without_api_key_is_disabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key=None,
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert registry.is_enabled("groq")
    assert not registry.is_enabled("openrouter")
    assert registry.get_enabled("openrouter") is None


def test_gemini_with_api_key_is_enabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    gemini = SimpleNamespace(name="gemini", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.GeminiProvider",
        lambda settings: gemini,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        gemini_api_key="gemini-test-key",
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert registry.is_enabled("gemini")
    assert registry.get_enabled("gemini") is gemini


def test_gemini_without_api_key_is_disabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        gemini_api_key=None,
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("gemini")
    assert registry.get_enabled("gemini") is None


def test_cloudflare_with_token_and_account_id_is_enabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    cloudflare = SimpleNamespace(name="cloudflare", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.CloudflareProvider",
        lambda settings: cloudflare,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        cloudflare_api_token="cloudflare-test-token",
        cloudflare_account_id="test-account-id",
        cloudflare_zero_cost_verified=True,
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert registry.is_enabled("cloudflare")
    assert registry.get_enabled("cloudflare") is cloudflare


@pytest.mark.parametrize(
    ("token", "account_id"),
    [
        ("cloudflare-test-token", None),
        (None, "test-account-id"),
        (None, None),
    ],
)
def test_cloudflare_with_incomplete_credentials_is_disabled(
    monkeypatch,
    token: str | None,
    account_id: str | None,
) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    cloudflare_type = AsyncMock()
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.CloudflareProvider",
        cloudflare_type,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        cloudflare_api_token=token,
        cloudflare_account_id=account_id,
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("cloudflare")
    assert registry.get_enabled("cloudflare") is None
    cloudflare_type.assert_not_called()


def test_ollama_with_api_key_remains_unverified_and_disabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    ollama = SimpleNamespace(name="ollama", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.OllamaProvider",
        lambda settings: ollama,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        ollama_api_key="ollama-test-key",
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("ollama")
    assert registry.get_enabled("ollama") is None


def test_ollama_without_api_key_is_disabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    ollama_type = AsyncMock()
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.OllamaProvider",
        ollama_type,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        ollama_api_key="  ",
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("ollama")
    ollama_type.assert_not_called()


def test_kilo_is_disabled_by_default(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    kilo_type = AsyncMock()
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr("app.llm.registry.KiloProvider", kilo_type)
    settings = Settings(_env_file=None, groq_api_key="test-key")

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("kilo")
    kilo_type.assert_not_called()


def test_kilo_is_enabled_only_with_explicit_opt_in(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    kilo = SimpleNamespace(name="kilo", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.KiloProvider",
        lambda settings: kilo,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        kilo_enabled=True,
        kilo_general_model="kilo-auto/free",
        kilo_code_model="kilo-auto/free",
        kilo_long_context_model="kilo-auto/free",
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert registry.is_enabled("kilo")
    assert registry.get_enabled("kilo") is kilo


def test_llm7_is_disabled_by_default(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    llm7_type = AsyncMock()
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr("app.llm.registry.LLM7Provider", llm7_type)
    settings = Settings(_env_file=None, groq_api_key="test-key")

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("llm7")
    llm7_type.assert_not_called()


def test_llm7_is_enabled_with_opt_in_and_key(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    llm7 = SimpleNamespace(name="llm7", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        "app.llm.registry.LLM7Provider",
        lambda settings: llm7,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        llm7_enabled=True,
        llm7_api_key="llm7-test-key",
        llm7_zero_cost_verified=True,
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert registry.is_enabled("llm7")
    assert registry.get_enabled("llm7") is llm7


def test_llm7_opt_in_without_key_remains_disabled(monkeypatch) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    llm7_type = AsyncMock()
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr("app.llm.registry.LLM7Provider", llm7_type)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        llm7_enabled=True,
        llm7_api_key="  ",
    )

    registry = asyncio.run(create_provider_registry(settings))

    assert not registry.is_enabled("llm7")
    llm7_type.assert_not_called()


def test_registry_closes_all_enabled_providers() -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    openrouter = SimpleNamespace(name="openrouter", close=AsyncMock())
    gemini = SimpleNamespace(name="gemini", close=AsyncMock())
    cloudflare = SimpleNamespace(name="cloudflare", close=AsyncMock())
    ollama = SimpleNamespace(name="ollama", close=AsyncMock())
    kilo = SimpleNamespace(name="kilo", close=AsyncMock())
    llm7 = SimpleNamespace(name="llm7", close=AsyncMock())
    registry = ProviderRegistry(
        {
            "groq": groq,
            "openrouter": openrouter,
            "gemini": gemini,
            "cloudflare": cloudflare,
            "ollama": ollama,
            "kilo": kilo,
            "llm7": llm7,
        },
    )

    asyncio.run(registry.close())

    groq.close.assert_awaited_once()
    openrouter.close.assert_awaited_once()
    gemini.close.assert_awaited_once()
    cloudflare.close.assert_awaited_once()
    ollama.close.assert_awaited_once()
    kilo.close.assert_awaited_once()
    llm7.close.assert_awaited_once()


def test_all_enabled_providers_initialize_once_and_close_once(monkeypatch) -> None:
    created: dict[str, SimpleNamespace] = {}

    def factory(name: str):
        def create(settings):
            del settings
            provider = SimpleNamespace(name=name, close=AsyncMock())
            created[name] = provider
            return provider

        return create

    provider_types = {
        "GroqProvider": "groq",
        "OpenRouterProvider": "openrouter",
        "GeminiProvider": "gemini",
        "CloudflareProvider": "cloudflare",
        "OllamaProvider": "ollama",
        "KiloProvider": "kilo",
        "LLM7Provider": "llm7",
    }
    for type_name, provider_name in provider_types.items():
        monkeypatch.setattr(
            f"app.llm.registry.{type_name}",
            factory(provider_name),
        )

    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="test-key",
        gemini_api_key="test-key",
        cloudflare_api_token="test-token",
        cloudflare_account_id="test-account",
        cloudflare_zero_cost_verified=True,
        ollama_api_key="test-key",
        kilo_enabled=True,
        kilo_general_model="kilo-auto/free",
        kilo_code_model="kilo-auto/free",
        kilo_long_context_model="kilo-auto/free",
        llm7_enabled=True,
        llm7_api_key="test-key",
        llm7_zero_cost_verified=True,
    )

    registry = asyncio.run(create_provider_registry(settings))
    asyncio.run(registry.close())

    assert set(created) == set(provider_types.values()) - {"ollama"}
    assert all(registry.is_enabled(name) for name in created)
    for provider in created.values():
        provider.close.assert_awaited_once()


def test_registry_cleanup_continues_after_close_failure(caplog) -> None:
    first = SimpleNamespace(
        name="groq",
        close=AsyncMock(side_effect=RuntimeError("private close detail")),
    )
    second = SimpleNamespace(name="openrouter", close=AsyncMock())
    registry = ProviderRegistry({"groq": first, "openrouter": second})

    asyncio.run(registry.close())

    first.close.assert_awaited_once()
    second.close.assert_awaited_once()
    assert "private close detail" not in caplog.text


def test_repeated_registry_cleanup_remains_safe() -> None:
    provider = SimpleNamespace(name="groq", close=AsyncMock())
    registry = ProviderRegistry({"groq": provider})

    async def exercise() -> None:
        await registry.close()
        await registry.close()

    asyncio.run(exercise())

    assert provider.close.await_count == 2


def test_partial_registry_startup_failure_closes_created_providers(
    monkeypatch,
) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )

    def fail_openrouter(settings):
        raise RuntimeError("constructor failed")

    monkeypatch.setattr(
        "app.llm.registry.OpenRouterProvider",
        fail_openrouter,
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="openrouter-test-key",
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        asyncio.run(create_provider_registry(settings))

    groq.close.assert_awaited_once()


def test_kilo_startup_failure_closes_previously_created_providers(
    monkeypatch,
) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )

    def fail_kilo(settings):
        raise RuntimeError("constructor failed")

    monkeypatch.setattr("app.llm.registry.KiloProvider", fail_kilo)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        kilo_enabled=True,
        kilo_general_model="kilo-auto/free",
        kilo_code_model="kilo-auto/free",
        kilo_long_context_model="kilo-auto/free",
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        asyncio.run(create_provider_registry(settings))

    groq.close.assert_awaited_once()


def test_llm7_startup_failure_closes_previously_created_providers(
    monkeypatch,
) -> None:
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider",
        lambda settings: groq,
    )

    def fail_llm7(settings):
        raise RuntimeError("constructor failed")

    monkeypatch.setattr("app.llm.registry.LLM7Provider", fail_llm7)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        llm7_enabled=True,
        llm7_api_key="llm7-test-key",
        llm7_zero_cost_verified=True,
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        asyncio.run(create_provider_registry(settings))

    groq.close.assert_awaited_once()
