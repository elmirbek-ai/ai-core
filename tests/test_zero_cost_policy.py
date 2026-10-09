import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config import Settings
from app.evaluation.loader import load_dataset
from app.evaluation.models import EvaluationError
from app.evaluation.targets import ExistingProviderTarget, create_live_target
from app.llm.cost_policy import (
    PROVIDER_ELIGIBILITY,
    ZeroCostEligibility,
    configuration_allowed,
    model_eligibility,
    require_zero_cost_model,
)
from app.llm.exceptions import LLMRateLimitError
from app.llm.registry import create_provider_registry
from app.llm.router import LLMRouter

SECRET = "TEST_SECRET_DO_NOT_LOG"


@pytest.mark.parametrize(
    ("provider", "model", "status"),
    [
        ("groq", "openai/gpt-oss-20b", "ELIGIBLE_FREE"),
        ("groq", "openai/gpt-oss-120b", "ELIGIBLE_FREE"),
        ("openrouter", "openrouter/free", "ELIGIBLE_FREE"),
        ("openrouter", "vendor/model:free", "ELIGIBLE_FREE"),
        ("openrouter", "openai/gpt-oss-120b", "NOT_VERIFIED"),
        ("openrouter", "vendor/model:free?key=" + SECRET, "NOT_VERIFIED"),
        ("openrouter", "nvidia/model:free", "NOT_VERIFIED"),
        ("gemini", "gemini-3.8-flash", "ELIGIBLE_FREE"),
        ("gemini", "paid-model", "NOT_VERIFIED"),
        ("cloudflare", "cf/openai/gpt-oss-120b", "CONDITIONAL_FREE"),
        ("cloudflare", "paid-model", "NOT_VERIFIED"),
        ("ollama", "gpt-oss:120b", "NOT_VERIFIED"),
        ("ollama", "starter-model", "NOT_VERIFIED"),
        ("kilo", "kilo-auto/free", "ELIGIBLE_FREE"),
        ("kilo", "stepfun/step-3.7-flash:free", "NOT_VERIFIED"),
        ("kilo", "cohere/north-mini-code:free", "NOT_VERIFIED"),
        ("kilo", "dots-studio/dots-3-note-preview:free", "NOT_VERIFIED"),
        ("kilo", "nvidia/model", "NOT_VERIFIED"),
        ("llm7", "gpt-oss:20b", "CONDITIONAL_FREE"),
        ("llm7", "paid-model", "NOT_VERIFIED"),
        ("openai", "gpt-6-luna", "REJECTED_PAID"),
        ("deepseek", "any", "REJECTED_PAID"),
        ("anthropic", "any", "REJECTED_PAID"),
        ("xai", "any", "REJECTED_PAID"),
        ("mistral", "any", "NOT_VERIFIED"),
        ("unknown", "any", "NOT_VERIFIED"),
        ("groq", None, "NOT_VERIFIED"),
    ],
)
def test_reviewed_model_cost_semantics(provider, model, status):
    assert model_eligibility(provider, model).value == status


def test_provider_decisions_are_detached_from_model_admission_and_immutable():
    assert PROVIDER_ELIGIBILITY["mistral"] == ZeroCostEligibility.CONDITIONAL_FREE
    assert PROVIDER_ELIGIBILITY["openai"] == ZeroCostEligibility.REJECTED_PAID
    with pytest.raises(TypeError):
        PROVIDER_ELIGIBILITY["openai"] = ZeroCostEligibility.ELIGIBLE_FREE


@pytest.mark.parametrize("provider", ["cloudflare", "llm7"])
def test_conditional_free_account_model_requires_explicit_verification(provider):
    model = "cf/openai/gpt-oss-120b" if provider == "cloudflare" else "gpt-oss:20b"
    with pytest.raises(ValueError, match="zero-cost policy"):
        require_zero_cost_model(provider, model)
    require_zero_cost_model(provider, model, free_mode_verified=True)
    with pytest.raises(ValueError):
        require_zero_cost_model(provider, "paid-model", free_mode_verified=True)


def test_unknown_configuration_and_secret_label_fail_closed():
    settings = Settings(_env_file=None, groq_api_key="test-key")
    assert not configuration_allowed(settings, "unknown")
    with pytest.raises(ValueError) as error:
        require_zero_cost_model("openrouter", SECRET)
    assert SECRET not in str(error.value)


def test_default_configuration_keeps_only_reviewed_free_models():
    settings = Settings(_env_file=None, groq_api_key="test-key")
    for name in ["groq", "openrouter", "gemini"]:
        assert configuration_allowed(settings, name)
    for name in ["cloudflare", "ollama", "kilo", "llm7", "openai"]:
        assert not configuration_allowed(settings, name)
    assert not settings.cloudflare_zero_cost_verified
    assert not settings.llm7_zero_cost_verified


@pytest.mark.parametrize(
    ("provider", "changes"),
    [
        (
            "openrouter",
            {"openrouter_api_key": SECRET, "openrouter_model": "paid-model"},
        ),
        ("gemini", {"gemini_api_key": SECRET, "gemini_model": "paid-model"}),
        (
            "cloudflare",
            {
                "cloudflare_api_token": SECRET,
                "cloudflare_account_id": "test",
                "cloudflare_zero_cost_verified": True,
                "cloudflare_model": "paid-model",
            },
        ),
        ("ollama", {"ollama_api_key": SECRET}),
        ("kilo", {"kilo_enabled": True}),
        ("llm7", {"llm7_enabled": True, "llm7_api_key": SECRET}),
        ("openai", {"openai_api_key": SECRET}),
    ],
)
def test_registry_does_not_construct_paid_or_unverified_clients(
    monkeypatch, provider, changes
):
    groq = SimpleNamespace(name="groq", close=AsyncMock())
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    types = {
        "openrouter": "OpenRouterProvider",
        "gemini": "GeminiProvider",
        "cloudflare": "CloudflareProvider",
        "ollama": "OllamaProvider",
        "kilo": "KiloProvider",
        "llm7": "LLM7Provider",
        "openai": "OpenAIProvider",
    }
    factory = Mock(side_effect=AssertionError("Ineligible client must not be created"))
    monkeypatch.setattr("app.llm.registry." + types[provider], factory)
    settings = Settings(_env_file=None, groq_api_key="test-key", **changes)

    async def exercise():
        registry = await create_provider_registry(settings)
        assert not registry.is_enabled(provider)
        await registry.close()

    asyncio.run(exercise())
    factory.assert_not_called()
    groq.close.assert_awaited_once()


@pytest.mark.parametrize("field", ["groq_fast_model", "groq_reasoning_model"])
def test_paid_primary_configuration_stops_before_client_construction(
    monkeypatch, field
):
    factory = Mock()
    monkeypatch.setattr("app.llm.registry.GroqProvider", factory)
    settings = Settings(_env_file=None, groq_api_key="test-key", **{field: SECRET})
    with pytest.raises(ValueError, match="zero-cost gate") as error:
        asyncio.run(create_provider_registry(settings))
    assert SECRET not in str(error.value)
    factory.assert_not_called()


def test_free_quota_failure_never_falls_back_to_paid_model(monkeypatch):
    groq = SimpleNamespace(
        name="groq",
        chat=AsyncMock(side_effect=LLMRateLimitError("free quota exhausted")),
        close=AsyncMock(),
    )
    paid_factory = Mock()
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    monkeypatch.setattr("app.llm.registry.OpenRouterProvider", paid_factory)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key=SECRET,
        openrouter_model="vendor/paid",
    )

    async def exercise():
        registry = await create_provider_registry(settings)
        router = LLMRouter(
            registry.require_enabled("groq"), registry.get_enabled("openrouter")
        )
        with pytest.raises(LLMRateLimitError):
            await router.chat([{"role": "user", "content": "hello"}])
        await registry.close()

    asyncio.run(exercise())
    paid_factory.assert_not_called()
    groq.chat.assert_awaited_once()


@pytest.mark.parametrize(
    "provider", ["openrouter", "gemini", "cloudflare", "ollama", "kilo", "llm7"]
)
def test_evaluation_paid_or_unverified_model_rejected_before_registry(
    monkeypatch, provider
):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    factory = AsyncMock()
    monkeypatch.setattr("app.llm.registry.create_provider_registry", factory)
    with pytest.raises(EvaluationError, match="zero-cost gate"):
        asyncio.run(
            create_live_target(load_dataset(), provider, "paid-model", live=True)
        )
    factory.assert_not_awaited()


@pytest.mark.parametrize("streaming", [False, True])
def test_evaluation_rechecks_model_before_every_call(monkeypatch, streaming):
    from test_evaluation_targets import Provider

    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    provider = Provider()
    dataset = load_dataset()
    target = ExistingProviderTarget(provider, dataset, "openai/gpt-oss-20b", live=True)
    target.model = SECRET

    async def exercise():
        with pytest.raises(EvaluationError, match="zero-cost gate"):
            if streaming:
                _ = [chunk async for chunk in target.stream(dataset.cases[0])]
            else:
                await target.chat(dataset.cases[0])
        await target.close()

    asyncio.run(exercise())
    assert provider.calls == 0
