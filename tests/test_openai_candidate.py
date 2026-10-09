import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.config import Settings
from app.evaluation.cli import main
from app.evaluation.loader import load_dataset
from app.evaluation.models import EvaluationError
from app.evaluation.targets import ExistingProviderTarget, create_live_target
from app.llm.capabilities import ProviderCapabilities
from app.llm.registry import ProviderRegistry, create_provider_registry
from app.llm.router import LLMRouter
from app.llm.task import TaskType

SECRET = "TEST_OPENAI_SECRET_DO_NOT_LOG"


def stub_provider(name):
    return SimpleNamespace(
        name=name,
        capabilities=ProviderCapabilities(images=True, streaming=True),
        chat=AsyncMock(),
        close=AsyncMock(),
    )


@pytest.fixture
def mock_live(monkeypatch):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)


@pytest.mark.parametrize("key", [None, "", " ", SECRET])
def test_rejected_openai_never_activates_even_with_key(monkeypatch, key):
    groq = stub_provider("groq")
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    factory = Mock(
        side_effect=AssertionError("Rejected client must not be constructed")
    )
    monkeypatch.setattr("app.llm.registry.OpenAIProvider", factory)
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openai_api_key=key,
        openai_model="gpt-6-luna",
    )

    async def exercise():
        registry = await create_provider_registry(settings)
        assert not registry.is_enabled("openai")
        assert registry.get_enabled("openai") is None
        await registry.close()

    asyncio.run(exercise())
    factory.assert_not_called()
    groq.close.assert_awaited_once()


@pytest.mark.parametrize("field", ["llm_primary_provider", "llm_fallback_provider"])
@pytest.mark.parametrize("key", [None, SECRET])
def test_production_configuration_cannot_admit_candidate(monkeypatch, field, key):
    factory = Mock()
    monkeypatch.setattr("app.llm.registry.GroqProvider", factory)
    settings = Settings(
        _env_file=None, groq_api_key="test-key", openai_api_key=key, **{field: "openai"}
    )
    with pytest.raises(ValueError, match="ZERO-COST GATE FAILED"):
        asyncio.run(create_provider_registry(settings))
    factory.assert_not_called()


@pytest.mark.parametrize("model", [None, "gpt-6-luna", SECRET])
def test_openai_rejection_precedes_settings_registry_and_key(
    monkeypatch, mock_live, model
):
    settings_factory = Mock(side_effect=AssertionError("No settings or secrets needed"))
    registry_factory = AsyncMock()
    monkeypatch.setattr("app.core.config.get_settings", settings_factory)
    monkeypatch.setattr("app.llm.registry.create_provider_registry", registry_factory)
    with pytest.raises(EvaluationError, match="ZERO-COST GATE FAILED") as error:
        asyncio.run(create_live_target(load_dataset(), "openai", model, live=True))
    assert SECRET not in str(error.value)
    settings_factory.assert_not_called()
    registry_factory.assert_not_awaited()


def test_direct_evaluation_wrapper_cannot_bypass_rejection(mock_live):
    provider = stub_provider("openai")
    with pytest.raises(EvaluationError, match="zero-cost gate"):
        ExistingProviderTarget(
            provider,
            load_dataset(),
            "gpt-6-luna",
            live=True,
            zero_cost_free_mode_verified=True,
        )
    provider.chat.assert_not_awaited()


@pytest.mark.parametrize(
    ("live", "env", "ci"),
    [(False, "1", ""), (True, "", ""), (True, "1", "true"), (True, "1", "")],
)
def test_cli_never_permits_rejected_openai(monkeypatch, capsys, live, env, ci):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", env)
    monkeypatch.setenv("CI", ci)
    factory = Mock()
    monkeypatch.setattr("app.llm.registry.OpenAIProvider", factory)
    args = ["run", "--provider", "openai", "--model", "gpt-6-luna"] + (
        ["--live"] if live else []
    )
    assert main(args) == 2
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert (
        "ZERO-COST GATE FAILED" in captured.err
        if live and env == "1" and not ci
        else "requires --live" in captured.err
    )
    factory.assert_not_called()


@pytest.mark.parametrize("task", list(TaskType))
def test_every_production_chain_preserves_exact_existing_order(task):
    names = ["groq", "openrouter", "gemini", "cloudflare", "ollama", "kilo", "llm7"]
    registry = ProviderRegistry(
        {name: stub_provider(name) for name in [*names, "openai"]}
    )
    router = LLMRouter(
        primary_provider=registry.require_enabled("groq"),
        fallback_provider=registry.get_enabled("openrouter"),
        **{name + "_provider": registry.get_enabled(name) for name in names[2:]},
    )
    expected = {
        TaskType.LONG_CONTEXT: [
            "gemini",
            "openrouter",
            "kilo",
            "cloudflare",
            "ollama",
            "groq",
        ],
        TaskType.MULTIMODAL: ["gemini", "openrouter", "groq"],
        TaskType.CODE: ["groq", "cloudflare", "ollama", "openrouter", "kilo", "llm7"],
        TaskType.REASONING: ["groq", "cloudflare", "ollama", "openrouter", "llm7"],
    }.get(task, ["groq", "openrouter", "cloudflare", "ollama", "kilo", "llm7"])
    chain = router._provider_chain(task, None)
    assert [provider.name for provider, _ in chain] == expected
    if task == TaskType.MULTIMODAL:
        assert [p.name for p, _ in router._provider_chain(task, None, True)] == [
            "gemini",
            "openrouter",
        ]
    assert registry.is_enabled("openai")
    assert registry.get_enabled("openai").chat.await_count == 0
