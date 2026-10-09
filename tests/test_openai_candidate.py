import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_openai_provider import (
    MODEL,
    SECRET,
    FakeStream,
    client_for,
    completed,
    done,
)

from app.core.config import Settings
from app.evaluation.cli import main
from app.evaluation.loader import load_dataset
from app.evaluation.models import EvaluationError, RunConfig
from app.evaluation.pricing import PriceCatalog
from app.evaluation.reporting import render_markdown
from app.evaluation.runner import run_evaluation
from app.evaluation.targets import (
    ExistingProviderTarget,
    FixtureClock,
    create_live_target,
)
from app.llm.capabilities import ProviderCapabilities
from app.llm.providers.openai import OpenAIProvider
from app.llm.registry import ProviderRegistry, create_provider_registry
from app.llm.router import LLMRouter
from app.llm.task import TaskType


def candidate_settings(**changes):
    values = {
        "groq_api_key": "test-key",
        "openai_api_key": SECRET,
        "openai_model": MODEL,
        "openrouter_api_key": None,
        "gemini_api_key": None,
        "cloudflare_api_token": None,
        "cloudflare_account_id": None,
        "ollama_api_key": None,
        "kilo_enabled": False,
        "llm7_enabled": False,
    }
    return Settings(_env_file=None, **(values | changes))


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
def test_registry_candidate_enablement_and_client_cleanup(monkeypatch, key):
    groq, candidate = stub_provider("groq"), stub_provider("openai")
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    factory = Mock(return_value=candidate)
    monkeypatch.setattr("app.llm.registry.OpenAIProvider", factory)

    async def exercise():
        registry = await create_provider_registry(
            candidate_settings(openai_api_key=key)
        )
        assert registry.get_enabled("groq") is groq
        assert registry.is_enabled("openai") == (key == SECRET)
        assert registry.get_enabled("openai") is (candidate if key == SECRET else None)
        await registry.close()

    asyncio.run(exercise())
    groq.close.assert_awaited_once()
    assert factory.call_count == (1 if key == SECRET else 0)
    assert candidate.close.await_count == (1 if key == SECRET else 0)


def test_registry_accepts_key_without_default_model(monkeypatch):
    client = client_for()
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", lambda **kw: client)
    groq = stub_provider("groq")
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)

    async def exercise():
        registry = await create_provider_registry(candidate_settings(openai_model=None))
        assert registry.get_enabled("openai").model is None
        await registry.close()

    asyncio.run(exercise())
    client.responses.create.assert_not_awaited()
    client.close.assert_awaited_once()


@pytest.mark.parametrize("field", ["llm_primary_provider", "llm_fallback_provider"])
@pytest.mark.parametrize("key", [None, SECRET])
def test_production_configuration_cannot_admit_candidate(monkeypatch, field, key):
    factory = Mock()
    monkeypatch.setattr("app.llm.registry.GroqProvider", factory)
    with pytest.raises(ValueError, match="evaluation-only"):
        asyncio.run(
            create_provider_registry(
                candidate_settings(**{field: "openai", "openai_api_key": key})
            )
        )
    factory.assert_not_called()


def test_failed_candidate_construction_closes_previous_clients(monkeypatch):
    groq = stub_provider("groq")
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    factory = Mock(side_effect=ValueError("controlled construction failure"))
    monkeypatch.setattr("app.llm.registry.OpenAIProvider", factory)
    with pytest.raises(ValueError):
        asyncio.run(create_provider_registry(candidate_settings()))
    groq.close.assert_awaited_once()


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


@pytest.mark.parametrize(
    ("live", "env", "ci"),
    [(False, "1", ""), (True, "", ""), (True, "1", "true")],
)
def test_openai_guard_blocks_before_sdk_construction(monkeypatch, live, env, ci):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", env)
    monkeypatch.setenv("CI", ci)
    factory = Mock(side_effect=AssertionError("SDK must not be constructed"))
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", factory)
    with pytest.raises(EvaluationError, match="requires --live"):
        asyncio.run(create_live_target(load_dataset(), "openai", MODEL, live=live))
    factory.assert_not_called()


@pytest.mark.parametrize("missing", ["key", "model"])
def test_missing_openai_configuration_is_clear_and_safe(
    monkeypatch, mock_live, missing
):
    settings = candidate_settings(
        **{"openai_api_" + "key": None} if missing == "key" else {"openai_model": None}
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    factory = AsyncMock()
    monkeypatch.setattr("app.llm.registry.create_provider_registry", factory)
    with pytest.raises(
        EvaluationError, match="OPENAI_API_KEY" if missing == "key" else "OPENAI_MODEL"
    ) as error:
        asyncio.run(create_live_target(load_dataset(), "openai", None, live=True))
    assert SECRET not in str(error.value)
    factory.assert_not_awaited()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("override", [None, "evaluation-only-model"])
def test_mocked_registry_to_evaluation_uses_responses_and_isolated_model(
    monkeypatch, mock_live, streaming, override
):
    settings = candidate_settings()
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    groq = stub_provider("groq")
    monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
    upstream = (
        FakeStream(
            [SimpleNamespace(type="response.output_text.delta", delta="answer"), done()]
        )
        if streaming
        else completed()
    )
    client = client_for(upstream)
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", lambda **kw: client)

    async def exercise():
        dataset = load_dataset()
        target = await create_live_target(dataset, "openai", override, live=True)
        assert target.model == (override or MODEL)
        assert settings.openai_model == MODEL
        assert target._provider.model == target.model
        report = await run_evaluation(
            replace(dataset, cases=dataset.cases[:1]),
            target,
            RunConfig(streaming=streaming),
        )
        assert report.overall.successes == 1
        assert report.metadata.provider == "openai"
        assert report.metadata.model == target.model
        assert report.cases[0].input_tokens is report.cases[0].output_tokens is None
        assert report.cases[0].cost_usd is None
        assert report.cases[0].cost_status == "unavailable"
        assert (report.cases[0].ttft_seconds is not None) == streaming
        assert SECRET not in report.model_dump_json()
        assert SECRET not in render_markdown(report)
        assert client.responses.create.call_args.kwargs["model"] == (override or MODEL)
        assert client.responses.create.call_args.kwargs["store"] is False

    asyncio.run(exercise())
    client.close.assert_awaited_once()
    groq.close.assert_awaited_once()


@pytest.mark.parametrize("images", [False, True])
def test_multimodal_skip_or_explicit_mapping_without_url_persistence(mock_live, images):
    dataset = load_dataset()
    case = next(case for case in dataset.cases if case.domain == "multimodal")
    dataset = replace(dataset, cases=[case])
    client = client_for(completed())
    provider = OpenAIProvider(
        candidate_settings(openai_supports_images=images), client=client
    )
    placeholder = dataset.messages(case)[0]["content"][1]["image_url"]["url"]
    url = "https://operator-assets.invalid/approved-square.png"
    target = ExistingProviderTarget(
        provider, dataset, MODEL, live=True, image_mapping={placeholder: url}
    )
    report = asyncio.run(run_evaluation(dataset, target, RunConfig()))
    assert report.overall.skipped == (0 if images else 1)
    assert report.overall.successes == (1 if images else 0)
    assert url not in report.model_dump_json()
    if images:
        assert (
            client.responses.create.call_args.kwargs["input"][0]["content"][1][
                "image_url"
            ]
            == url
        )
    else:
        client.responses.create.assert_not_awaited()
    client.close.assert_awaited_once()


def test_openai_image_mapping_guard_precedes_provider_call(mock_live):
    dataset = load_dataset()
    case = next(case for case in dataset.cases if case.domain == "multimodal")
    client = client_for()
    provider = OpenAIProvider(
        candidate_settings(openai_supports_images=True), client=client
    )
    target = ExistingProviderTarget(provider, dataset, MODEL, live=True)
    with pytest.raises(EvaluationError, match="image asset mappings"):
        asyncio.run(target.chat(case))
    client.responses.create.assert_not_awaited()
    asyncio.run(target.close())


def test_exception_body_and_generated_secrets_absent_from_reports(mock_live, caplog):
    dataset = load_dataset()
    dataset = replace(dataset, cases=dataset.cases[:1])
    for response, error in [
        (SimpleNamespace(status="completed", error=None, output_text=SECRET), None),
        (
            None,
            RuntimeError(
                "Authorization: Bearer "
                + SECRET
                + " https://private.invalid/?token="
                + SECRET
            ),
        ),
    ]:
        client = client_for(response, error=error)
        provider = OpenAIProvider(candidate_settings(), client=client)
        target = ExistingProviderTarget(provider, dataset, MODEL, live=True)
        report = asyncio.run(run_evaluation(dataset, target, RunConfig()))
        assert SECRET not in report.model_dump_json()
        assert SECRET not in render_markdown(report)
        assert SECRET not in caplog.text
        if error:
            assert report.cases[0].error_category == "provider"
        client.close.assert_awaited_once()


def test_sdk_usage_does_not_invent_evaluation_cost(mock_live):
    dataset = load_dataset()
    dataset = replace(dataset, cases=dataset.cases[:1])
    client = client_for(
        completed(usage=SimpleNamespace(input_tokens=100, output_tokens=20))
    )
    provider = OpenAIProvider(candidate_settings(), client=client)
    target = ExistingProviderTarget(provider, dataset, MODEL, live=True)
    prices = PriceCatalog.model_validate(
        {
            "schema_version": 1,
            "version": "test-only-fake",
            "price_date": "2026-01-01",
            "currency": "USD",
            "evidence": "synthetic_fixture",
            "entries": [
                {
                    "provider": "openai",
                    "model": MODEL,
                    "input_usd_per_million": "1",
                    "output_usd_per_million": "2",
                }
            ],
        }
    )
    report = asyncio.run(run_evaluation(dataset, target, RunConfig(), prices=prices))
    assert report.cases[0].cost_status == "unavailable"
    assert (
        report.cases[0].cost_usd
        is report.cases[0].input_tokens
        is report.cases[0].output_tokens
        is None
    )


@pytest.mark.parametrize(
    ("live", "env", "ci"), [(False, "1", ""), (True, "", ""), (True, "1", "true")]
)
def test_cli_guard_does_not_create_openai_client(monkeypatch, capsys, live, env, ci):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", env)
    monkeypatch.setenv("CI", ci)
    factory = Mock()
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", factory)
    args = ["run", "--provider", "openai", "--model", MODEL] + (
        ["--live"] if live else []
    )
    assert main(args) == 2
    assert "requires --live" in capsys.readouterr().err
    factory.assert_not_called()


def test_cli_missing_key_is_safe(monkeypatch, mock_live, capsys):
    monkeypatch.setattr(
        "app.core.config.get_settings", lambda: candidate_settings(openai_api_key=None)
    )
    factory = Mock()
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", factory)
    assert main(["run", "--provider", "openai", "--live", "--model", MODEL]) == 2
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY is missing" in captured.err
    assert SECRET not in captured.err + captured.out
    factory.assert_not_called()


def test_cli_mocked_openai_execution_reports_safely(
    monkeypatch, mock_live, tmp_path, capsys
):
    settings = candidate_settings()
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.llm.registry.GroqProvider", lambda settings: stub_provider("groq")
    )
    client = client_for(error=RuntimeError(SECRET))
    monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", lambda **kw: client)
    assert (
        main(
            [
                "run",
                "--provider",
                "openai",
                "--live",
                "--model",
                MODEL,
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    for filename in ["evaluation.json", "evaluation.md"]:
        assert SECRET not in (tmp_path / filename).read_text(encoding="utf-8")
    client.close.assert_awaited_once()


def test_evaluation_ttft_ignores_empty_and_nontext_responses_events(mock_live):
    clock = FixtureClock()

    class TimedStream(FakeStream):
        async def __anext__(self):
            event = await super().__anext__()
            clock.advance(1)
            return event

    stream = TimedStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta=""),
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="hello"),
            done(),
        ]
    )
    client = client_for(stream)
    provider = OpenAIProvider(candidate_settings(), client)
    dataset = load_dataset()
    dataset = replace(dataset, cases=dataset.cases[:1])
    target = ExistingProviderTarget(provider, dataset, MODEL, live=True)
    report = asyncio.run(
        run_evaluation(dataset, target, RunConfig(streaming=True), clock=clock)
    )
    assert report.cases[0].ttft_seconds == 3
    assert report.cases[0].latency_seconds == 4
    stream.close.assert_awaited_once()
    client.close.assert_awaited_once()


@pytest.mark.parametrize("after_delta", [False, True])
def test_cancelled_openai_evaluation_closes_stream_and_owned_registry(
    monkeypatch, mock_live, after_delta
):
    async def exercise():
        entered = asyncio.Event()
        stream = FakeStream(
            [SimpleNamespace(type="response.output_text.delta", delta="hello")]
            if after_delta
            else [],
            entered=entered,
        )
        client = client_for(stream)
        monkeypatch.setattr("app.llm.providers.openai.AsyncOpenAI", lambda **kw: client)
        groq = stub_provider("groq")
        monkeypatch.setattr("app.llm.registry.GroqProvider", lambda settings: groq)
        monkeypatch.setattr("app.core.config.get_settings", candidate_settings)
        dataset = load_dataset()
        dataset = replace(dataset, cases=dataset.cases[:1])
        target = await create_live_target(dataset, "openai", MODEL, live=True)
        task = asyncio.create_task(
            run_evaluation(dataset, target, RunConfig(streaming=True))
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        stream.close.assert_awaited_once()
        client.close.assert_awaited_once()
        groq.close.assert_awaited_once()

    asyncio.run(exercise())
