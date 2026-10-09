import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.evaluation.loader import load_dataset
from app.evaluation.models import EvaluationError
from app.evaluation.targets import (
    MODEL_SETTINGS,
    ExistingProviderTarget,
    FixtureProfile,
    FixtureTarget,
    create_live_target,
    require_live,
    validate_image_mapping,
)
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import ProviderCapabilities
from app.llm.streaming import ProviderStreamChunk


class Provider(BaseLLMProvider):
    def __init__(self, name="groq", *, content="safe", error=None):
        self._name = name
        self.content = content
        self.error = error
        self.messages = None
        self.selected_model = None
        self.calls = 0
        self.closed = False
        self.stream_closed = False

    @property
    def name(self):
        return self._name

    @property
    def capabilities(self):
        return ProviderCapabilities(images=True, streaming=True)

    async def chat(self, messages, model=None):
        self.calls += 1
        self.messages = messages
        self.selected_model = model
        if self.error:
            raise self.error
        return {
            "provider": self.name,
            "model": "untrusted-SDK-metadata",
            "content": self.content,
        }

    async def stream_chat(self, messages, model=None):
        self.messages = messages
        self.selected_model = model
        self.calls += 1
        try:
            yield ProviderStreamChunk(self.name, "untrusted", "")
            yield ProviderStreamChunk(self.name, "untrusted", self.content)
            if self.error:
                raise self.error
        finally:
            self.stream_closed = True

    async def close(self):
        self.closed = True


@pytest.fixture
def live_opt_in(monkeypatch):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)


@pytest.mark.parametrize(
    ("flag", "env", "ci", "allowed"),
    [
        (False, None, None, False),
        (True, None, None, False),
        (False, "1", None, False),
        (True, "0", None, False),
        (True, "1", "true", False),
        (True, "1", None, True),
        (True, "1", "false", True),
        (True, "1", "1", False),
    ],
)
def test_live_guard_requires_flag_env_and_outside_ci(
    monkeypatch, flag, env, ci, allowed
):
    for key, value in [("AI_CORE_EVAL_ALLOW_LIVE", env), ("CI", ci)]:
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    if allowed:
        require_live(flag)
    else:
        with pytest.raises(EvaluationError):
            require_live(flag)


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_wrapper_calls_existing_contract_without_usage(streaming, live_opt_in):
    async def exercise():
        dataset = load_dataset()
        provider = Provider()
        target = ExistingProviderTarget(
            provider, dataset, "openai/gpt-oss-20b", live=True
        )
        if streaming:
            output = [chunk async for chunk in target.stream(dataset.cases[0])]
            assert [chunk.content for chunk in output] == ["", "safe"]
            assert provider.stream_closed
        else:
            output = await target.chat(dataset.cases[0])
            assert output.content == "safe"
            assert output.input_tokens is output.output_tokens is None
            assert not output.usage_reliable
        assert provider.selected_model == target.model == "openai/gpt-oss-20b"
        assert provider.messages == dataset.messages(dataset.cases[0])
        await target.close()
        assert provider.closed

    asyncio.run(exercise())


@pytest.mark.parametrize("streaming", [False, True])
def test_revoked_opt_in_blocks_each_provider_call(streaming, live_opt_in, monkeypatch):
    dataset = load_dataset()
    provider = Provider()
    target = ExistingProviderTarget(provider, dataset, "openai/gpt-oss-20b", live=True)
    monkeypatch.delenv("AI_CORE_EVAL_ALLOW_LIVE")

    async def exercise():
        with pytest.raises(EvaluationError):
            if streaming:
                _ = [chunk async for chunk in target.stream(dataset.cases[0])]
            else:
                await target.chat(dataset.cases[0])
        await target.close()

    asyncio.run(exercise())
    assert provider.calls == 0


@pytest.mark.parametrize(
    "provider_name",
    [name for name in MODEL_SETTINGS if name not in {"openai", "ollama"}],
)
def test_evaluation_model_override_uses_isolated_settings_copy(
    provider_name, live_opt_in, monkeypatch
):
    from app.core.config import get_settings

    original = get_settings()
    original = original.model_copy(
        update={"cloudflare_zero_cost_verified": True, "llm7_zero_cost_verified": True}
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: original)
    selected_model = {
        "groq": "openai/gpt-oss-120b",
        "openrouter": "test-provider/model:free",
        "gemini": "gemini-3.8-flash",
        "cloudflare": "cf/openai/gpt-oss-120b",
        "kilo": "kilo-auto/free",
        "llm7": "gpt-oss:20b",
    }[provider_name]
    old_model = getattr(original, MODEL_SETTINGS[provider_name])
    captured = []
    registry = SimpleNamespace(
        get_enabled=lambda name: Provider(name), close=AsyncMock()
    )

    async def factory(settings):
        captured.append(settings)
        return registry

    monkeypatch.setattr("app.llm.registry.create_provider_registry", factory)

    async def exercise():
        target = await create_live_target(
            load_dataset(), provider_name, selected_model, live=True
        )
        assert target.provider == provider_name
        assert target.model == selected_model
        await target.close()

    asyncio.run(exercise())
    assert captured[0] is not original
    assert getattr(captured[0], MODEL_SETTINGS[provider_name]) == selected_model
    assert getattr(original, MODEL_SETTINGS[provider_name]) == old_model
    registry.close.assert_awaited_once()


def test_factory_default_model_disabled_provider_and_unknown_provider(
    live_opt_in, monkeypatch
):
    registry = SimpleNamespace(
        get_enabled=lambda name: Provider(name), close=AsyncMock()
    )
    factory = AsyncMock(return_value=registry)
    monkeypatch.setattr("app.llm.registry.create_provider_registry", factory)

    async def exercise():
        target = await create_live_target(load_dataset(), "groq", None, live=True)
        assert target.model
        await target.close()
        registry.get_enabled = lambda name: None
        with pytest.raises(EvaluationError, match="not enabled"):
            await create_live_target(load_dataset(), "groq", None, live=True)
        with pytest.raises(EvaluationError, match="only existing"):
            await create_live_target(load_dataset(), "new-provider", None, live=True)

    asyncio.run(exercise())
    assert factory.await_count == 2
    assert registry.close.await_count == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://assets.invalid/img.png",
        "https://user:pass@assets.invalid/img.png",
        "https://assets.invalid/img.png?key=TEST_SECRET_DO_NOT_LOG",
        "https://assets.invalid/img.png#fragment",
        "https://",
    ],
)
def test_image_url_mapping_requires_public_https_without_credentials(url):
    with pytest.raises(EvaluationError):
        validate_image_mapping({"placeholder": url})


def test_live_images_require_explicit_detached_mapping(live_opt_in):
    dataset = load_dataset()
    case = next(case for case in dataset.cases if case.domain == "multimodal")
    provider = Provider()
    missing = ExistingProviderTarget(provider, dataset, "openai/gpt-oss-20b", live=True)
    with pytest.raises(EvaluationError, match="image asset mappings"):
        asyncio.run(missing.chat(case))
    assert provider.calls == 0
    placeholder = case.messages[0].model_dump(mode="json")["content"][1]["image_url"][
        "url"
    ]
    mapping = {placeholder: "https://assets.invalid/approved.png"}
    target = ExistingProviderTarget(
        provider, dataset, "openai/gpt-oss-20b", live=True, image_mapping=mapping
    )
    mapping[placeholder] = "changed"
    asyncio.run(target.chat(case))
    assert (
        provider.messages[0]["content"][1]["image_url"]["url"]
        == "https://assets.invalid/approved.png"
    )
    assert len(target.asset_mapping_hash) == 64
    assert dataset.messages(case)[0]["content"][1]["image_url"]["url"] == placeholder
    asyncio.run(target.close())


def test_fixture_validation_unknown_target_bad_profile_and_dataset_mismatch(
    monkeypatch,
):
    dataset = load_dataset()
    with pytest.raises(EvaluationError, match="Unknown fixture"):
        FixtureTarget.load(dataset, "missing")
    profile = FixtureTarget.load(dataset).profile.model_dump(mode="json")
    profile["ttft_seconds"] = profile["latency_seconds"] + 1
    with pytest.raises(ValueError):
        FixtureProfile.model_validate(profile)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: "invalid JSON")
    with pytest.raises(EvaluationError, match="profile validation"):
        FixtureTarget.load(dataset)


def test_fixture_dataset_mismatch_rejected():
    from dataclasses import replace

    dataset = load_dataset()
    with pytest.raises(EvaluationError, match="does not match"):
        FixtureTarget.load(replace(dataset, hash="0" * 64))


@pytest.mark.parametrize("mapping", [["url"], {"placeholder": 1}])
def test_image_mapping_rejects_non_object_or_non_string_values(mapping):
    with pytest.raises(EvaluationError, match="JSON object"):
        validate_image_mapping(mapping)


def test_factory_guard_precedes_settings_and_registry(monkeypatch):
    monkeypatch.delenv("AI_CORE_EVAL_ALLOW_LIVE", raising=False)
    factory = AsyncMock()
    monkeypatch.setattr("app.llm.registry.create_provider_registry", factory)
    with pytest.raises(EvaluationError):
        asyncio.run(create_live_target(load_dataset(), "groq", None, live=True))
    factory.assert_not_awaited()


@pytest.mark.parametrize(
    "model", ["", "bad\nmodel", "https://private.invalid/model", "x" * 201]
)
def test_invalid_target_model_rejected_before_provider_call(model, live_opt_in):
    provider = Provider()
    with pytest.raises(EvaluationError, match="model identifier"):
        ExistingProviderTarget(provider, load_dataset(), model, live=True)
    assert provider.calls == 0


def test_direct_target_cannot_mislabel_fixed_model_adapter(live_opt_in):
    provider = Provider("gemini")
    provider.model = "configured-model"
    with pytest.raises(EvaluationError, match="configured adapter"):
        ExistingProviderTarget(provider, load_dataset(), "different-model", live=True)
    assert provider.calls == 0
    with pytest.raises(EvaluationError, match="only existing"):
        ExistingProviderTarget(
            Provider("new-provider"), load_dataset(), "openai/gpt-oss-20b", live=True
        )
