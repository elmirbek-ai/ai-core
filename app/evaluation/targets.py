from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field, TypeAdapter, model_validator

from app.evaluation.loader import PROJECT_ROOT, Dataset, content_hash
from app.evaluation.models import (
    Digest,
    Duration,
    ErrorCategory,
    EvaluationCase,
    EvaluationError,
    ModelName,
    SafeName,
    StrictModel,
    TargetOutput,
)
from app.llm.base import BaseLLMProvider
from app.llm.capabilities import ProviderCapabilities
from app.llm.cost_policy import free_mode_verified, require_zero_cost_model

EXISTING_PROVIDERS = (
    "groq",
    "openrouter",
    "gemini",
    "cloudflare",
    "ollama",
    "kilo",
    "llm7",
    "openai",  # Archived name for controlled rejection, never live permission.
)
MODEL_SETTINGS = {
    "groq": "groq_fast_model",
    "openrouter": "openrouter_model",
    "gemini": "gemini_model",
    "cloudflare": "cloudflare_model",
    "ollama": "ollama_model",
    "kilo": "kilo_general_model",
    "llm7": "llm7_general_model",
    "openai": "openai_model",
}


class TargetFailure(Exception):
    def __init__(self, category: ErrorCategory) -> None:
        super().__init__("Evaluation target failed")
        self.category = category


class EvaluationTarget(Protocol):
    provider: str
    model: str
    capabilities: ProviderCapabilities
    fixture: bool
    profile_hash: str | None
    asset_mapping_hash: str | None

    async def chat(self, case: EvaluationCase) -> TargetOutput: ...
    def stream(self, case: EvaluationCase) -> AsyncGenerator[TargetOutput]: ...
    async def close(self) -> None: ...


class FixtureClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FixtureAnswer(TargetOutput):
    error_category: ErrorCategory | None = None


class FixtureProfile(StrictModel):
    schema_version: Literal[1]
    dataset_version: SafeName
    dataset_hash: Digest
    provider: Annotated[str, Field(pattern=r"^fixture-[a-z0-9-]+$", max_length=128)]
    model: ModelName
    capabilities: ProviderCapabilities = Field(
        default_factory=lambda: ProviderCapabilities(images=True, streaming=True)
    )
    latency_seconds: Duration
    ttft_seconds: Duration
    answers: dict[SafeName, FixtureAnswer]

    @model_validator(mode="after")
    def valid_timings(self) -> FixtureProfile:
        if self.ttft_seconds > self.latency_seconds:
            raise ValueError("Fixture TTFT exceeds duration")
        return self


class FixtureTarget:
    fixture = True
    asset_mapping_hash: str | None = None

    def __init__(self, profile: FixtureProfile, clock: FixtureClock) -> None:
        self.profile = profile
        self.clock = clock
        self.provider = profile.provider
        self.model = profile.model
        self.capabilities = profile.capabilities
        self.profile_hash: str | None = content_hash(profile.model_dump(mode="json"))
        self.closed = False

    @classmethod
    def load(cls, dataset: Dataset, name: str = "target-a") -> FixtureTarget:
        if name not in {"target-a", "target-b"}:
            raise EvaluationError("Unknown fixture target")
        path = PROJECT_ROOT / "evals" / "fixtures" / "v1" / (name + ".json")
        try:
            profile = FixtureProfile.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if (
                profile.dataset_hash != dataset.hash
                or profile.dataset_version != dataset.manifest.dataset_version
                or set(profile.answers) != {case.id for case in dataset.cases}
            ):
                raise EvaluationError("Fixture profile does not match the dataset")
            return cls(profile, FixtureClock())
        except EvaluationError:
            raise
        except Exception:
            raise EvaluationError("Fixture profile validation failed") from None

    async def chat(self, case: EvaluationCase) -> TargetOutput:
        self.clock.advance(self.profile.latency_seconds)
        answer = self.profile.answers[case.id]
        if answer.error_category:
            raise TargetFailure(answer.error_category)
        return TargetOutput.model_validate(
            answer.model_dump(exclude={"error_category"})
        )

    async def stream(self, case: EvaluationCase) -> AsyncGenerator[TargetOutput]:
        answer = self.profile.answers[case.id]
        self.clock.advance(self.profile.ttft_seconds / 2)
        yield TargetOutput(content="")
        self.clock.advance(self.profile.ttft_seconds / 2)
        if answer.error_category:
            self.clock.advance(self.profile.latency_seconds - self.profile.ttft_seconds)
            raise TargetFailure(answer.error_category)
        yield TargetOutput(content=answer.content)
        self.clock.advance(self.profile.latency_seconds - self.profile.ttft_seconds)
        yield TargetOutput(
            content="",
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            usage_reliable=answer.usage_reliable,
        )

    async def close(self) -> None:
        self.closed = True


def require_live(live: bool) -> None:
    ci_enabled = os.environ.get("CI", "").lower() not in {"", "0", "false"}
    if not live or os.environ.get("AI_CORE_EVAL_ALLOW_LIVE") != "1" or ci_enabled:
        raise EvaluationError(
            "Live evaluation requires --live and AI_CORE_EVAL_ALLOW_LIVE=1 outside CI"
        )


def validate_model_name(model: str) -> str:
    try:
        value = TypeAdapter(ModelName).validate_python(model)
        if "://" in value:
            raise ValueError("A URL is not a model identifier")
        return value
    except ValueError:
        raise EvaluationError("Invalid evaluation model identifier") from None


def validate_image_mapping(mapping: dict[str, str]) -> None:
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise EvaluationError("Image mapping requires a JSON object of URL strings")
    for url in mapping.values():
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise EvaluationError(
                "Live image mappings require public HTTPS URLs "
                "without credentials or query strings"
            )


class ExistingProviderTarget:
    fixture = False
    profile_hash: str | None = None

    def __init__(
        self,
        provider: BaseLLMProvider,
        dataset: Dataset,
        model: str,
        *,
        live: bool,
        image_mapping: dict[str, str] | None = None,
        zero_cost_free_mode_verified: bool = False,
    ) -> None:
        require_live(live)
        if provider.name not in EXISTING_PROVIDERS:
            raise EvaluationError("Evaluation supports only existing providers")
        model = validate_model_name(model)
        if (
            provider.name in {"openrouter", "gemini", "cloudflare", "ollama"}
            and getattr(provider, "model", model) != model
        ):
            raise EvaluationError(
                "Evaluation model must match the configured adapter model"
            )
        try:
            require_zero_cost_model(
                provider.name, model, free_mode_verified=zero_cost_free_mode_verified
            )
        except ValueError:
            raise EvaluationError(
                "Evaluation target failed the mandatory zero-cost gate"
            ) from None
        self._provider = provider
        self._dataset = dataset
        self._live = live
        self._zero_cost_free_mode_verified = zero_cost_free_mode_verified
        validate_image_mapping(image_mapping if image_mapping is not None else {})
        self._image_mapping = dict(image_mapping or {})
        self.asset_mapping_hash = (
            content_hash(self._image_mapping) if self._image_mapping else None
        )
        self.provider = provider.name
        self.model = model
        self.capabilities = provider.capabilities
        self._registry: Any = None

    def _messages(self, case: EvaluationCase) -> list[dict[str, Any]]:
        require_live(self._live)
        try:
            require_zero_cost_model(
                self.provider,
                self.model,
                free_mode_verified=self._zero_cost_free_mode_verified,
            )
        except ValueError:
            raise EvaluationError(
                "Evaluation target failed the mandatory zero-cost gate"
            ) from None
        messages = self._dataset.messages(case)
        for message in messages:
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "image_url":
                        placeholder = part["image_url"]["url"]
                        if placeholder not in self._image_mapping:
                            raise EvaluationError(
                                "Live multimodal evaluation requires "
                                "explicit image asset mappings"
                            )
                        part["image_url"]["url"] = self._image_mapping[placeholder]
        return messages

    async def chat(self, case: EvaluationCase) -> TargetOutput:
        response = await self._provider.chat(self._messages(case), model=self.model)
        # Only response content is used in memory. Untrusted SDK model/provider
        # metadata is not copied into reports; the selected target is provenance.
        if not isinstance(response.get("content"), str):
            raise TargetFailure("invalid_output")
        return TargetOutput(content=response["content"])

    async def stream(self, case: EvaluationCase) -> AsyncGenerator[TargetOutput]:
        stream = self._provider.stream_chat(self._messages(case), model=self.model)
        async with aclosing(stream):
            async for chunk in stream:
                if not isinstance(chunk.content, str):
                    raise TargetFailure("invalid_output")
                yield TargetOutput(content=chunk.content)

    async def close(self) -> None:
        if self._registry is not None:
            await self._registry.close()
        else:
            await self._provider.close()


async def create_live_target(
    dataset: Dataset,
    provider_name: str,
    model: str | None,
    *,
    live: bool,
    image_mapping: dict[str, str] | None = None,
) -> ExistingProviderTarget:
    require_live(live)
    if provider_name not in EXISTING_PROVIDERS:
        raise EvaluationError("Evaluation supports only existing providers")
    if provider_name == "openai":
        raise EvaluationError("OpenAI direct is rejected: ZERO-COST GATE FAILED")
    from app.core.config import get_settings
    from app.llm.registry import create_provider_registry

    settings = get_settings()
    selected_model = validate_model_name(
        model or getattr(settings, MODEL_SETTINGS[provider_name])
    )
    verified = free_mode_verified(settings, provider_name)
    try:
        require_zero_cost_model(
            provider_name, selected_model, free_mode_verified=verified
        )
    except ValueError:
        raise EvaluationError(
            "Evaluation target failed the mandatory zero-cost gate"
        ) from None
    # Several existing adapters intentionally use their configured model. A
    # dedicated registry/settings copy supports targeting without editing them.
    settings = settings.model_copy(
        update={MODEL_SETTINGS[provider_name]: selected_model}
    )
    registry = await create_provider_registry(settings)
    try:
        provider = registry.get_enabled(provider_name)
        if provider is None:
            raise EvaluationError("Requested evaluation provider is not enabled")
        target = ExistingProviderTarget(
            provider,
            dataset,
            selected_model,
            live=live,
            image_mapping=image_mapping,
            zero_cost_free_mode_verified=verified,
        )
        target._registry = registry
        return target
    except BaseException:
        await registry.close()
        raise
