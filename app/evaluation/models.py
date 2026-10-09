from __future__ import annotations

import keyword
import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.llm.capabilities import messages_contain_images
from app.llm.task import TaskType
from app.schemas.chat import ChatMessage

Domain = Literal[
    "general",
    "reasoning",
    "code",
    "kyrgyz",
    "russian",
    "english",
    "translation",
    "summarization",
    "extraction",
    "long_context",
    "multimodal",
]
Language = Literal["ky", "ru", "en"]
Metric = Literal[
    "accuracy_score",
    "instruction_following_score",
    "reasoning_score",
    "formatting_score",
    "groundedness_score",
]
ScorerKind = Literal[
    "exact",
    "required_terms",
    "forbidden_terms",
    "numeric",
    "json",
    "format",
    "overlap_f1",
    "python_syntax",
    "grounded_facts",
    "forbidden_claims",
    "length",
]
ErrorCategory = Literal[
    "authentication",
    "rate_limit",
    "timeout",
    "upstream",
    "provider",
    "invalid_output",
    "output_limit",
    "empty_stream",
    "fixture_failure",
]
SafeName = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$")
]
ModelName = Annotated[
    str, Field(min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$")
]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Weight = Annotated[float, Field(gt=0, le=1000, allow_inf_nan=False, strict=True)]
ScoreValue = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Duration = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Tokens = Annotated[int, Field(ge=0, strict=True)]


class EvaluationError(ValueError):
    """A controlled operator-facing message, never raw exception text."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def strict_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Schema version must be an integer")
        return value


class Requirements(StrictModel):
    text: bool = True
    images: bool = False
    streaming: bool = False


class ScorerSpec(StrictModel):
    kind: ScorerKind
    metric: Metric
    weight: Weight = 1.0
    expected: str | float | dict[str, JsonValue] | None = None
    terms: list[Annotated[str, Field(min_length=1, max_length=500)]] | None = Field(
        default=None, max_length=100
    )
    tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    output_format: Literal["single_line", "bullets", "json", "python"] | None = None
    required_symbols: list[SafeName] | None = Field(default=None, max_length=100)
    min_words: Annotated[int, Field(ge=0, le=100000, strict=True)] | None = None
    max_words: Annotated[int, Field(ge=0, le=100000, strict=True)] | None = None

    @field_validator("expected", mode="before")
    @classmethod
    def numeric_reference_is_not_boolean(
        cls, value: object, info: ValidationInfo
    ) -> object:
        if info.data.get("kind") == "numeric" and isinstance(value, bool):
            raise ValueError("Numeric reference cannot be a boolean")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> ScorerSpec:
        allowed = {
            "exact": {"expected"},
            "numeric": {"expected", "tolerance"},
            "json": {"expected"},
            "overlap_f1": {"expected"},
            "required_terms": {"terms"},
            "forbidden_terms": {"terms"},
            "grounded_facts": {"terms"},
            "forbidden_claims": {"terms"},
            "format": {"output_format"},
            "python_syntax": {"required_symbols"},
            "length": {"min_words", "max_words"},
        }[self.kind]
        config_fields = {
            "expected",
            "terms",
            "tolerance",
            "output_format",
            "required_symbols",
            "min_words",
            "max_words",
        }
        if any(getattr(self, name) is not None for name in config_fields - allowed):
            raise ValueError("Scorer configuration has inapplicable fields")
        if self.kind in {"exact", "overlap_f1"} and not isinstance(self.expected, str):
            raise ValueError("Scorer requires a text reference")
        if self.kind == "numeric" and (
            not isinstance(self.expected, float) or self.tolerance is None
        ):
            raise ValueError("Numeric scorer requires a finite value and tolerance")
        if self.kind == "numeric" and self.expected is not None:
            assert isinstance(self.expected, float)
            if not math.isfinite(self.expected):
                raise ValueError("Numeric reference must be finite")
        if self.kind == "json" and (
            not isinstance(self.expected, dict) or not self.expected
        ):
            raise ValueError("JSON scorer requires expected fields")
        if self.kind in {
            "required_terms",
            "forbidden_terms",
            "grounded_facts",
            "forbidden_claims",
        } and (
            not self.terms
            or len(set(self.terms)) != len(self.terms)
            or any(not term.strip() for term in self.terms)
        ):
            raise ValueError("Scorer requires unique nonempty terms")
        if self.required_symbols and (
            len(set(self.required_symbols)) != len(self.required_symbols)
            or any(
                not symbol.isidentifier() or keyword.iskeyword(symbol)
                for symbol in self.required_symbols
            )
        ):
            raise ValueError("Python scorer requires unique valid symbol names")
        if self.kind == "format" and self.output_format is None:
            raise ValueError("Format scorer requires output_format")
        if self.kind == "length" and (
            (self.min_words is None and self.max_words is None)
            or (
                self.min_words is not None
                and self.max_words is not None
                and self.min_words > self.max_words
            )
        ):
            raise ValueError("Length scorer requires consistent bounds")
        if (
            self.kind in {"grounded_facts", "forbidden_claims"}
            and self.metric != "groundedness_score"
        ):
            raise ValueError("Grounding scorers require the groundedness metric")
        return self


class TextAsset(StrictModel):
    name: SafeName
    path: Annotated[str, Field(min_length=1, max_length=200)]
    message_index: Annotated[int, Field(ge=0, strict=True)]


class EvaluationCase(StrictModel):
    id: SafeName
    domain: Domain
    language: Language
    task: TaskType
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    scorers: list[ScorerSpec] = Field(min_length=1, max_length=30)
    weight: Weight = 1.0
    tags: list[SafeName] = Field(default_factory=list, max_length=20)
    requirements: Requirements = Field(default_factory=Requirements)
    text_assets: list[TextAsset] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_assets_and_images(self) -> EvaluationCase:
        if messages_contain_images(
            [message.model_dump(mode="json") for message in self.messages]
        ) and (not self.requirements.images or self.task != TaskType.MULTIMODAL):
            raise ValueError("Image cases require multimodal task and capability")
        names = set()
        for asset in self.text_assets:
            if asset.name in names or asset.message_index >= len(self.messages):
                raise ValueError("Invalid text asset reference")
            names.add(asset.name)
            content = self.messages[asset.message_index].content
            if (
                not isinstance(content, str)
                or "{{asset:" + asset.name + "}}" not in content
            ):
                raise ValueError("Missing text asset placeholder")
        return self


class Manifest(StrictModel):
    schema_version: Literal[1]
    dataset_version: SafeName
    created: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    domain_files: dict[Domain, Annotated[str, Field(min_length=1, max_length=200)]] = (
        Field(min_length=1)
    )
    supported_languages: list[Language] = Field(min_length=1, max_length=3)
    case_count: Annotated[int, Field(gt=0, le=10000, strict=True)]
    quality_weights: dict[Metric, Weight] = Field(min_length=1)
    content_hash: Digest


class TargetOutput(StrictModel):
    content: str
    input_tokens: Tokens | None = None
    output_tokens: Tokens | None = None
    usage_reliable: bool = False


class Scores(StrictModel):
    quality_score: ScoreValue | None = None
    accuracy_score: ScoreValue | None = None
    instruction_following_score: ScoreValue | None = None
    reasoning_score: ScoreValue | None = None
    formatting_score: ScoreValue | None = None
    groundedness_score: ScoreValue | None = None
    hallucination_flag: bool | None = None


class RunConfig(StrictModel):
    streaming: bool = False
    repetitions: Annotated[int, Field(ge=1, le=100, strict=True)] = 1
    timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60.0
    max_output_chars: Annotated[int, Field(ge=1, le=2000000, strict=True)] = 100000


class CaseResult(StrictModel):
    case_id: SafeName
    domain: Domain
    language: Language
    repetition: Annotated[int, Field(ge=1, strict=True)]
    weight: Weight
    provider: SafeName
    model: ModelName
    status: Literal["success", "failure", "skipped"]
    error_category: ErrorCategory | None = None
    skip_reason: Literal["capability"] | None = None
    latency_seconds: Duration | None = None
    ttft_seconds: Duration | None = None
    output_chars: Tokens = 0
    scores: Scores = Field(default_factory=Scores)
    cost_usd: Duration | None = None
    cost_status: Literal["computed", "unavailable"] = "unavailable"
    input_tokens: Tokens | None = None
    output_tokens: Tokens | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> CaseResult:
        if self.status == "success" and (
            self.error_category or self.skip_reason or self.latency_seconds is None
        ):
            raise ValueError("Invalid success outcome")
        if self.status == "failure" and (
            self.error_category is None
            or self.skip_reason
            or self.latency_seconds is None
        ):
            raise ValueError("Invalid failure outcome")
        if self.status == "skipped" and (
            self.skip_reason is None
            or self.error_category
            or self.latency_seconds is not None
            or self.ttft_seconds is not None
        ):
            raise ValueError("Invalid skipped outcome")
        if self.status != "success" and any(
            value is not None for value in self.scores.model_dump().values()
        ):
            raise ValueError("Unsuccessful cases cannot carry quality scores")
        if (self.cost_status == "computed") != (self.cost_usd is not None):
            raise ValueError("Inconsistent cost availability")
        return self


class Provenance(StrictModel):
    schema_version: Literal[1] = 1
    framework_version: Literal["1.0.0"] = "1.0.0"
    dataset_version: SafeName
    dataset_hash: Digest
    scorer_config_hash: Digest
    price_catalog_hash: Digest | None = None
    asset_mapping_hash: Digest | None = None
    fixture_profile_hash: Digest | None = None
    git_sha: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")] | None = None
    git_dirty: bool | None = None
    python_version: str
    generated_at: str
    provider: SafeName
    model: ModelName
    mode: Literal["fixture", "live"]
    case_ids: list[SafeName]
    config: RunConfig


class Statistics(StrictModel):
    count: int
    mean: float | None
    p50: float | None
    p95: float | None


class Aggregate(StrictModel):
    total: int
    attempted: int
    successes: int
    failures: int
    skipped: int
    success_rate: float | None
    failure_rate: float | None
    scores: Scores
    metric_availability: dict[str, int]
    latency: Statistics
    ttft: Statistics
    hallucination_rate: float | None
    cost_available_count: int
    cost_total_usd: float | None
    cost_average_usd: float | None
    error_categories: dict[ErrorCategory, int]


class EvaluationReport(StrictModel):
    metadata: Provenance
    cases: list[CaseResult]
    overall: Aggregate
    domains: dict[Domain, Aggregate]
    languages: dict[Language, Aggregate]

    @model_validator(mode="after")
    def validate_case_set(self) -> EvaluationReport:
        expected = {
            (case_id, repetition)
            for case_id in self.metadata.case_ids
            for repetition in range(1, self.metadata.config.repetitions + 1)
        }
        actual = [(case.case_id, case.repetition) for case in self.cases]
        if (
            len(set(self.metadata.case_ids)) != len(self.metadata.case_ids)
            or len(set(actual)) != len(actual)
            or set(actual) != expected
        ):
            raise ValueError("Report case set does not match provenance")
        if any(
            case.provider != self.metadata.provider or case.model != self.metadata.model
            for case in self.cases
        ):
            raise ValueError("Report target does not match provenance")
        return self
