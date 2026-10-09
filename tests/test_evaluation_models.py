import pytest
from pydantic import ValidationError

from app.evaluation.loader import load_dataset
from app.evaluation.models import (
    CaseResult,
    EvaluationCase,
    Manifest,
    RunConfig,
    ScorerSpec,
    Scores,
    TargetOutput,
)


@pytest.mark.parametrize(
    "change",
    [
        {"domain": "unknown"},
        {"language": "fr"},
        {"id": "bad\nID"},
        {"weight": 0},
        {"weight": -1},
        {"weight": float("nan")},
        {"weight": float("inf")},
        {"messages": []},
        {"messages": [{"role": "invalid", "content": "test"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"Authorization": "TEST_SECRET_DO_NOT_LOG"},
        {"scorers": [{"kind": "unknown", "metric": "accuracy_score"}]},
    ],
)
def test_case_validation_rejects_invalid_values(change):
    data = load_dataset().cases[0].model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(data)


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "exact"},
        {"kind": "overlap_f1", "expected": 1},
        {"kind": "numeric", "expected": 7},
        {"kind": "numeric", "expected": "7", "tolerance": 0},
        {"kind": "numeric", "expected": True, "tolerance": 0},
        {"kind": "numeric", "expected": float("inf"), "tolerance": 0},
        {"kind": "numeric", "expected": 7, "tolerance": -1},
        {"kind": "json", "expected": {}},
        {"kind": "json", "expected": "{}"},
        {"kind": "required_terms", "terms": []},
        {"kind": "required_terms", "terms": [" "]},
        {"kind": "required_terms", "terms": ["a", "a"]},
        {"kind": "forbidden_terms"},
        {"kind": "format"},
        {"kind": "length"},
        {"kind": "length", "min_words": 3, "max_words": 1},
        {"kind": "exact", "expected": "x", "terms": ["x"]},
        {"kind": "grounded_facts", "terms": ["fact"]},
        {"kind": "forbidden_claims", "terms": ["claim"]},
        {"kind": "python_syntax", "required_symbols": ["bad-name"]},
        {"kind": "python_syntax", "required_symbols": ["class"]},
        {"kind": "python_syntax", "required_symbols": ["add", "add"]},
    ],
)
def test_impossible_scorer_configs_are_rejected(spec):
    with pytest.raises(ValidationError):
        ScorerSpec.model_validate({"metric": "accuracy_score", **spec})


def test_image_and_asset_requirements_validated():
    dataset = load_dataset()
    image = next(case for case in dataset.cases if case.domain == "multimodal")
    data = image.model_dump(mode="json")
    data["requirements"]["images"] = False
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(data)
    asset_case = next(case for case in dataset.cases if case.text_assets)
    for changes in [{"message_index": 999}, {"name": "missing"}]:
        data = asset_case.model_dump(mode="json")
        data["text_assets"][0].update(changes)
        with pytest.raises(ValidationError):
            EvaluationCase.model_validate(data)
    data = asset_case.model_dump(mode="json")
    data["text_assets"].append(dict(data["text_assets"][0]))
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"case_count": 0},
        {"quality_weights": {}},
        {"supported_languages": ["xx"]},
        {"content_hash": "bad"},
    ],
)
def test_manifest_schema_is_strict(change):
    data = load_dataset().manifest.model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError):
        Manifest.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"repetitions": 0},
        {"repetitions": 101},
        {"timeout_seconds": 0},
        {"max_output_chars": 0},
        {"api_key": "private"},
    ],
)
def test_run_config_bounds(change):
    with pytest.raises(ValidationError):
        RunConfig.model_validate(change)


@pytest.mark.parametrize("change", [{"input_tokens": -1}, {"output_tokens": True}])
def test_token_usage_validation(change):
    with pytest.raises(ValidationError):
        TargetOutput.model_validate({"content": "ok", **change})


def test_null_score_and_outcome_consistency():
    assert all(value is None for value in Scores().model_dump().values())
    base = dict(
        case_id="case",
        domain="general",
        language="en",
        repetition=1,
        weight=1.0,
        provider="fixture-a",
        model="synthetic-a",
    )
    for changes in [
        {"status": "success"},
        {"status": "failure", "latency_seconds": 1.0},
        {"status": "skipped"},
        {"status": "skipped", "skip_reason": "capability", "latency_seconds": 1.0},
        {
            "status": "failure",
            "error_category": "provider",
            "latency_seconds": 1.0,
            "scores": {"quality_score": 0},
        },
        {"status": "success", "latency_seconds": 1.0, "cost_status": "computed"},
    ]:
        with pytest.raises(ValidationError):
            CaseResult.model_validate({**base, **changes})
