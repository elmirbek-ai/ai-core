import asyncio
from dataclasses import replace

import pytest
from evaluation_helpers import clone_dataset, edit_case
from test_evaluation_targets import Provider

from app.evaluation.cli import main
from app.evaluation.loader import load_dataset
from app.evaluation.models import RunConfig
from app.evaluation.reporting import render_markdown
from app.evaluation.runner import run_evaluation
from app.evaluation.targets import ExistingProviderTarget
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)

SECRET = "TEST_SECRET_DO_NOT_LOG"


@pytest.mark.parametrize(
    ("error_type", "category"),
    [
        (LLMAuthenticationError, "authentication"),
        (LLMRateLimitError, "rate_limit"),
        (LLMTimeoutError, "timeout"),
        (LLMUpstreamError, "upstream"),
        (LLMProviderError, "provider"),
        (RuntimeError, "provider"),
    ],
)
def test_raw_provider_error_environment_secret_and_authorization_never_persist(
    error_type, category, monkeypatch, caplog
):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("EVAL_TEST_API_KEY", SECRET + "_KEY")
    dataset = load_dataset()
    provider = Provider(
        error=error_type(
            SECRET
            + "_BODY Authorization: Bearer "
            + SECRET
            + "_KEY https://private.invalid/?token="
            + SECRET
        )
    )
    target = ExistingProviderTarget(provider, dataset, "selected-model", live=True)
    report = asyncio.run(
        run_evaluation(replace(dataset, cases=(dataset.cases[0],)), target, RunConfig())
    )
    assert report.cases[0].error_category == category
    assert provider.closed
    assert (
        SECRET not in report.model_dump_json() + render_markdown(report) + caplog.text
    )
    assert "private.invalid" not in report.model_dump_json()


@pytest.mark.parametrize("streaming", [False, True])
def test_generated_response_content_never_persists_in_standard_reports(
    streaming, monkeypatch, caplog
):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    dataset = load_dataset()
    provider = Provider(content=SECRET + "_RESPONSE")
    target = ExistingProviderTarget(provider, dataset, "selected-model", live=True)
    report = asyncio.run(
        run_evaluation(
            replace(dataset, cases=(dataset.cases[0],)),
            target,
            RunConfig(streaming=streaming),
        )
    )
    assert report.cases[0].status == "success"
    assert report.cases[0].output_chars == len(SECRET + "_RESPONSE")
    assert (
        SECRET not in report.model_dump_json() + render_markdown(report) + caplog.text
    )
    assert report.cases[0].cost_usd is None


def test_bad_dataset_cli_error_hides_sensitive_message(tmp_path, capsys):
    root = clone_dataset(tmp_path)
    edit_case(root, "general", messages=[{"role": "invalid", "content": SECRET}])
    assert main(["validate", "--dataset", str(root)]) == 2
    output = capsys.readouterr().err
    assert "Dataset validation failed" in output
    assert SECRET not in output


def test_cli_unexpected_exception_is_sanitized(monkeypatch, capsys):
    def broken(*args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr("app.evaluation.cli.load_dataset", broken)
    assert main(["validate"]) == 2
    output = capsys.readouterr().err
    assert "Evaluation operation failed" in output
    assert SECRET not in output


@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_provider_output_becomes_safe_failure(streaming, monkeypatch):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    dataset = load_dataset()
    provider = Provider(content=None)
    target = ExistingProviderTarget(provider, dataset, "selected-model", live=True)
    report = asyncio.run(
        run_evaluation(
            replace(dataset, cases=(dataset.cases[0],)),
            target,
            RunConfig(streaming=streaming),
        )
    )
    assert report.cases[0].error_category == "invalid_output"


def test_partial_stream_failure_retains_ttft_but_no_quality_or_response(monkeypatch):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    dataset = load_dataset()
    provider = Provider(content=SECRET, error=LLMUpstreamError(SECRET + "_BODY"))
    target = ExistingProviderTarget(provider, dataset, "selected-model", live=True)
    report = asyncio.run(
        run_evaluation(
            replace(dataset, cases=(dataset.cases[0],)),
            target,
            RunConfig(streaming=True),
        )
    )
    case = report.cases[0]
    assert case.status == "failure"
    assert case.error_category == "upstream"
    assert case.ttft_seconds is not None
    assert case.scores.quality_score is None
    assert provider.stream_closed and provider.closed
    assert SECRET not in report.model_dump_json() + render_markdown(report)
