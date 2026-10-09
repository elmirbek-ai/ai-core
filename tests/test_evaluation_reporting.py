import json

import pytest
from evaluation_helpers import fixture_report
from pydantic import ValidationError

from app.evaluation.models import EvaluationError, EvaluationReport
from app.evaluation.reporting import (
    compare_reports,
    percentile,
    read_report,
    render_markdown,
    write_comparison,
    write_report,
)


def test_json_markdown_aggregation_grouping_and_detached_results(tmp_path):
    report = fixture_report(prices=True)
    write_report(report, tmp_path)
    assert read_report(tmp_path / "evaluation.json") == report
    markdown = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert (
        "Provenance" in markdown and "Domains" in markdown and "Languages" in markdown
    )
    assert "synthetic" in markdown
    assert report.overall.total == sum(group.total for group in report.domains.values())
    assert report.overall.total == sum(
        group.total for group in report.languages.values()
    )
    assert report.domains["reasoning"].scores.reasoning_score == 1
    assert report.domains["code"].scores.formatting_score == 1
    assert report.overall.metric_availability["groundedness_score"] == 3
    assert report.overall.metric_availability["hallucination_flag"] == 3
    serialized = report.model_dump()
    serialized["cases"][0]["scores"]["quality_score"] = 0
    serialized["metadata"]["case_ids"].clear()
    assert report.cases[0].scores.quality_score == 1
    assert len(report.metadata.case_ids) == 29
    assert "content" not in serialized["cases"][0]
    assert report.overall.cost_total_usd == pytest.approx(29 * 0.000014)


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        ([], 50, None),
        ([5, 1, 3, 2, 4], 50, 3),
        ([5, 1, 3, 2, 4], 95, 5),
        ([5, 1], 0, 1),
        ([5, 1], 100, 5),
    ],
)
def test_nearest_rank_percentiles(values, percent, expected):
    assert percentile(values, percent) == expected


@pytest.mark.parametrize("percent", [-1, 101])
def test_invalid_percentiles_rejected(percent):
    with pytest.raises(ValueError):
        percentile([1], percent)


def test_compatible_fixture_comparison_contains_all_dimensions(tmp_path):
    left = fixture_report(streaming=True, prices=True)
    right = fixture_report("target-b", streaming=True, prices=True)
    comparison = compare_reports(left, right)
    assert comparison["score_differences"]["quality_score"] < 0
    assert comparison["latency_mean_difference_seconds"] == pytest.approx(0.06)
    assert comparison["ttft_mean_difference_seconds"] == pytest.approx(0.02)
    assert comparison["failure_rate_difference"] == pytest.approx(4 / 29)
    assert comparison["common_successful_case_count"] == 25
    assert comparison["common_case_quality_difference"] < 0
    assert len(comparison["domain_score_differences"]) == 11
    assert len(comparison["language_score_differences"]) == 3
    assert comparison["cost_total_difference_usd"] is None  # Failed-call costs unknown.
    write_comparison(comparison, tmp_path)
    assert json.loads((tmp_path / "comparison.json").read_text()) == comparison
    assert "right minus left" in (tmp_path / "comparison.md").read_text()
    comparison["left"]["case_ids"].clear()
    assert left.metadata.case_ids


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_version": "2.0.0"},
        {"dataset_hash": "0" * 64},
        {"scorer_config_hash": "1" * 64},
        {"case_ids": ["different"]},
        {"mode": "live"},
        {"asset_mapping_hash": "2" * 64},
    ],
)
def test_incompatible_comparison_is_rejected(change):
    report = fixture_report()
    right = report.model_copy(
        update={"metadata": report.metadata.model_copy(update=change)}
    )
    with pytest.raises(EvaluationError, match="incompatible"):
        compare_reports(report, right)


def test_incompatible_run_configuration_is_rejected():
    with pytest.raises(EvaluationError):
        compare_reports(fixture_report(), fixture_report(streaming=True))


def test_cost_difference_requires_complete_matching_price_evidence():
    report = fixture_report(prices=True)
    assert compare_reports(report, report)["cost_total_difference_usd"] == 0
    unavailable = fixture_report()
    assert (
        compare_reports(unavailable, unavailable)["cost_total_difference_usd"] is None
    )


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "target"])
def test_report_case_set_and_target_are_validated(mutation):
    data = fixture_report().model_dump(mode="json")
    if mutation == "duplicate":
        data["cases"].append(dict(data["cases"][0]))
    elif mutation == "missing":
        data["cases"].pop()
    else:
        data["cases"][0]["model"] = "different"
    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(data)


def test_report_reader_rejects_bad_json_unknown_fields_and_inconsistent_aggregates(
    tmp_path,
):
    path = tmp_path / "report.json"
    path.write_text("not JSON")
    with pytest.raises(EvaluationError, match="validation failed"):
        read_report(path)
    data = fixture_report().model_dump(mode="json")
    data["overall"]["successes"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(EvaluationError, match="aggregates"):
        read_report(path)
    data["api_key"] = "TEST_SECRET_DO_NOT_LOG"
    path.write_text(json.dumps(data))
    with pytest.raises(EvaluationError, match="validation failed"):
        read_report(path)


def test_missing_metrics_markdown_shows_unavailable():
    markdown = render_markdown(fixture_report())
    assert "not_applicable / unavailable" in markdown
    assert "incomplete availability is not full run spend" in markdown
