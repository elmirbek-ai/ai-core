from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from app.evaluation.models import (
    Aggregate,
    CaseResult,
    EvaluationError,
    EvaluationReport,
    Provenance,
    Scores,
    Statistics,
)


def percentile(values: list[float], percent: int) -> float | None:
    """Nearest rank: sorted[ceil(p / 100 * n) - 1], clamped for p=0."""
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent / 100 * len(ordered)) - 1)]


def statistics(values: list[float]) -> Statistics:
    return Statistics(
        count=len(values),
        mean=sum(values) / len(values) if values else None,
        p50=percentile(values, 50),
        p95=percentile(values, 95),
    )


def aggregate(cases: list[CaseResult]) -> Aggregate:
    successes = sum(case.status == "success" for case in cases)
    failures = sum(case.status == "failure" for case in cases)
    attempted = successes + failures
    metrics = {}
    availability = {}
    for metric in Scores.model_fields:
        values = [
            (getattr(case.scores, metric), case.weight)
            for case in cases
            if getattr(case.scores, metric) is not None
        ]
        availability[metric] = len(values)
        if metric != "hallucination_flag":
            metrics[metric] = (
                (
                    sum(value * weight for value, weight in values)
                    / sum(weight for _, weight in values)
                )
                if values
                else None
            )
    flags = [
        case.scores.hallucination_flag
        for case in cases
        if case.scores.hallucination_flag is not None
    ]
    costs = [case.cost_usd for case in cases if case.cost_usd is not None]
    return Aggregate(
        total=len(cases),
        attempted=attempted,
        successes=successes,
        failures=failures,
        skipped=len(cases) - attempted,
        success_rate=successes / attempted if attempted else None,
        failure_rate=failures / attempted if attempted else None,
        scores=Scores.model_validate(metrics),
        metric_availability=availability,
        latency=statistics(
            [case.latency_seconds for case in cases if case.latency_seconds is not None]
        ),
        ttft=statistics(
            [case.ttft_seconds for case in cases if case.ttft_seconds is not None]
        ),
        hallucination_rate=sum(bool(flag) for flag in flags) / len(flags)
        if flags
        else None,
        cost_available_count=len(costs),
        cost_total_usd=sum(costs) if costs else None,
        cost_average_usd=sum(costs) / len(costs) if costs else None,
        error_categories=dict(
            Counter(
                case.error_category for case in cases if case.error_category is not None
            )
        ),
    )


def build_report(metadata: Provenance, cases: list[CaseResult]) -> EvaluationReport:
    return EvaluationReport(
        metadata=metadata,
        cases=[case.model_copy(deep=True) for case in cases],
        overall=aggregate(cases),
        domains={
            domain: aggregate([case for case in cases if case.domain == domain])
            for domain in sorted({case.domain for case in cases})
        },
        languages={
            language: aggregate([case for case in cases if case.language == language])
            for language in sorted({case.language for case in cases})
        },
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# AI Core evaluation report",
        "",
        f"Evidence mode: **{report.metadata.mode}**.",
        "",
        "Fixture results are synthetic and are not provider admission evidence.",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(report.metadata.model_dump(mode="json"), indent=2, sort_keys=True),
        "```",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    overall = report.overall.model_dump(mode="json")
    for key in (
        "total",
        "attempted",
        "successes",
        "failures",
        "skipped",
        "success_rate",
        "failure_rate",
        "hallucination_rate",
        "cost_available_count",
        "cost_total_usd",
        "cost_average_usd",
    ):
        lines.append(f"| {key} | {_display(overall[key])} |")
    for metric, value in overall["scores"].items():
        if metric != "hallucination_flag":
            lines.append(f"| {metric} | {_display(value)} |")
    lines.extend(
        [
            "",
            "## Availability",
            "",
            "| Metric | Applicable scored outputs |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {metric} | {count} |"
        for metric, count in sorted(report.overall.metric_availability.items())
    )
    lines.extend(
        [
            "",
            "## Timings",
            "",
            "| Metric | Count | Mean | p50 | p95 |",
            "|---|---:|---|---|---|",
        ]
    )
    for name, timing in (
        ("Latency seconds", report.overall.latency),
        ("TTFT seconds", report.overall.ttft),
    ):
        lines.append(
            f"| {name} | {timing.count} | {_display(timing.mean)} | "
            f"{_display(timing.p50)} | {_display(timing.p95)} |"
        )
    for title, groups in (("Domains", report.domains), ("Languages", report.languages)):
        lines.extend(
            [
                "",
                "## " + title,
                "",
                "| Group | Total | Success | Failure | Skip | Quality | Accuracy | "
                "Instruction | Reasoning | Formatting | Groundedness |",
                "|---|---:|---:|---:|---:|---|---|---|---|---|---|",
            ]
        )
        for name, group in groups.items():
            values = [
                getattr(group.scores, metric)
                for metric in Scores.model_fields
                if metric != "hallucination_flag"
            ]
            lines.append(
                f"| {name} | {group.total} | {group.successes} | "
                f"{group.failures} | {group.skipped} | "
                + " | ".join(_display(value) for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Case outcomes",
            "",
            "| Case | Repetition | Status | Category | Output characters | Quality |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    lines.extend(
        f"| {case.case_id} | {case.repetition} | {case.status} | "
        f"{_display(case.error_category or case.skip_reason)} | "
        f"{case.output_chars} | {_display(case.scores.quality_score)} |"
        for case in report.cases
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Scores are deterministic lexical/structural proxies. Translation "
            "semantics, code behavior and general hallucinations "
            "require additional evidence.",
            "NULL means not applicable or unavailable; failures/skips do not "
            "receive fabricated zero quality scores.",
            "Score means cover successful applicable outputs; compare "
            "reliability and capability coverage alongside quality.",
            "Cost totals cover only outcomes with reliable usage and matching "
            "prices; incomplete availability is not full run spend.",
            "Percentiles use nearest rank. No responses, prompts, exception "
            "bodies or asset URLs are persisted.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: Any) -> str:
    if value is None:
        return "not_applicable / unavailable"
    return str(value)


def write_report(report: EvaluationReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (directory / "evaluation.md").write_text(render_markdown(report), encoding="utf-8")


def read_report(path: Path) -> EvaluationReport:
    try:
        report = EvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))
        rebuilt = build_report(report.metadata, report.cases)
        if report != rebuilt:
            raise EvaluationError("Report aggregates do not match its case results")
        return report
    except EvaluationError:
        raise
    except Exception:
        raise EvaluationError("Evaluation report validation failed") from None


def compare_reports(left: EvaluationReport, right: EvaluationReport) -> dict[str, Any]:
    keys = (
        "schema_version",
        "framework_version",
        "dataset_version",
        "dataset_hash",
        "scorer_config_hash",
        "case_ids",
        "mode",
    )
    if (
        any(getattr(left.metadata, key) != getattr(right.metadata, key) for key in keys)
        or left.metadata.config != right.metadata.config
        or left.metadata.asset_mapping_hash != right.metadata.asset_mapping_hash
    ):
        raise EvaluationError(
            "Reports have incompatible datasets, cases or run configurations"
        )
    score_differences = {
        metric: _difference(
            getattr(left.overall.scores, metric), getattr(right.overall.scores, metric)
        )
        for metric in Scores.model_fields
        if metric != "hallucination_flag"
    }
    domains = {
        name: {
            metric: _difference(
                getattr(left.domains[name].scores, metric),
                getattr(right.domains[name].scores, metric),
            )
            for metric in Scores.model_fields
            if metric != "hallucination_flag"
        }
        for name in left.domains
    }
    languages = {
        name: {
            metric: _difference(
                getattr(left.languages[name].scores, metric),
                getattr(right.languages[name].scores, metric),
            )
            for metric in Scores.model_fields
            if metric != "hallucination_flag"
        }
        for name in left.languages
    }
    right_cases = {(case.case_id, case.repetition): case for case in right.cases}
    common = [
        (case, right_cases[(case.case_id, case.repetition)])
        for case in left.cases
        if case.status == "success"
        and right_cases[(case.case_id, case.repetition)].status == "success"
        and case.scores.quality_score is not None
        and right_cases[(case.case_id, case.repetition)].scores.quality_score
        is not None
    ]
    common_delta = (
        sum(
            ((b.scores.quality_score or 0.0) - (a.scores.quality_score or 0.0))
            * a.weight
            for a, b in common
        )
        / sum(a.weight for a, _ in common)
        if common
        else None
    )
    complete_cost = (
        left.overall.cost_available_count == left.overall.attempted > 0
        and right.overall.cost_available_count == right.overall.attempted > 0
        and left.metadata.price_catalog_hash == right.metadata.price_catalog_hash
    )
    return {
        "direction": "right_minus_left",
        "left": left.metadata.model_dump(mode="json"),
        "right": right.metadata.model_dump(mode="json"),
        "score_differences": score_differences,
        "domain_score_differences": domains,
        "language_score_differences": languages,
        "latency_mean_difference_seconds": _difference(
            left.overall.latency.mean, right.overall.latency.mean
        ),
        "ttft_mean_difference_seconds": _difference(
            left.overall.ttft.mean, right.overall.ttft.mean
        ),
        "failure_rate_difference": _difference(
            left.overall.failure_rate, right.overall.failure_rate
        ),
        "hallucination_rate_difference": _difference(
            left.overall.hallucination_rate, right.overall.hallucination_rate
        ),
        "cost_total_difference_usd": _difference(
            left.overall.cost_total_usd, right.overall.cost_total_usd
        )
        if complete_cost
        else None,
        "common_successful_case_count": len(common),
        "common_case_quality_difference": common_delta,
        "left_summary": left.overall.model_dump(mode="json"),
        "right_summary": right.overall.model_dump(mode="json"),
        "note": "Quality means exclude failed/skipped/unavailable outcomes. "
        "Fixture evidence is synthetic; no routing decisions are made.",
    }


def _difference(left: float | None, right: float | None) -> float | None:
    return right - left if left is not None and right is not None else None


def write_comparison(comparison: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False)
    (directory / "comparison.json").write_text(encoded + "\n", encoding="utf-8")
    # Preserve all comparison metadata, metrics and coverage in the Markdown
    # companion; the JSON is inspectable and never contains output content.
    (directory / "comparison.md").write_text(
        "# AI Core evaluation comparison\n\n"
        "Differences are right minus left.\n\n```json\n" + encoded + "\n```\n",
        encoding="utf-8",
    )
