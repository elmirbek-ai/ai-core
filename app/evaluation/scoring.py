from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from collections import Counter

from app.evaluation.models import EvaluationCase, Metric, ScorerSpec, Scores


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def type_safe_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            type_safe_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            type_safe_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def parse_json(text: str) -> object:
    def reject_nonfinite(value: str) -> None:
        raise ValueError("Nonfinite JSON number")

    def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_nonfinite,
        object_pairs_hook=unique_fields,
    )


def score_one(spec: ScorerSpec, output: str) -> float:
    text = normalize(output)
    match spec.kind:
        case "exact":
            return float(text == normalize(str(spec.expected)))
        case "required_terms" | "grounded_facts":
            terms = spec.terms or []
            return sum(normalize(term) in text for term in terms) / len(terms)
        case "forbidden_terms" | "forbidden_claims":
            return float(not any(normalize(term) in text for term in spec.terms or []))
        case "numeric":
            # Score observable final answers only, never reasoning traces.
            assert isinstance(spec.expected, float) and spec.tolerance is not None
            try:
                value = float(output.strip())
                return float(
                    math.isfinite(value)
                    and abs(value - spec.expected) <= spec.tolerance
                )
            except ValueError:
                return 0.0
        case "json":
            try:
                parsed = parse_json(output)
            except (ValueError, RecursionError):
                return 0.0
            expected = spec.expected
            assert isinstance(expected, dict)
            if not isinstance(parsed, dict):
                return 0.0
            return sum(
                key in parsed and type_safe_equal(parsed[key], value)
                for key, value in expected.items()
            ) / len(expected)
        case "format":
            match spec.output_format:
                case "single_line":
                    return float(
                        bool(output.strip())
                        and "\n" not in output.strip()
                        and "\r" not in output.strip()
                    )
                case "bullets":
                    lines = output.strip().splitlines()
                    return float(
                        bool(lines)
                        and all(
                            line.startswith("- ") and line[2:].strip() for line in lines
                        )
                    )
                case "json":
                    try:
                        parse_json(output)
                        return 1.0
                    except (ValueError, RecursionError):
                        return 0.0
                case "python":
                    return _python_score(output, [])
        case "overlap_f1":
            actual, expected_words = (
                Counter(words(output)),
                Counter(words(str(spec.expected))),
            )
            overlap = sum((actual & expected_words).values())
            denominator = sum(actual.values()) + sum(expected_words.values())
            return 2 * overlap / denominator if denominator else 1.0
        case "python_syntax":
            return _python_score(output, spec.required_symbols or [])
        case "length":
            count = len(words(output))
            return float(
                (spec.min_words is None or count >= spec.min_words)
                and (spec.max_words is None or count <= spec.max_words)
            )
    raise ValueError("Unsupported scorer")


def _python_score(output: str, required_symbols: list[str]) -> float:
    try:
        tree = ast.parse(output)
    except (SyntaxError, ValueError, RecursionError):
        return 0.0
    symbols = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    return float(bool(output.strip()) and set(required_symbols) <= symbols)


def score_case(
    case: EvaluationCase, output: str, quality_weights: dict[Metric, float]
) -> Scores:
    metrics: dict[Metric, float | None] = {}
    for metric in quality_weights:
        applicable = [spec for spec in case.scorers if spec.metric == metric]
        metrics[metric] = (
            (
                sum(score_one(spec, output) * spec.weight for spec in applicable)
                / sum(spec.weight for spec in applicable)
            )
            if applicable
            else None
        )
    active = [(metric, value) for metric, value in metrics.items() if value is not None]
    quality = (
        (
            sum(float(value) * quality_weights[metric] for metric, value in active)
            / sum(quality_weights[metric] for metric, _ in active)
        )
        if active
        else None
    )
    hallucination_checks = [
        spec for spec in case.scorers if spec.kind == "forbidden_claims"
    ]
    hallucination = (
        any(score_one(spec, output) == 0 for spec in hallucination_checks)
        if hallucination_checks
        else None
    )
    return Scores.model_validate(
        {**metrics, "quality_score": quality, "hallucination_flag": hallucination}
    )
