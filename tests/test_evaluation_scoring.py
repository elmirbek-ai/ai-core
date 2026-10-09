import pytest

from app.evaluation.loader import load_dataset
from app.evaluation.models import ScorerSpec
from app.evaluation.scoring import normalize, score_case, score_one, type_safe_equal


def spec(kind, **config):
    return ScorerSpec.model_validate(
        {"kind": kind, "metric": "accuracy_score", **config}
    )


@pytest.mark.parametrize(
    ("output", "expected", "score"),
    [
        ("  HELLO\nWORLD ", "hello world", 1),
        ("\uff21\uff22\uff23", "abc", 1),
        ("wrong", "right", 0),
    ],
)
def test_normalized_exact(output, expected, score):
    assert score_one(spec("exact", expected=expected), output) == score


@pytest.mark.parametrize(
    ("output", "score"),
    [
        ("7.04", 1),
        ("7.2", 0),
        ("nan", 0),
        ("inf", 0),
        ("Final answer: 7", 0),
        ("private chain of thought", 0),
    ],
)
def test_numeric_final_answer_and_tolerance(output, score):
    assert score_one(spec("numeric", expected=7.0, tolerance=0.05), output) == score


@pytest.mark.parametrize(
    ("output", "score"),
    [
        ('{"name":"Aida","age":28}', 1),
        ('{"name":"Aida","extra":"optional"}', 0.5),
        ('{"name":"Aida","age":true}', 0.5),
        ('{"name":"Aida","age":28.0}', 0.5),
        ("invalid", 0),
        ("[]", 0),
        ('{"age":NaN}', 0),
        ('{"name":"wrong","name":"Aida","age":28}', 0),
    ],
)
def test_json_extraction_is_type_safe_and_malformed_output_scores_zero(output, score):
    assert (
        score_one(spec("json", expected={"name": "Aida", "age": 28}), output) == score
    )


def test_nested_json_type_safety():
    assert type_safe_equal({"a": [1, {"b": True}]}, {"a": [1, {"b": True}]})
    assert not type_safe_equal({"a": [1]}, {"a": [True]})
    assert not type_safe_equal([1], [1, 2])


def test_term_coverage_and_forbidden_claims():
    assert score_one(spec("required_terms", terms=["red", "blue"]), "Red only") == 0.5
    assert score_one(spec("forbidden_terms", terms=["invented"]), "supported") == 1
    assert score_one(spec("forbidden_terms", terms=["invented"]), "invented") == 0
    forbidden = ScorerSpec(
        kind="forbidden_claims", metric="groundedness_score", terms=["900 units"]
    )
    assert score_one(forbidden, "Delivered 900 units") == 0


@pytest.mark.parametrize(
    ("format_name", "good", "bad"),
    [
        ("single_line", "one", "one\ntwo"),
        ("bullets", "- one\n- two", "one\ntwo"),
        ("json", '{"a":1}', "not JSON"),
        ("python", "x = 1", "def broken(:"),
    ],
)
def test_output_format_constraints(format_name, good, bad):
    scorer = spec("format", output_format=format_name)
    assert score_one(scorer, good) == 1
    assert score_one(scorer, bad) == 0
    if format_name in {"single_line", "bullets", "python"}:
        assert score_one(scorer, "") == 0


@pytest.mark.parametrize(
    ("output", "score"), [("one two", 1), ("one", 0), ("one two three four", 0)]
)
def test_length_constraints(output, score):
    assert score_one(spec("length", min_words=2, max_words=3), output) == score


def test_overlap_f1_is_multiset_based():
    assert score_one(spec("overlap_f1", expected="one two"), "one") == pytest.approx(
        2 / 3
    )
    assert score_one(spec("overlap_f1", expected=""), "") == 1


def test_python_syntax_required_symbols_and_no_code_execution(tmp_path):
    target = tmp_path / "must-not-exist"
    malicious = f"open({str(target)!r}, 'w').write('bad')"
    assert score_one(spec("python_syntax"), malicious) == 1
    assert not target.exists()
    scorer = spec("python_syntax", required_symbols=["add"])
    assert score_one(scorer, "def add(a,b): return a+b") == 1
    assert score_one(scorer, "def other(): pass") == 0
    assert score_one(scorer, "def add(:") == 0


def test_applicability_explicit_weights_grounding_and_hallucination():
    dataset = load_dataset()
    simple = next(case for case in dataset.cases if case.id == "general-capital")
    scores = score_case(simple, "Bishkek", dataset.manifest.quality_weights)
    assert (
        scores.quality_score
        == scores.accuracy_score
        == scores.instruction_following_score
        == 1
    )
    assert (
        scores.reasoning_score
        is scores.groundedness_score
        is scores.hallucination_flag
        is None
    )
    weighted = score_case(
        simple,
        "Bishkek\nextra",
        {"accuracy_score": 3.0, "instruction_following_score": 1.0},
    )
    assert weighted.quality_score == 0
    partial = score_case(
        simple, "wrong", {"accuracy_score": 3.0, "instruction_following_score": 1.0}
    )
    assert partial.quality_score == 0.25
    summary = next(case for case in dataset.cases if case.id == "summary-delivery")
    hallucinated = score_case(
        summary, "Aida delivered 900 units in Osh", dataset.manifest.quality_weights
    )
    assert hallucinated.hallucination_flag is True
    assert hallucinated.groundedness_score == pytest.approx(0.125)
    assert (
        normalize("\u0411\u0418\u0428\u041a\u0415\u041a")
        == "\u0431\u0438\u0448\u043a\u0435\u043a"
    )
