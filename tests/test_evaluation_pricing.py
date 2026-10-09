import json

import pytest

from app.evaluation.loader import PROJECT_ROOT
from app.evaluation.models import EvaluationError, TargetOutput
from app.evaluation.pricing import compute_cost, load_prices


def prices():
    return load_prices(PROJECT_ROOT / "evals/fixtures/v1/pricing.json")


def test_fake_price_catalog_and_reliable_usage_compute_cost():
    output = TargetOutput(
        content="ignored", input_tokens=100, output_tokens=20, usage_reliable=True
    )
    assert compute_cost("fixture-a", "synthetic-a", output, prices()) == pytest.approx(
        0.000014
    )
    assert prices().currency == "USD"
    assert prices().evidence == "synthetic_fixture"
    assert len(prices().hash) == 64
    assert (
        compute_cost(
            "fixture-a",
            "synthetic-a",
            output.model_copy(update={"input_tokens": 0, "output_tokens": 0}),
            prices(),
        )
        == 0
    )


@pytest.mark.parametrize(
    "changes",
    [{"input_tokens": None}, {"output_tokens": None}, {"usage_reliable": False}],
)
def test_missing_or_unreliable_usage_is_unavailable(changes):
    output = TargetOutput(
        content="ignored", input_tokens=100, output_tokens=20, usage_reliable=True
    ).model_copy(update=changes)
    assert compute_cost("fixture-a", "synthetic-a", output, prices()) is None


def test_missing_catalog_and_price_entry_are_unavailable():
    output = TargetOutput(
        content="ignored", input_tokens=100, output_tokens=20, usage_reliable=True
    )
    assert compute_cost("fixture-a", "synthetic-a", output, None) is None
    assert compute_cost("fixture-a", "missing-model", output, prices()) is None


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"currency": "EUR"},
        {"price_date": "not-a-date"},
        {"entries": []},
    ],
)
def test_invalid_catalog_is_rejected_without_raw_input(tmp_path, change):
    catalog = prices().model_dump(mode="json")
    catalog.update(change)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(EvaluationError, match="Price catalog validation failed"):
        load_prices(path)


def test_duplicate_and_negative_catalog_entries_rejected(tmp_path):
    catalog = prices().model_dump(mode="json")
    path = tmp_path / "prices.json"
    catalog["entries"].append(dict(catalog["entries"][0]))
    path.write_text(json.dumps(catalog))
    with pytest.raises(EvaluationError):
        load_prices(path)
    catalog["entries"].pop()
    catalog["entries"][0]["input_usd_per_million"] = "-1"
    path.write_text(json.dumps(catalog))
    with pytest.raises(EvaluationError):
        load_prices(path)
