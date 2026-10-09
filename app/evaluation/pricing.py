from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.evaluation.loader import content_hash
from app.evaluation.models import (
    EvaluationError,
    ModelName,
    SafeName,
    StrictModel,
    TargetOutput,
)


class PriceEntry(StrictModel):
    provider: SafeName
    model: ModelName
    input_usd_per_million: Annotated[Decimal, Field(ge=0)]
    output_usd_per_million: Annotated[Decimal, Field(ge=0)]


class PriceCatalog(StrictModel):
    schema_version: Literal[1]
    version: SafeName
    price_date: date
    currency: Literal["USD"]
    evidence: Literal["operator_supplied", "synthetic_fixture"]
    entries: list[PriceEntry] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_entries(self) -> PriceCatalog:
        if len({(entry.provider, entry.model) for entry in self.entries}) != len(
            self.entries
        ):
            raise ValueError("Duplicate pricing target")
        return self

    @property
    def hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def load_prices(path: Path) -> PriceCatalog:
    try:
        return PriceCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        raise EvaluationError("Price catalog validation failed") from None


def compute_cost(
    provider: str, model: str, output: TargetOutput, catalog: PriceCatalog | None
) -> float | None:
    if (
        catalog is None
        or not output.usage_reliable
        or output.input_tokens is None
        or output.output_tokens is None
    ):
        return None
    entry = next(
        (
            price
            for price in catalog.entries
            if price.provider == provider and price.model == model
        ),
        None,
    )
    if entry is None:
        return None
    amount = (
        Decimal(output.input_tokens) * entry.input_usd_per_million
        + Decimal(output.output_tokens) * entry.output_usd_per_million
    ) / Decimal(1000000)
    return float(amount)
