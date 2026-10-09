from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.evaluation.models import EvaluationCase, EvaluationError, Manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "evals" / "datasets" / "v1"


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not path.is_relative_to(root.resolve())
        or not path.is_file()
    ):
        raise EvaluationError("Dataset file reference is missing or outside its root")
    return path


@dataclass(frozen=True)
class Dataset:
    root: Path
    manifest: Manifest
    cases: tuple[EvaluationCase, ...]
    hash: str
    scorer_hash: str
    assets: Mapping[str, str]

    def messages(self, case: EvaluationCase) -> list[dict[str, Any]]:
        messages = [message.model_dump(mode="json") for message in case.messages]
        for asset in case.text_assets:
            text = self.assets[asset.path]
            index = asset.message_index
            messages[index]["content"] = messages[index]["content"].replace(
                "{{asset:" + asset.name + "}}", text
            )
        return messages


def dataset_hash(
    manifest: Manifest, cases: list[EvaluationCase], assets: dict[str, str]
) -> str:
    return content_hash(
        {
            "manifest": manifest.model_dump(mode="json", exclude={"content_hash"}),
            "cases": [case.model_dump(mode="json") for case in cases],
            "assets": assets,
        }
    )


def load_dataset(root: Path = DEFAULT_DATASET) -> Dataset:
    try:
        root = root.resolve()
        manifest = Manifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        cases = []
        assets = {}
        asset_texts = {}
        for domain, relative in sorted(manifest.domain_files.items()):
            lines = (
                contained_path(root, relative).read_text(encoding="utf-8").splitlines()
            )
            for line in lines:
                if not line.strip():
                    continue
                case = EvaluationCase.model_validate_json(line)
                if case.domain != domain:
                    raise EvaluationError("Dataset case domain does not match its file")
                cases.append(case)
                for asset in case.text_assets:
                    text = contained_path(root, asset.path).read_text(encoding="utf-8")
                    asset_texts[asset.path] = text
                    # Canonical UTF-8/LF assets survive Git autocrlf checkout.
                    assets[asset.path] = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()
        cases.sort(key=lambda case: (case.domain, case.id))
        if len(cases) != manifest.case_count or len({case.id for case in cases}) != len(
            cases
        ):
            raise EvaluationError("Dataset count or unique case IDs are invalid")
        if {case.language for case in cases} != set(manifest.supported_languages):
            raise EvaluationError("Dataset languages do not match the manifest")
        for case in cases:
            if any(
                scorer.metric not in manifest.quality_weights for scorer in case.scorers
            ):
                raise EvaluationError(
                    "Dataset scorer metric has no declared quality weight"
                )
        digest = dataset_hash(manifest, cases, assets)
        if digest != manifest.content_hash:
            raise EvaluationError(
                "Dataset content checksum does not match the manifest"
            )
        scorer_digest = content_hash(
            {
                "weights": manifest.quality_weights,
                "cases": {
                    case.id: [scorer.model_dump(mode="json") for scorer in case.scorers]
                    for case in cases
                },
            }
        )
        return Dataset(
            root,
            manifest,
            tuple(cases),
            digest,
            scorer_digest,
            MappingProxyType(asset_texts),
        )
    except EvaluationError:
        raise
    except Exception:
        raise EvaluationError("Dataset validation failed") from None
