import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from app.evaluation.loader import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    dataset_hash,
    load_dataset,
)
from app.evaluation.models import EvaluationCase, Manifest, RunConfig
from app.evaluation.pricing import load_prices
from app.evaluation.runner import run_evaluation
from app.evaluation.targets import FixtureTarget


def fixture_report(name="target-a", *, streaming=False, repetitions=1, prices=False):
    dataset = load_dataset()
    target = FixtureTarget.load(dataset, name)
    catalog = (
        load_prices(PROJECT_ROOT / "evals/fixtures/v1/pricing.json") if prices else None
    )
    return asyncio.run(
        run_evaluation(
            dataset,
            target,
            RunConfig(streaming=streaming, repetitions=repetitions),
            prices=catalog,
            clock=target.clock,
        )
    )


def clone_dataset(tmp_path):
    root = tmp_path / "dataset"
    shutil.copytree(DEFAULT_DATASET, root)
    return root


def edit_manifest(root, **changes):
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def edit_case(root, file_domain, index=0, **changes):
    path = root / (file_domain + ".jsonl")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cases[index].update(changes)
    path.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")


def reseal_dataset(root: Path):
    manifest = Manifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    cases = []
    assets = {}
    for path in manifest.domain_files.values():
        for line in (root / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                case = EvaluationCase.model_validate_json(line)
                cases.append(case)
                for asset in case.text_assets:
                    assets[asset.path] = hashlib.sha256(
                        (root / asset.path).read_text(encoding="utf-8").encode()
                    ).hexdigest()
    cases.sort(key=lambda case: (case.domain, case.id))
    digest = dataset_hash(manifest, cases, assets)
    edit_manifest(root, content_hash=digest)
    return digest
