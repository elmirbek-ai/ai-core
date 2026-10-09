import json

import pytest
from evaluation_helpers import clone_dataset, edit_case, edit_manifest, reseal_dataset

from app.evaluation.loader import contained_path, load_dataset
from app.evaluation.models import EvaluationError


def test_versioned_starter_dataset_covers_all_domains_languages_and_pairs():
    dataset = load_dataset()
    assert dataset.manifest.schema_version == 1
    assert dataset.manifest.dataset_version == "1.0.0"
    assert len(dataset.cases) == dataset.manifest.case_count == 29
    assert len({case.domain for case in dataset.cases}) == 11
    assert {case.language for case in dataset.cases} == {"ky", "ru", "en"}
    assert len([case for case in dataset.cases if case.domain == "translation"]) == 4
    assert all(
        sum(case.domain == domain for case in dataset.cases) >= 2
        for domain in dataset.manifest.domain_files
    )
    assert dataset.hash == dataset.manifest.content_hash
    assert len(dataset.scorer_hash) == 64


def test_hash_and_order_ignore_json_layout_file_order_and_cwd(tmp_path, monkeypatch):
    before = load_dataset()
    root = clone_dataset(tmp_path)
    path = root / "general.jsonl"
    path.write_text(
        "\n\n"
        + "\n".join(reversed(path.read_text(encoding="utf-8").splitlines()))
        + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    monkeypatch.chdir(tmp_path)
    after = load_dataset(root)
    assert load_dataset().hash == before.hash == after.hash
    assert [case.id for case in before.cases] == [case.id for case in after.cases]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_asset_hash_is_checkout_line_ending_independent(tmp_path, newline):
    root = clone_dataset(tmp_path)
    asset = root / "assets/reference_archive.txt"
    text = asset.read_text(encoding="utf-8")
    asset.write_bytes(text.encode().replace(b"\n", newline))
    assert load_dataset(root).hash == load_dataset().hash


def test_assets_are_cached_and_messages_are_detached(tmp_path):
    root = clone_dataset(tmp_path)
    dataset = load_dataset(root)
    case = next(case for case in dataset.cases if case.text_assets)
    before = dataset.messages(case)
    assert len(before[0]["content"]) > 20000
    assert "ORBIT-137" in before[0]["content"]
    (root / case.text_assets[0].path).write_text("changed", encoding="utf-8")
    assert dataset.messages(case) == before
    before[0]["content"] = "modified"
    assert dataset.messages(case)[0]["content"] != "modified"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"case_count": 30},
        {"supported_languages": ["en"]},
        {"content_hash": "0" * 64},
        {"quality_weights": {"accuracy_score": 1.0}},
    ],
)
def test_invalid_manifest_consistency_is_rejected(tmp_path, change):
    root = clone_dataset(tmp_path)
    edit_manifest(root, **change)
    with pytest.raises(EvaluationError):
        load_dataset(root)


def test_duplicate_case_id_rejected(tmp_path):
    root = clone_dataset(tmp_path)
    first_id = json.loads((root / "general.jsonl").read_text().splitlines()[0])["id"]
    edit_case(root, "general", index=1, id=first_id)
    with pytest.raises(EvaluationError, match="unique case IDs"):
        load_dataset(root)


@pytest.mark.parametrize(
    "changes",
    [
        {"domain": "reasoning"},
        {"domain": "invalid"},
        {"scorers": [{"kind": "unknown", "metric": "accuracy_score"}]},
        {"messages": [{"role": "user", "content": "changed prompt"}]},
    ],
)
def test_invalid_or_tampered_case_rejected(tmp_path, changes):
    root = clone_dataset(tmp_path)
    edit_case(root, "general", **changes)
    with pytest.raises(EvaluationError):
        load_dataset(root)


def test_missing_outside_and_absolute_files_rejected(tmp_path):
    root = clone_dataset(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("safe")
    for reference in ["missing.jsonl", "../outside.txt", str(outside)]:
        with pytest.raises(EvaluationError):
            contained_path(root, reference)
    edit_manifest(root, domain_files={"general": "../outside.txt"})
    with pytest.raises(EvaluationError):
        load_dataset(root)


def test_changed_asset_changes_dataset_digest(tmp_path):
    root = clone_dataset(tmp_path)
    asset = root / "assets/reference_archive.txt"
    asset.write_text(
        asset.read_text(encoding="utf-8") + "New fact.\n", encoding="utf-8"
    )
    with pytest.raises(EvaluationError, match="checksum"):
        load_dataset(root)
    digest = reseal_dataset(root)
    assert load_dataset(root).hash == digest != load_dataset().hash
