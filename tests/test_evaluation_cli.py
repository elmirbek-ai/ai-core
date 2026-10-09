import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.evaluation.cli import main
from app.evaluation.loader import PROJECT_ROOT
from app.evaluation.targets import FixtureTarget


def test_validate_is_offline_and_cwd_independent(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["validate"]) == 0
    output = capsys.readouterr().out
    assert "cases=29" in output and "sha256=" in output


def test_fixture_cli_run_and_compare_end_to_end(tmp_path, capsys):
    prices = PROJECT_ROOT / "evals/fixtures/v1/pricing.json"
    for name in ("target-a", "target-b"):
        assert (
            main(
                [
                    "run",
                    "--fixture",
                    name,
                    "--streaming",
                    "--repetitions",
                    "2",
                    "--prices",
                    str(prices),
                    "--output-dir",
                    str(tmp_path / name),
                ]
            )
            == 0
        )
    output_dir = tmp_path / "comparison"
    assert (
        main(
            [
                "compare",
                str(tmp_path / "target-a/evaluation.json"),
                str(tmp_path / "target-b/evaluation.json"),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    comparison = json.loads((output_dir / "comparison.json").read_text())
    assert comparison["score_differences"]["quality_score"] < 0
    assert comparison["right_summary"]["failures"] == 8
    assert "Compatible" in capsys.readouterr().out


@pytest.mark.parametrize(("flag", "env"), [(False, None), (True, None), (False, "1")])
def test_cli_live_guard_runs_before_provider_factory(flag, env, monkeypatch, capsys):
    factory = AsyncMock()
    monkeypatch.setattr("app.evaluation.cli.create_live_target", factory)
    monkeypatch.delenv("CI", raising=False)
    if env:
        monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", env)
    else:
        monkeypatch.delenv("AI_CORE_EVAL_ALLOW_LIVE", raising=False)
    args = ["run", "--provider", "groq"] + (["--live"] if flag else [])
    assert main(args) == 2
    assert "requires --live" in capsys.readouterr().err
    factory.assert_not_awaited()


def test_cli_both_live_guards_allow_only_mock_target(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    created = []

    async def factory(dataset, provider, model, **kwargs):
        created.append((provider, model))
        # No provider client is constructed; only a synthetic target runs.
        return FixtureTarget.load(dataset)

    monkeypatch.setattr("app.evaluation.cli.create_live_target", factory)
    assert (
        main(
            [
                "run",
                "--provider",
                "groq",
                "--model",
                "selected-model",
                "--live",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert created == [("groq", "selected-model")]


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--model", "selected-model"],
        ["run", "--image-mapping", "missing.json"],
        ["run", "--live"],
        ["run", "--repetitions", "0"],
        ["validate", "--dataset", "missing-dataset"],
    ],
)
def test_cli_input_errors_are_clear_and_controlled(arguments, monkeypatch, capsys):
    monkeypatch.setenv("AI_CORE_EVAL_ALLOW_LIVE", "1")
    monkeypatch.delenv("CI", raising=False)
    assert main(arguments) == 2
    assert capsys.readouterr().err


def test_parser_errors_do_not_echo_secret_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        main(["run", "--provider", "TEST_SECRET_DO_NOT_LOG"])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "Invalid evaluation CLI arguments" in output
    assert "TEST_SECRET_DO_NOT_LOG" not in output


def test_cli_cancellation_has_safe_exit_code(monkeypatch, capsys):
    def cancelled(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "run", cancelled)
    assert main(["run"]) == 130
    assert "cancelled" in capsys.readouterr().err
