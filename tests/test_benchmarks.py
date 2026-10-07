import asyncio
import json
from pathlib import Path

from benchmarks.report import report_path, sanitize_report, write_json_report
from benchmarks.resilience import build_parser, run
from benchmarks.scenarios import run_mock_scenarios


def test_mock_benchmark_is_deterministic_and_passes() -> None:
    report = asyncio.run(run_mock_scenarios())

    assert report["passed"] is True
    assert report["summary"]["total_requests"] == 50
    assert report["summary"]["successes"] == 46
    assert report["summary"]["failures"] == 4
    assert report["summary"]["max_fallback_depth"] == 2


def test_telemetry_reconciliation_passes() -> None:
    report = asyncio.run(run_mock_scenarios())

    reconciliation = report["telemetry_reconciliation"]
    assert reconciliation["passed"] is True
    assert reconciliation["actual"] == reconciliation["expected"]


def test_all_final_concurrency_snapshots_have_zero_in_flight() -> None:
    report = asyncio.run(run_mock_scenarios())

    snapshots = report["concurrency"].values()
    assert all(
        state["in_flight"] == 0
        for snapshot in snapshots
        for state in snapshot.values()
    )


def test_json_report_serialization() -> None:
    path = Path("benchmarks/results/_test_report.json")
    report = {"mode": "mock", "summary": {"successes": 1}}

    try:
        written = write_json_report(report, path)

        assert written == path
        assert json.loads(path.read_text(encoding="utf-8")) == report
    finally:
        path.unlink(missing_ok=True)


def test_report_sanitization_removes_sensitive_fields() -> None:
    secret = "benchmark-secret-value"
    sanitized = sanitize_report(
        {
            "provider": "groq",
            "api_key": secret,
            "nested": {
                "authorization": f"Bearer {secret}",
                "prompt": "private prompt",
                "raw_exception": "private SDK detail",
                "safe": "timeout",
            },
        }
    )

    encoded = json.dumps(sanitized)
    assert secret not in encoded
    assert "private prompt" not in encoded
    assert "private SDK detail" not in encoded
    assert sanitized == {"provider": "groq", "nested": {"safe": "timeout"}}


def test_external_mode_is_disabled_by_default() -> None:
    args = build_parser().parse_args([])
    report = asyncio.run(run(real=args.real))

    assert args.real is False
    assert report["mode"] == "mock"
    assert report["real_mode_requested"] is False
    assert "real" not in report


def test_real_mode_requires_explicit_flag() -> None:
    args = build_parser().parse_args(["--real"])

    assert args.real is True


def test_report_path_generation() -> None:
    root = Path("benchmark-root")
    assert report_path(root) == (
        root / "results" / "resilience_report.json"
    )
