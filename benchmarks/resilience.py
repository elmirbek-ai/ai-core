from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import logging
import time
from typing import Any

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)

from benchmarks.report import render_table, report_path, write_json_report
from benchmarks.scenarios import run_mock_scenarios


REAL_PROVIDER_NAMES = (
    "groq",
    "openrouter",
    "gemini",
    "cloudflare",
    "ollama",
    "kilo",
    "llm7",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated AI Core resilience benchmarks.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Explicitly enable conservative sequential provider smoke calls.",
    )
    return parser


async def run(real: bool = False) -> dict[str, Any]:
    mock_report = await run_mock_scenarios()
    report: dict[str, Any] = {
        **mock_report,
        "generated_at": datetime.now(UTC).isoformat(),
        "real_mode_requested": real,
    }
    if real:
        if not mock_report["passed"]:
            report["real"] = {
                "ran": False,
                "reason": "mock benchmark failed",
                "providers": [],
            }
        else:
            report["real"] = await run_real_smoke()
            report["mode"] = "mock+real"
    return report


async def run_real_smoke() -> dict[str, Any]:
    """Run at most one sequential request per enabled provider."""
    from app.core.config import get_settings
    from app.llm.registry import create_provider_registry

    settings = get_settings()
    registry = await create_provider_registry(settings)
    results: list[dict[str, Any]] = []
    messages = [
        {"role": "user", "content": "Reply with exactly: AI Core benchmark OK"}
    ]
    model_overrides = {
        "groq": settings.groq_fast_model,
        "kilo": settings.kilo_general_model,
        "llm7": settings.llm7_general_model,
    }
    try:
        for name in REAL_PROVIDER_NAMES:
            provider = registry.get_enabled(name)
            if provider is None:
                results.append({"provider": name, "enabled": False, "attempted": False})
                continue
            started = time.monotonic()
            try:
                response = await provider.chat(
                    messages,
                    model=model_overrides.get(name),
                )
            except Exception as error:
                results.append(
                    {
                        "provider": name,
                        "enabled": True,
                        "attempted": True,
                        "success": False,
                        "latency_seconds": time.monotonic() - started,
                        "error_category": _error_category(error),
                    }
                )
                continue
            results.append(
                {
                    "provider": name,
                    "enabled": True,
                    "attempted": True,
                    "success": True,
                    "latency_seconds": time.monotonic() - started,
                    "model": response.get("model", "unknown"),
                }
            )
    finally:
        await registry.close()
    attempted = [item for item in results if item.get("attempted")]
    return {
        "ran": True,
        "sequential": True,
        "max_requests_per_provider": 1,
        "providers": results,
        "attempted": len(attempted),
        "successes": sum(item.get("success") is True for item in attempted),
        "failures": sum(item.get("success") is False for item in attempted),
    }


def _error_category(error: Exception) -> str:
    if isinstance(error, LLMAuthenticationError):
        return "authentication"
    if isinstance(error, LLMTimeoutError):
        return "timeout"
    if isinstance(error, LLMRateLimitError):
        return "rate_limit"
    if isinstance(error, LLMUpstreamError):
        return "upstream"
    if isinstance(error, LLMProviderError):
        return "provider"
    return "unknown"


def _print_report(report: dict[str, Any]) -> None:
    print(render_table(report))
    summary = report["summary"]
    print()
    print(f"Total: {summary['total_requests']}")
    print(f"Success: {summary['successes']}")
    print(f"Failures: {summary['failures']}")
    print(f"Avg latency: {summary['average_latency_seconds']:.6f}s")
    print(f"Max concurrency: {summary['max_observed_concurrency']}")
    print(f"Circuit skips: {summary['circuit_skips']}")
    real = report.get("real")
    if real and real.get("ran"):
        print()
        print(
            "Real smoke: "
            f"{real['successes']} success, {real['failures']} failure, "
            f"{real['attempted']} attempted"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.getLogger("app.llm.router").setLevel(logging.ERROR)
    logging.getLogger("app.llm.health").setLevel(logging.ERROR)
    report = asyncio.run(run(real=args.real))
    path = write_json_report(report, report_path())
    _print_report(report)
    print(f"Report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
