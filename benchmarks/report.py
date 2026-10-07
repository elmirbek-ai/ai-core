from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SENSITIVE_KEY_PARTS = (
    "api_key",
    "token",
    "authorization",
    "account_id",
    "prompt",
    "messages",
    "content",
    "response_body",
    "raw_exception",
)


def report_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parent
    return base / "results" / "resilience_report.json"


def sanitize_report(value: Any) -> Any:
    """Return a JSON-safe copy without sensitive fields."""
    if isinstance(value, dict):
        return {
            str(key): sanitize_report(item)
            for key, item in value.items()
            if not _is_sensitive_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_report(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def write_json_report(report: dict[str, Any], path: Path) -> Path:
    sanitized = sanitize_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def render_table(report: dict[str, Any]) -> str:
    rows = report.get("scenarios", [])
    headers = ("Scenario", "Result", "Latency", "Provider", "Depth")
    rendered = [headers]
    for row in rows:
        rendered.append(
            (
                str(row.get("name", "-")),
                "PASS" if row.get("passed") else "FAIL",
                f"{float(row.get('latency_seconds', 0.0)):.4f}s",
                str(row.get("selected_provider") or "-"),
                str(row.get("fallback_depth", 0)),
            )
        )
    widths = [max(len(row[index]) for row in rendered) for index in range(5)]
    lines = [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rendered
    ]
    return "\n".join(lines)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)

