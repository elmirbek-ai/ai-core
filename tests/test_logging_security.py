import asyncio
import json
import logging
import sys
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_llm_service
from app.core.observability import (
    EVENTS,
    JSONFormatter,
    configure_logging,
    error_category,
    log_event,
)
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.streaming import StreamEvent
from app.main import app

SECRET = "TEST_SECRET_DO_NOT_LOG"


def test_json_schema_initialization_no_duplicate_lines(capsys):
    root_handlers = list(logging.getLogger().handlers)
    uvicorn_handlers = list(logging.getLogger("uvicorn.error").handlers)
    configure_logging()
    configure_logging()
    log_event(
        logging.getLogger("app.test"),
        "request_completed",
        method="GET",
        route="/health",
        status_code=200,
        duration_ms=1.25,
        body=SECRET,
    )
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert {"timestamp", "level", "event", "request_id", "component"} <= payload.keys()
    assert payload["event"] == "request_completed"
    assert payload["request_id"] is None
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 1.25
    assert SECRET not in lines[0]
    assert logging.getLogger().handlers == root_handlers
    assert logging.getLogger("uvicorn.error").handlers == uvicorn_handlers
    assert logging.getLogger("uvicorn.access").disabled


def test_formatter_discards_raw_message_arguments_exception_and_unknown_fields():
    try:
        raise RuntimeError(SECRET + " https://upstream.invalid/?key=" + SECRET)
    except RuntimeError:
        record = logging.LogRecord(
            "app.test", logging.ERROR, "private", 1, "raw %s", (SECRET,), sys.exc_info()
        )
    record.stack_info = SECRET
    record.safe_fields = {"body": SECRET, "Authorization": SECRET, "status_code": 502}
    result = JSONFormatter().format(record)
    assert SECRET not in result
    assert "upstream.invalid" not in result
    assert json.loads(result)["event"] == "application_event"


@pytest.mark.parametrize("hostile", ["bad\nID", "bad\rID", "bad\x00ID", "x" * 129])
def test_invalid_header_cannot_inject_logs(hostile, capsys):
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": hostile})
    assert UUID(response.headers["X-Request-ID"]).version == 4
    output = capsys.readouterr().err
    assert hostile not in output
    for line in output.splitlines():
        assert json.loads(line)["event"] in EVENTS


@pytest.mark.parametrize("fail", [False, True])
def test_http_logs_exclude_auth_prompt_response_query_and_upstream_error(
    fail, capsys, caplog
):
    class Service:
        async def chat(self, messages, task):
            if fail:
                raise LLMUpstreamError(
                    SECRET
                    + "_ERROR upstream body https://private.invalid/?token="
                    + SECRET
                )
            return {
                "provider": "groq",
                "model": "mock",
                "content": SECRET + "_RESPONSE",
            }

    app.dependency_overrides[get_llm_service] = Service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat?token=" + SECRET + "_QUERY",
                json={"messages": [{"role": "user", "content": SECRET + "_PROMPT"}]},
                headers={
                    "Authorization": "Bearer " + SECRET + "_AUTH",
                    "X-Request-ID": "redaction-id",
                },
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == (502 if fail else 200)
    output = capsys.readouterr().err
    assert SECRET not in output + caplog.text
    assert "private.invalid" not in output
    payloads = [json.loads(line) for line in output.splitlines()]
    assert payloads
    assert all(payload["request_id"] == "redaction-id" for payload in payloads)
    assert (
        sum(
            payload["event"] in {"request_completed", "request_failed"}
            for payload in payloads
        )
        == 1
    )
    if fail:
        handled = [
            payload for payload in payloads if payload["event"] == "error_handled"
        ]
        assert handled[0]["error_category"] == "upstream"


def test_sse_logs_exclude_delta_and_raw_upstream_body(capsys, caplog):
    class Service:
        async def stream_chat(self, messages, task):
            yield StreamEvent("meta", {"provider": "groq", "model": "mock"})
            yield StreamEvent("delta", {"content": SECRET + "_DELTA"})
            raise LLMTimeoutError(SECRET + "_UPSTREAM_BODY")

    app.dependency_overrides[get_llm_service] = Service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/stream?key=" + SECRET,
                json={"messages": [{"role": "user", "content": SECRET}]},
                headers={"Authorization": "Bearer " + SECRET},
            )
    finally:
        app.dependency_overrides.clear()
    assert "event: error" in response.text
    assert SECRET + "_DELTA" in response.text
    assert SECRET + "_UPSTREAM_BODY" not in response.text
    assert SECRET not in capsys.readouterr().err + caplog.text


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (LLMAuthenticationError(SECRET), "authentication"),
        (LLMRateLimitError(SECRET), "rate_limit"),
        (LLMTimeoutError(SECRET), "timeout"),
        (LLMUpstreamError(SECRET), "upstream"),
        (LLMProviderError(SECRET), "provider"),
        (RuntimeError(SECRET), "provider"),
        (asyncio.CancelledError(), "cancelled"),
        (GeneratorExit(), "cancelled"),
    ],
)
def test_safe_domain_error_categories(error, category):
    assert error_category(error) == category


def test_unknown_event_is_rejected():
    with pytest.raises(ValueError, match="Unknown observability event"):
        log_event(logging.getLogger("app.test"), SECRET)
