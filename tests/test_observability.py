import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.observability import HTTPObservability, ObservabilityMiddleware
from app.core.rate_limit import APIRateLimiter
from app.core.request_context import request_context
from app.main import app


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/health", 200),
        ("/docs", 200),
        ("/redoc", 200),
        ("/openapi.json", 200),
        ("/missing", 404),
        ("/metrics", 404),
    ],
)
def test_response_request_id_on_public_and_unknown_routes(path, status):
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == status
    assert UUID(response.headers["X-Request-ID"]).version == 4


def test_safe_header_on_validation_failure_and_metrics_lifecycle(caplog):
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat", json={"messages": []}, headers={"X-Request-ID": "validation-id"}
        )
        assert response.status_code == 422
        assert response.headers["X-Request-ID"] == "validation-id"
        first = app.state.http_observability
        assert first.snapshot()["total"]["failures"] == 1
    with TestClient(app):
        assert app.state.http_observability is not first
        assert app.state.http_observability.snapshot()["total"]["requests"] == 0
    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "request_failed"
    ]
    assert failures[-1].safe_fields["error_category"] == "validation"


def test_request_metrics_detached_bounded_and_consistent():
    manager = HTTPObservability()
    assert manager.snapshot()["total"]["average_latency_seconds"] == 0
    manager.record("/v1/chat", 200, 0.2, failed=False)
    manager.record("/v1/chat", 429, 0.4, failed=False)
    manager.record("/sensitive/arbitrary", 200, -1, failed=True)
    snapshot = manager.snapshot()
    assert snapshot["total"] == {
        "requests": 3,
        "successes": 1,
        "failures": 2,
        "total_latency_seconds": pytest.approx(0.6),
        "last_latency_seconds": 0.0,
        "average_latency_seconds": pytest.approx(0.2),
    }
    assert set(snapshot["routes"]) == {"/v1/chat", "other"}
    snapshot["routes"]["/v1/chat"]["requests"] = 999
    snapshot["total"]["requests"] = 999
    assert manager.snapshot()["routes"]["/v1/chat"]["requests"] == 2
    assert manager.snapshot()["total"]["requests"] == 3


@pytest.mark.parametrize("status", [200, 401, 403, 429, 500])
def test_monotonic_latency_and_status_categories(status, caplog):
    async def exercise():
        manager = HTTPObservability()
        times = iter([100.0, 100.125])
        sent = []

        async def inner(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [(b"x-request-id", b"wrong")],
                }
            )
            await send({"type": "http.response.body", "body": b"unchanged"})

        async def send(message):
            sent.append(message)

        await ObservabilityMiddleware(inner, clock=lambda: next(times))(
            {
                "type": "http",
                "method": "BAD\nMETHOD",
                "path": "/arbitrary",
                "headers": [(b"X-Request-ID", b"safe-id")],
                "app": SimpleNamespace(
                    state=SimpleNamespace(http_observability=manager)
                ),
            },
            None,
            send,
        )
        assert dict(sent[0]["headers"])[b"x-request-id"] == b"safe-id"
        assert sent[1]["body"] == b"unchanged"
        assert manager.snapshot()["total"]["last_latency_seconds"] == 0.125

    asyncio.run(exercise())
    record = [
        record
        for record in caplog.records
        if getattr(record, "event", None) in {"request_completed", "request_failed"}
    ][-1]
    assert record.safe_fields["duration_ms"] == 125
    assert record.safe_fields["method"] == "OTHER"
    assert record.safe_fields["route"] == "other"
    assert record.safe_fields["status_code"] == status


@pytest.mark.parametrize("started", [False, True])
def test_unhandled_error_is_sanitized_and_context_reset(started, caplog):
    async def exercise():
        sent = []

        async def inner(scope, receive, send):
            if started:
                await send(
                    {"type": "http.response.start", "status": 500, "headers": []}
                )
            raise RuntimeError("TEST_SECRET_DO_NOT_LOG")

        async def send(message):
            sent.append(message)

        with pytest.raises(RuntimeError, match="AI Core request failed") as captured:
            await ObservabilityMiddleware(inner)(
                {"type": "http", "method": "GET", "path": "/health"}, None, send
            )
        assert captured.value.__suppress_context__
        assert request_context.get() is None
        assert (
            len(
                [
                    message
                    for message in sent
                    if message["type"] == "http.response.start"
                ]
            )
            == 1
        )
        assert b"x-request-id" in dict(sent[0]["headers"])

    asyncio.run(exercise())
    assert "TEST_SECRET_DO_NOT_LOG" not in caplog.text


def test_http_cancellation_resets_context_and_records_failure(caplog):
    async def exercise():
        entered = asyncio.Event()
        manager = HTTPObservability()

        async def inner(scope, receive, send):
            entered.set()
            await asyncio.Event().wait()

        async def run():
            try:
                await ObservabilityMiddleware(inner)(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/chat/stream",
                        "headers": [(b"x-request-id", b"cancel-id")],
                        "app": SimpleNamespace(
                            state=SimpleNamespace(http_observability=manager)
                        ),
                    },
                    None,
                    None,
                )
            finally:
                assert request_context.get() is None

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.snapshot()["total"]["failures"] == 1

    asyncio.run(exercise())
    assert any(
        getattr(record, "safe_fields", {}).get("error_category") == "cancelled"
        and record.request_id == "cancel-id"
        for record in caplog.records
    )


def test_non_http_scope_passes_through_without_context():
    observed = []

    async def inner(scope, receive, send):
        observed.append(scope)
        assert request_context.get() is None

    scope = {"type": "lifespan"}
    asyncio.run(ObservabilityMiddleware(inner)(scope, None, None))
    assert observed == [scope]


def test_duplicate_incoming_request_ids_are_replaced():
    with TestClient(app) as client:
        response = client.get(
            "/health", headers=[("X-Request-ID", "first"), ("X-Request-ID", "second")]
        )
    assert UUID(response.headers["X-Request-ID"]).version == 4


def test_stream_body_is_forwarded_without_buffering():
    async def exercise():
        sent = []

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            for index in range(3):
                message = {
                    "type": "http.response.body",
                    "body": str(index).encode(),
                    "more_body": index < 2,
                }
                await send(message)
                assert sent[-1] is message

        async def send(message):
            sent.append(message)

        await ObservabilityMiddleware(inner)(
            {"type": "http", "method": "POST", "path": "/v1/chat/stream"}, None, send
        )
        assert len(sent) == 4

    asyncio.run(exercise())


def test_protected_rejections_have_request_id_and_auth_precedes_limit(caplog):
    with TestClient(app) as client:
        app.state.ai_core_auth_enabled = True
        limiter = APIRateLimiter(burst_size=1)
        app.state.api_rate_limiter = limiter
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers={"X-Request-ID": "unauthorized-id"},
        )
        assert response.status_code == 401
        assert response.headers["X-Request-ID"] == "unauthorized-id"
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert asyncio.run(limiter.snapshot())["allowed"] == 0
        app.state.ai_core_auth_enabled = False
        asyncio.run(limiter.acquire_request())
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers={"X-Request-ID": "rate-limited-id"},
        )
        assert response.status_code == 429
        assert response.headers["X-Request-ID"] == "rate-limited-id"
        assert "Retry-After" in response.headers
        assert app.state.http_observability.snapshot()["total"]["failures"] == 2
    categories = [
        record.safe_fields["error_category"]
        for record in caplog.records
        if getattr(record, "event", None) == "request_failed"
    ]
    assert categories == ["authentication", "rate_limited"]
