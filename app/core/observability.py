import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import (
    RequestContext,
    request_context,
    resolve_request_id,
)
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)

EVENTS = frozenset(
    {
        "request_started",
        "request_completed",
        "request_failed",
        "provider_attempt_started",
        "provider_attempt_completed",
        "provider_attempt_failed",
        "provider_circuit_skipped",
        "stream_started",
        "stream_first_token",
        "stream_completed",
        "stream_failed",
        "stream_cancelled",
        "application_event",
        "error_handled",
    }
)
SAFE_FIELDS = frozenset(
    {
        "method",
        "route",
        "status_code",
        "duration_ms",
        "provider",
        "model",
        "task",
        "error_category",
        "fallback_depth",
        "streaming",
        "exception_type",
        "ttft_ms",
    }
)
# Only these fixed categories can become HTTP metric dimensions or log routes.
HTTP_ROUTES = frozenset(
    {
        "/v1/chat",
        "/v1/chat/stream",
        "/health",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
    }
)


def error_category(error: BaseException) -> str:
    if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        return "cancelled"
    if isinstance(error, LLMAuthenticationError):
        return "authentication"
    if isinstance(error, LLMRateLimitError):
        return "rate_limit"
    if isinstance(error, LLMTimeoutError):
        return "timeout"
    if isinstance(error, LLMUpstreamError):
        return "upstream"
    return "provider"


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any
) -> None:
    if event not in EVENTS:
        raise ValueError("Unknown observability event")
    context = request_context.get()
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "request_id": context.request_id if context else None,
            "safe_fields": {
                key: value for key, value in fields.items() if key in SAFE_FIELDS
            },
        },
    )


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Never format messages, arguments, traceback, exception or stack text.
        context = request_context.get()
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "application_event"),
            "request_id": getattr(
                record,
                "request_id",
                context.request_id if context else None,
            ),
            "component": record.name,
        }
        payload.update(
            {
                key: value
                for key, value in getattr(record, "safe_fields", {}).items()
                if key in SAFE_FIELDS
            }
        )
        return json.dumps(
            payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        )


class _JSONHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        # Resolve stderr at emission time, including under embedding/test capture.
        self.stream = sys.stderr
        super().emit(record)


def configure_logging() -> None:
    """Explicit idempotent setup; root and Uvicorn startup logs stay intact."""
    logger = logging.getLogger("app")
    if not any(isinstance(handler, _JSONHandler) for handler in logger.handlers):
        handler = _JSONHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Uvicorn's default access line contains the raw query string. HTTP events
    # replace it without changing Uvicorn startup/shutdown/error configuration.
    logging.getLogger("uvicorn.access").disabled = True


@dataclass(slots=True)
class HTTPMetrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0


class HTTPObservability:
    """Event-loop-local bounded aggregates; updates have no await/interleaving."""

    def __init__(self) -> None:
        self._total = HTTPMetrics()
        self._routes: dict[str, HTTPMetrics] = {}

    def record(
        self, route: str, status_code: int, duration: float, *, failed: bool
    ) -> None:
        route = route if route in HTTP_ROUTES else "other"
        for metrics in (self._total, self._routes.setdefault(route, HTTPMetrics())):
            metrics.requests += 1
            if failed or status_code >= 400:
                metrics.failures += 1
            else:
                metrics.successes += 1
            metrics.total_latency_seconds += max(0.0, duration)
            metrics.last_latency_seconds = max(0.0, duration)

    @staticmethod
    def _snapshot(metrics: HTTPMetrics) -> dict[str, int | float]:
        return {
            **asdict(metrics),
            "average_latency_seconds": metrics.total_latency_seconds / metrics.requests
            if metrics.requests
            else 0.0,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "total": self._snapshot(self._total),
            "routes": {
                route: self._snapshot(metrics)
                for route, metrics in self._routes.items()
            },
        }


class ObservabilityMiddleware:
    def __init__(
        self, app: ASGIApp, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.app = app
        self.clock = clock
        self.logger = logging.getLogger(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        values = [
            value
            for key, value in scope.get("headers", [])
            if key.lower() == b"x-request-id"
        ]
        request_id = resolve_request_id(values[0] if len(values) == 1 else None)
        path = scope.get("path", "")
        route = path if path in HTTP_ROUTES else "other"
        method = (
            scope["method"]
            if scope["method"]
            in {
                "GET",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
                "HEAD",
                "OPTIONS",
                "TRACE",
                "CONNECT",
            }
            else "OTHER"
        )
        token = request_context.set(RequestContext(request_id, method, route))
        started = self.clock()
        status_code = 500
        response_started = False
        failure: str | None = None
        log_event(self.logger, "request_started", method=method, route=route)

        async def observed_send(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                message = {
                    **message,
                    "headers": [
                        *headers,
                        (b"x-request-id", request_id.encode("ascii")),
                    ],
                }
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except (asyncio.CancelledError, GeneratorExit):
            failure = "cancelled"
            raise
        except Exception:
            failure = "internal"
            # Generate a generic 500 inside the correlation boundary. Outer
            # ServerErrorMiddleware otherwise sends its response after reset.
            if not response_started:
                await observed_send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (b"content-type", b"text/plain; charset=utf-8"),
                            (b"content-length", b"21"),
                        ],
                    }
                )
                await send(
                    {"type": "http.response.body", "body": b"Internal Server Error"}
                )
            raise RuntimeError("AI Core request failed") from None
        finally:
            try:
                duration = max(0.0, self.clock() - started)
                category = failure or {
                    401: "authentication",
                    403: "authentication",
                    422: "validation",
                    429: "rate_limited",
                }.get(status_code)
                if category is None and status_code >= 400:
                    category = "http_error"
                manager = getattr(scope.get("app", None), "state", None)
                metrics = getattr(manager, "http_observability", None)
                if metrics is not None:
                    metrics.record(
                        route, status_code, duration, failed=failure is not None
                    )
                log_event(
                    self.logger,
                    "request_failed" if category else "request_completed",
                    method=method,
                    route=route,
                    status_code=status_code,
                    duration_ms=duration * 1000,
                    error_category=category,
                )
            finally:
                request_context.reset(token)
