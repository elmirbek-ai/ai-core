import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.core.observability import HTTPObservability, ObservabilityMiddleware
from app.core.request_context import RequestContext, request_context, resolve_request_id


@pytest.mark.parametrize(
    "value",
    [
        None,
        b"",
        b"bad\nID",
        b"bad\rID",
        b"bad\x00ID",
        b"bad\x1fID",
        b"bad\x7fID",
        b"a" * 129,
        "bad\u2028ID".encode(),
        "bad\u200bID".encode(),
        b"bad ID",
        b"bad\xffID",
    ],
)
def test_invalid_request_ids_are_replaced(value):
    result = resolve_request_id(value)
    assert UUID(result).version == 4
    assert result.encode() != value


@pytest.mark.parametrize("value", [b"safe-ID_123.abc:xyz", b"x", b"a" * 128])
def test_safe_request_id_is_preserved(value):
    assert resolve_request_id(value) == value.decode()


def test_generated_request_ids_do_not_collide():
    assert len({resolve_request_id(None) for _ in range(100)}) == 100


def test_request_context_concurrent_isolation_and_reset(caplog):
    async def exercise():
        entered = [asyncio.Event() for _ in range(3)]
        release = asyncio.Event()
        observed = []
        responses = [[] for _ in entered]
        manager = HTTPObservability()

        async def inner(scope, receive, send):
            index = scope["index"]
            before = request_context.get()
            entered[index].set()
            await release.wait()
            assert request_context.get() is before
            observed.append(before.request_id)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = ObservabilityMiddleware(inner)

        async def run(index):
            async def send(message):
                responses[index].append(message)

            headers = [(b"x-request-id", b"supplied-safe")] if index == 0 else []
            await middleware(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/health",
                    "headers": headers,
                    "index": index,
                    "app": SimpleNamespace(
                        state=SimpleNamespace(http_observability=manager)
                    ),
                },
                None,
                send,
            )
            assert request_context.get() is None

        tasks = [asyncio.create_task(run(index)) for index in range(3)]
        await asyncio.gather(*(event.wait() for event in entered))
        assert request_context.get() is None
        release.set()
        await asyncio.gather(*tasks)
        ids = [
            dict(response[0]["headers"])[b"x-request-id"].decode()
            for response in responses
        ]
        assert ids[0] == "supplied-safe"
        assert len(set(ids)) == 3
        assert set(observed) == set(ids)
        assert manager.snapshot()["total"]["requests"] == 3
        return ids

    ids = asyncio.run(exercise())
    completed = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "request_completed"
    ]
    assert {record.request_id for record in completed} == set(ids)


def test_nested_context_is_restored():
    async def exercise():
        outer = RequestContext("outer", "GET", "/health")
        token = request_context.set(outer)

        async def inner(scope, receive, send):
            assert request_context.get().request_id != "outer"

        try:
            await ObservabilityMiddleware(inner)(
                {"type": "http", "method": "GET", "path": "/health"}, None, None
            )
            assert request_context.get() is outer
        finally:
            request_context.reset(token)

    asyncio.run(exercise())
