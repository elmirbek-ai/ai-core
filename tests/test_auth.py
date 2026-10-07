from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.core.auth as auth_module
from app.api.chat import get_llm_service
from app.core.auth import (
    AIClientAuthConfigurationError,
    validate_auth_configuration,
)
from app.core.config import Settings
from app.llm.streaming import StreamEvent
from app.llm.task import TaskType
from app.main import app


main_module = importlib.import_module("app.main")


_CLIENT_KEY = "unit-test-client-credential"
_WRONG_KEY = "unit-test-wrong-credential"
_CHAT_PAYLOAD = {
    "messages": [{"role": "user", "content": "Hello"}],
}


@dataclass
class SpyService:
    chat_calls: int = 0
    stream_calls: int = 0

    async def chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        self.chat_calls += 1
        return {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "content": "test response",
        }

    async def stream_chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        yield StreamEvent(
            "meta",
            {"provider": "groq", "model": "openai/gpt-oss-20b"},
        )
        yield StreamEvent("delta", {"content": "test response"})
        yield StreamEvent("done", {})


@pytest.fixture(scope="module")
def protected_client() -> Iterator[tuple[TestClient, SpyService]]:
    service = SpyService()
    app.dependency_overrides[get_llm_service] = lambda: service
    try:
        with TestClient(app) as client:
            app.state.ai_core_auth_enabled = True
            app.state.ai_core_api_key = SecretStr(_CLIENT_KEY)
            yield client, service
    finally:
        app.dependency_overrides.clear()


def authorization(key: str = _CLIENT_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_auth_disabled_allows_request(
    protected_client: tuple[TestClient, SpyService],
) -> None:
    client, service = protected_client
    before = service.chat_calls
    app.state.ai_core_auth_enabled = False
    try:
        response = client.post("/v1/chat", json=_CHAT_PAYLOAD)
    finally:
        app.state.ai_core_auth_enabled = True

    assert response.status_code == 200
    assert service.chat_calls == before + 1


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": "Basic abc123"},
        authorization(_WRONG_KEY),
    ],
)
def test_non_stream_rejects_invalid_authorization_before_service(
    protected_client: tuple[TestClient, SpyService],
    headers: dict[str, str],
) -> None:
    client, service = protected_client
    before = service.chat_calls

    response = client.post("/v1/chat", json=_CHAT_PAYLOAD, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert service.chat_calls == before


def test_auth_runs_before_request_body_validation(
    protected_client: tuple[TestClient, SpyService],
) -> None:
    client, service = protected_client
    before = service.chat_calls

    response = client.post("/v1/chat", json={"messages": []})

    assert response.status_code == 401
    assert service.chat_calls == before


def test_correct_bearer_key_preserves_chat_contract(
    protected_client: tuple[TestClient, SpyService],
) -> None:
    client, service = protected_client
    before = service.chat_calls

    response = client.post(
        "/v1/chat",
        json=_CHAT_PAYLOAD,
        headers=authorization(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "content": "test response",
    }
    assert service.chat_calls == before + 1


@pytest.mark.parametrize("headers", [{}, authorization(_WRONG_KEY)])
def test_stream_rejects_auth_before_sse_and_provider(
    protected_client: tuple[TestClient, SpyService],
    headers: dict[str, str],
) -> None:
    client, service = protected_client
    before = service.stream_calls

    response = client.post(
        "/v1/chat/stream",
        json=_CHAT_PAYLOAD,
        headers=headers,
    )

    assert response.status_code == 401
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" not in response.text
    assert service.stream_calls == before


def test_correct_bearer_key_preserves_sse_contract(
    protected_client: tuple[TestClient, SpyService],
) -> None:
    client, service = protected_client
    before = service.stream_calls

    response = client.post(
        "/v1/chat/stream",
        json=_CHAT_PAYLOAD,
        headers=authorization(),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: meta" in response.text
    assert "event: delta" in response.text
    assert "event: done" in response.text
    assert service.stream_calls == before + 1


def test_health_and_docs_remain_public(
    protected_client: tuple[TestClient, SpyService],
) -> None:
    client, _ = protected_client

    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_openapi_has_bearer_scheme_without_key(
    protected_client: tuple[TestClient, SpyService],
) -> None:
    client, _ = protected_client

    schema = client.get("/openapi.json").json()
    bearer = schema["components"]["securitySchemes"]["AI Core Bearer"]

    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"
    assert schema["paths"]["/v1/chat"]["post"]["security"]
    assert schema["paths"]["/v1/chat/stream"]["post"]["security"]
    assert _CLIENT_KEY not in str(schema)


def test_credentials_are_absent_from_response_and_logs(
    protected_client: tuple[TestClient, SpyService],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _ = protected_client

    response = client.post(
        "/v1/chat",
        json=_CHAT_PAYLOAD,
        headers=authorization(_WRONG_KEY),
    )

    combined = response.text + caplog.text
    assert _CLIENT_KEY not in combined
    assert _WRONG_KEY not in combined
    assert "Authorization" not in combined


def test_key_comparison_uses_compare_digest(
    protected_client: tuple[TestClient, SpyService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = protected_client
    calls: list[tuple[str, str]] = []

    def compare_digest(left: str, right: str) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(auth_module.secrets, "compare_digest", compare_digest)

    response = client.post(
        "/v1/chat",
        json=_CHAT_PAYLOAD,
        headers=authorization(),
    )

    assert response.status_code == 200
    assert calls == [(_CLIENT_KEY, _CLIENT_KEY)]


def test_blank_api_key_normalizes_to_none() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-provider-key",
        ai_core_auth_enabled=False,
        ai_core_api_key="   ",
    )

    assert settings.ai_core_api_key is None


def test_auth_disabled_allows_missing_configured_key() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-provider-key",
        ai_core_auth_enabled=False,
        ai_core_api_key=None,
    )

    validate_auth_configuration(settings)


def test_auth_enabled_without_key_fails_startup_validation() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-provider-key",
        ai_core_auth_enabled=True,
        ai_core_api_key=None,
    )

    with pytest.raises(
        AIClientAuthConfigurationError,
        match="authentication is enabled but no API key is configured",
    ):
        validate_auth_configuration(settings)


def test_application_lifespan_fails_closed_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-provider-key",
        ai_core_auth_enabled=True,
        ai_core_api_key=None,
    )
    monkeypatch.setattr(main_module, "settings", settings)
    startup_app = FastAPI(lifespan=main_module.lifespan)

    with pytest.raises(AIClientAuthConfigurationError):
        with TestClient(startup_app):
            pass
