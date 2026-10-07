from collections.abc import Iterator
import importlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_llm_service
from app.core.config import Settings
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.llm.model_router import TaskModelRouter
from app.llm.providers.groq import GroqProvider
from app.llm.router import LLMRouter
from app.llm.task import TaskType
from app.main import app
from app.services.llm_service import LLMService


class FakeLLMService:
    async def chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        assert messages
        return {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "content": "test response",
        }


class FailingLLMService:
    async def chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        raise RuntimeError("internal provider detail")


class DomainFailingLLMService:
    def __init__(self, error: LLMProviderError) -> None:
        self.error = error

    async def chat(
        self,
        messages: list[object],
        task: TaskType = TaskType.GENERAL,
    ) -> dict[str, str]:
        raise self.error


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat(client: TestClient) -> None:
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()

    try:
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "content": "test response",
    }


def test_chat_rejects_empty_messages(client: TestClient) -> None:
    response = client.post("/v1/chat", json={"messages": []})

    assert response.status_code == 422


def test_chat_does_not_leak_provider_exception(client: TestClient) -> None:
    app.dependency_overrides[get_llm_service] = lambda: FailingLLMService()

    try:
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider request failed"}
    assert "internal provider detail" not in response.text


def test_lifespan_closes_provider(monkeypatch) -> None:
    close = AsyncMock()
    monkeypatch.setattr(GroqProvider, "close", close)

    with TestClient(app):
        pass

    close.assert_awaited_once()


def test_lifespan_shares_health_manager_with_router() -> None:
    with TestClient(app):
        manager = app.state.provider_health_manager
        router_manager = app.state.llm_service.router.health_manager

        assert router_manager is manager


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (LLMAuthenticationError("auth failed"), 502),
        (LLMRateLimitError("rate limited"), 503),
        (LLMTimeoutError("timed out"), 504),
        (LLMUpstreamError("upstream failed"), 502),
        (LLMProviderError("provider failed"), 502),
    ],
)
def test_domain_error_http_mapping(
    client: TestClient,
    error: LLMProviderError,
    expected_status: int,
) -> None:
    app.dependency_overrides[get_llm_service] = lambda: (
        DomainFailingLLMService(error)
    )

    try:
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == expected_status
    assert response.json() == {"detail": "LLM provider request failed"}
    assert str(error) not in response.text


def test_secret_is_not_exposed_in_response_or_logs(
    client: TestClient,
    caplog,
) -> None:
    secret = "sdk-secret-detail"
    app.dependency_overrides[get_llm_service] = lambda: (
        DomainFailingLLMService(LLMUpstreamError(secret))
    )

    try:
        with caplog.at_level(logging.ERROR, logger="app.api.chat"):
            response = client.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert secret not in response.text
    assert secret not in caplog.text


def test_lifespan_closes_all_enabled_providers(monkeypatch) -> None:
    main_module = importlib.import_module("app.main")
    registry_module = importlib.import_module("app.llm.registry")
    groq = SimpleNamespace(
        name="groq",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    openrouter = SimpleNamespace(
        name="openrouter",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    gemini = SimpleNamespace(
        name="gemini",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    cloudflare = SimpleNamespace(
        name="cloudflare",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    ollama = SimpleNamespace(
        name="ollama",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    kilo = SimpleNamespace(
        name="kilo",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    llm7 = SimpleNamespace(
        name="llm7",
        chat=AsyncMock(),
        close=AsyncMock(),
    )
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openrouter_api_key="openrouter-test-key",
        gemini_api_key="gemini-test-key",
        cloudflare_api_token="cloudflare-test-token",
        cloudflare_account_id="test-account-id",
        ollama_api_key="ollama-test-key",
        kilo_enabled=True,
        llm7_enabled=True,
        llm7_api_key="llm7-test-key",
    )
    monkeypatch.setattr(main_module, "settings", settings)
    monkeypatch.setattr(
        registry_module,
        "GroqProvider",
        lambda settings: groq,
    )
    monkeypatch.setattr(
        registry_module,
        "OpenRouterProvider",
        lambda settings: openrouter,
    )
    monkeypatch.setattr(
        registry_module,
        "GeminiProvider",
        lambda settings: gemini,
    )
    monkeypatch.setattr(
        registry_module,
        "CloudflareProvider",
        lambda settings: cloudflare,
    )
    monkeypatch.setattr(
        registry_module,
        "OllamaProvider",
        lambda settings: ollama,
    )
    monkeypatch.setattr(
        registry_module,
        "KiloProvider",
        lambda settings: kilo,
    )
    monkeypatch.setattr(
        registry_module,
        "LLM7Provider",
        lambda settings: llm7,
    )

    with TestClient(app):
        pass

    groq.close.assert_awaited_once()
    openrouter.close.assert_awaited_once()
    gemini.close.assert_awaited_once()
    cloudflare.close.assert_awaited_once()
    ollama.close.assert_awaited_once()
    kilo.close.assert_awaited_once()
    llm7.close.assert_awaited_once()


def test_fallback_response_preserves_openrouter_provider(
    client: TestClient,
) -> None:
    primary = SimpleNamespace(
        name="groq",
        chat=AsyncMock(side_effect=LLMRateLimitError("rate limited")),
    )
    fallback = SimpleNamespace(
        name="openrouter",
        chat=AsyncMock(
            return_value={
                "provider": "openrouter",
                "model": "openrouter/free",
                "content": "fallback response",
            }
        ),
    )
    service = LLMService(
        router=LLMRouter(
            primary_provider=primary,
            fallback_provider=fallback,
        ),
        model_router=TaskModelRouter(
            fast_model="openai/gpt-oss-20b",
            reasoning_model="openai/gpt-oss-120b",
        ),
    )
    app.dependency_overrides[get_llm_service] = lambda: service

    try:
        response = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "provider": "openrouter",
        "model": "openrouter/free",
        "content": "fallback response",
    }


def test_unknown_task_returns_422(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={
            "task": "unknown",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("task", "expected_model"),
    [
        (None, "openai/gpt-oss-20b"),
        ("general", "openai/gpt-oss-20b"),
        ("reasoning", "openai/gpt-oss-120b"),
        ("code", "openai/gpt-oss-120b"),
    ],
)
def test_api_response_reports_selected_model(
    client: TestClient,
    task: str | None,
    expected_model: str,
) -> None:
    async def primary_chat(messages, model=None):
        return {
            "provider": "groq",
            "model": model,
            "content": "response",
        }

    primary = SimpleNamespace(
        name="groq",
        chat=AsyncMock(side_effect=primary_chat),
    )
    service = LLMService(
        router=LLMRouter(primary_provider=primary),
        model_router=TaskModelRouter(
            fast_model="openai/gpt-oss-20b",
            reasoning_model="openai/gpt-oss-120b",
        ),
    )
    app.dependency_overrides[get_llm_service] = lambda: service
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if task is not None:
        payload["task"] = task

    try:
        response = client.post(
            "/v1/chat",
            json=payload,
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "provider": "groq",
        "model": expected_model,
        "content": "response",
    }


@pytest.mark.parametrize("task", ["multimodal", "long_context"])
def test_api_routes_gemini_tasks_to_gemini(
    client: TestClient,
    task: str,
) -> None:
    groq = SimpleNamespace(
        name="groq",
        chat=AsyncMock(),
    )
    gemini = SimpleNamespace(
        name="gemini",
        chat=AsyncMock(
            return_value={
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "content": "Gemini response",
            }
        ),
    )
    service = LLMService(
        router=LLMRouter(
            primary_provider=groq,
            gemini_provider=gemini,
        ),
        model_router=TaskModelRouter(
            fast_model="openai/gpt-oss-20b",
            reasoning_model="openai/gpt-oss-120b",
        ),
    )
    app.dependency_overrides[get_llm_service] = lambda: service

    try:
        response = client.post(
            "/v1/chat",
            json={
                "task": task,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "content": "Gemini response",
    }
    groq.chat.assert_not_awaited()


def test_reasoning_fallback_response_can_come_from_cloudflare(
    client: TestClient,
) -> None:
    groq = SimpleNamespace(
        name="groq",
        chat=AsyncMock(side_effect=LLMTimeoutError("timed out")),
    )
    openrouter = SimpleNamespace(
        name="openrouter",
        chat=AsyncMock(),
    )
    cloudflare = SimpleNamespace(
        name="cloudflare",
        chat=AsyncMock(
            return_value={
                "provider": "cloudflare",
                "model": "cf/openai/gpt-oss-120b",
                "content": "Cloudflare response",
            }
        ),
    )
    service = LLMService(
        router=LLMRouter(
            primary_provider=groq,
            fallback_provider=openrouter,
            cloudflare_provider=cloudflare,
        ),
        model_router=TaskModelRouter(
            fast_model="openai/gpt-oss-20b",
            reasoning_model="openai/gpt-oss-120b",
        ),
    )
    app.dependency_overrides[get_llm_service] = lambda: service

    try:
        response = client.post(
            "/v1/chat",
            json={
                "task": "reasoning",
                "messages": [{"role": "user", "content": "Analyze"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "provider": "cloudflare",
        "model": "cf/openai/gpt-oss-120b",
        "content": "Cloudflare response",
    }
    openrouter.chat.assert_not_awaited()


def test_cloudflare_credentials_are_not_exposed_in_response_or_logs(
    client: TestClient,
    caplog,
) -> None:
    token = "private-cloudflare-token"
    account_id = "private-cloudflare-account"
    error = LLMProviderError(f"{token} {account_id}")
    app.dependency_overrides[get_llm_service] = lambda: (
        DomainFailingLLMService(error)
    )

    try:
        with caplog.at_level(logging.ERROR, logger="app.api.chat"):
            response = client.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert token not in response.text
    assert account_id not in response.text
    assert token not in caplog.text
    assert account_id not in caplog.text
