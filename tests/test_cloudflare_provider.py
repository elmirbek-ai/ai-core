import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import Settings
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import LLMProviderError
from app.llm.providers.cloudflare import CloudflareProvider


def cloudflare_settings(**overrides) -> Settings:
    values = {
        "groq_api_key": "test-key",
        "cloudflare_api_token": "cloudflare-test-token",
        "cloudflare_account_id": "test-account-id",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_cloudflare_provider_contract_and_close() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="cf/openai/gpt-oss-120b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Cloudflare response"),
                )
            ],
        )
    )
    close = AsyncMock()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=close,
    )
    provider = CloudflareProvider(
        settings=cloudflare_settings(),
        client=client,
    )

    async def exercise_provider() -> dict[str, str]:
        result = await provider.chat(
            [{"role": "user", "content": "Hello"}],
        )
        await provider.close()
        return result

    result = asyncio.run(exercise_provider())

    assert isinstance(provider, BaseLLMProvider)
    assert result == {
        "provider": "cloudflare",
        "model": "cf/openai/gpt-oss-120b",
        "content": "Cloudflare response",
    }
    create.assert_awaited_once_with(
        model="cf/openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello"}],
    )
    close.assert_awaited_once()


def test_cloudflare_client_configuration_and_base_url() -> None:
    settings = cloudflare_settings(
        cloudflare_timeout_seconds=18.0,
        cloudflare_max_retries=4,
    )

    with patch("app.llm.providers.cloudflare.AsyncOpenAI") as client_type:
        CloudflareProvider(settings=settings)

    client_type.assert_called_once_with(
        api_key="cloudflare-test-token",
        base_url=(
            "https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/v1"
        ),
        timeout=18.0,
        max_retries=4,
    )


def test_cloudflare_provider_rejects_missing_token() -> None:
    settings = cloudflare_settings(cloudflare_api_token=None)

    with pytest.raises(ValueError, match="provider is disabled"):
        CloudflareProvider(settings=settings)


def test_cloudflare_provider_rejects_missing_account_id() -> None:
    settings = cloudflare_settings(cloudflare_account_id=None)

    with pytest.raises(ValueError, match="provider is disabled"):
        CloudflareProvider(settings=settings)


def test_cloudflare_unknown_error_hides_credentials(caplog) -> None:
    token = "private-cloudflare-token"
    account_id = "private-account-id"
    create = AsyncMock(
        side_effect=RuntimeError(f"{token} {account_id}"),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
        close=AsyncMock(),
    )
    provider = CloudflareProvider(
        settings=cloudflare_settings(
            cloudflare_api_token=token,
            cloudflare_account_id=account_id,
        ),
        client=client,
    )

    with pytest.raises(LLMProviderError) as captured:
        asyncio.run(
            provider.chat([{"role": "user", "content": "Hello"}]),
        )

    assert token not in str(captured.value)
    assert account_id not in str(captured.value)
    assert token not in caplog.text
    assert account_id not in caplog.text
    assert captured.value.__cause__ is None


def test_cloudflare_client_initialization_hides_credentials() -> None:
    token = "private-cloudflare-token"
    account_id = "private-account-id"
    settings = cloudflare_settings(
        cloudflare_api_token=token,
        cloudflare_account_id=account_id,
    )

    with (
        patch(
            "app.llm.providers.cloudflare.AsyncOpenAI",
            side_effect=RuntimeError(f"{token} {account_id}"),
        ),
        pytest.raises(LLMProviderError) as captured,
    ):
        CloudflareProvider(settings=settings)

    assert token not in str(captured.value)
    assert account_id not in str(captured.value)
    assert captured.value.__cause__ is None
