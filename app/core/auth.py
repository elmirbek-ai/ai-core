from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated, NoReturn

from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from app.core.config import Settings


class AIClientAuthConfigurationError(RuntimeError):
    """Raised when client API authentication cannot be enabled safely."""


@dataclass(frozen=True, slots=True)
class AuthContext:
    client_id: str


bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="AI Core Bearer",
    description="AI Core client API key",
)


def validate_auth_configuration(settings: Settings) -> None:
    if settings.ai_core_auth_enabled and settings.ai_core_api_key is None:
        raise AIClientAuthConfigurationError(
            "AI Core client authentication is enabled but no API key is configured"
        )


def configure_auth_state(request_app: FastAPI, settings: Settings) -> None:
    validate_auth_configuration(settings)
    request_app.state.ai_core_auth_enabled = settings.ai_core_auth_enabled
    request_app.state.ai_core_api_key = settings.ai_core_api_key


def _unauthorized() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    ) from None


async def verify_api_key(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(bearer_scheme),
    ],
) -> AuthContext:
    if not request.app.state.ai_core_auth_enabled:
        return AuthContext(client_id="default")

    configured_key: SecretStr | None = request.app.state.ai_core_api_key
    if configured_key is None or credentials is None:
        _unauthorized()
    if credentials.scheme.lower() != "bearer":
        _unauthorized()
    if not secrets.compare_digest(
        credentials.credentials,
        configured_key.get_secret_value(),
    ):
        _unauthorized()
    return AuthContext(client_id="default")
