import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings

SECRET = "TEST_OPENAI_SECRET_DO_NOT_LOG"


def test_optional_candidate_defaults_preserve_production_defaults(monkeypatch):
    for key in ["OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_SUPPORTS_IMAGES"]:
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None, groq_api_key="test-key")
    assert settings.openai_api_key is settings.openai_model is None
    assert not settings.openai_supports_images
    assert settings.openai_timeout_seconds == 30
    assert settings.openai_max_retries == 2
    assert settings.llm_primary_provider == "groq"
    assert settings.llm_fallback_provider == "openrouter"


@pytest.mark.parametrize("value", [None, "", "  \t\n", SecretStr(" ")])
def test_blank_key_and_model_disabled(value):
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openai_api_key=value,
        openai_model=value,
    )
    assert settings.openai_api_key is settings.openai_model is None


@pytest.mark.parametrize("key", [SECRET, SecretStr(SECRET)])
def test_key_redacted_and_config_normalized(key):
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        openai_api_key=key,
        openai_model="  candidate-model  ",
    )
    assert settings.openai_api_key.get_secret_value() == SECRET
    assert settings.openai_model == "candidate-model"
    assert SECRET not in repr(settings)
    assert SECRET not in settings.model_dump_json()
    assert SECRET not in str(settings.model_dump())


def test_blank_environment_key_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    monkeypatch.setenv("OPENAI_MODEL", " ")
    settings = Settings(_env_file=None, groq_api_key="test-key")
    assert settings.openai_api_key is settings.openai_model is None


@pytest.mark.parametrize(
    "changes",
    [
        {"openai_timeout_seconds": 0},
        {"openai_timeout_seconds": -1},
        {"openai_max_retries": -1},
        {"openai_api_key": 1},
        {"openai_model": 1},
    ],
)
def test_invalid_candidate_settings_rejected(changes):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, groq_api_key="test-key", **changes)
