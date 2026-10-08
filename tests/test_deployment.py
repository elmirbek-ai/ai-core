from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_is_single_worker_non_root_and_secret_free() -> None:
    dockerfile = read("Dockerfile")

    assert dockerfile.startswith("FROM python:3.12.15-slim-bookworm")
    assert "USER 10001:10001" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    assert "COPY .env" not in dockerfile
    assert "API_KEY=" not in dockerfile


def test_dockerignore_excludes_secrets_tests_and_generated_files() -> None:
    ignored = set(read(".dockerignore").splitlines())

    assert {
        ".env",
        ".venv",
        ".git",
        "__pycache__",
        "*.py[cod]",
        ".pytest_cache",
        ".idea",
        "benchmarks/results",
        "tests",
        "*.log",
    }.issubset(ignored)


def test_production_requirements_are_pinned_without_test_tools() -> None:
    requirements = [
        line
        for line in read("requirements-prod.txt").splitlines()
        if line and not line.startswith("#")
    ]

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert not any(requirement.startswith("pytest==") for requirement in requirements)
    assert not any(
        requirement.startswith("iniconfig==") for requirement in requirements
    )
    assert not any(requirement.startswith("pluggy==") for requirement in requirements)


def test_environment_template_contains_no_secret_values() -> None:
    values = {}
    for line in read(".env.example").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", maxsplit=1)
            values[key] = value

    for secret_name in (
        "AI_CORE_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "OLLAMA_API_KEY",
        "LLM7_API_KEY",
    ):
        assert values[secret_name] == ""


def test_compose_uses_one_service_without_stateful_dependencies() -> None:
    compose = read("compose.yaml")

    assert "  ai-core:" in compose
    assert "--workers" not in compose
    assert "redis:" not in compose.lower()
    assert "postgres:" not in compose.lower()
    assert "mysql:" not in compose.lower()
    assert "healthcheck:" in compose
    assert "no-new-privileges:true" in compose
