from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NGINX_CONFIG = ROOT / "deploy" / "nginx" / "nginx.conf"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_production_compose_only_publishes_nginx_ports() -> None:
    compose = read("compose.prod.yaml")
    ai_core = compose.split("  ai-core:", maxsplit=1)[1].split(
        "  nginx:", maxsplit=1
    )[0]
    nginx = compose.split("  nginx:", maxsplit=1)[1].split(
        "\nnetworks:", maxsplit=1
    )[0]

    assert "ports:" not in ai_core
    assert '      - "8000"' in ai_core
    assert '      - "80:80"' in nginx
    assert '      - "443:443"' in nginx
    assert "privileged:" not in compose
    assert "network_mode: host" not in compose
    assert "no-new-privileges:true" in nginx


def test_nginx_has_tls_redirect_and_safe_proxy_baseline() -> None:
    config = NGINX_CONFIG.read_text(encoding="utf-8")

    assert "return 308 https://api.example.com$request_uri;" in config
    assert "ssl_certificate /etc/nginx/certs/fullchain.pem;" in config
    assert "ssl_certificate_key /etc/nginx/certs/privkey.pem;" in config
    assert "server_tokens off;" in config
    assert "client_max_body_size 2m;" in config
    assert "proxy_pass http://ai_core_backend;" in config
    assert "proxy_buffering off;" in config
    assert "proxy_cache off;" in config
    assert "X-Content-Type-Options" in config
    assert "Referrer-Policy" in config


def test_nginx_timeouts_cover_maximum_application_budget() -> None:
    config = NGINX_CONFIG.read_text(encoding="utf-8")
    read_timeouts = [
        int(value)
        for value in re.findall(r"proxy_read_timeout\s+(\d+)s;", config)
    ]

    assert read_timeouts
    assert all(timeout >= 90 for timeout in read_timeouts)
    assert max(read_timeouts) < 300


def test_nginx_logs_do_not_include_sensitive_request_data() -> None:
    config = NGINX_CONFIG.read_text(encoding="utf-8")
    lowered = config.lower()

    assert "$request_body" not in config
    assert "$http_authorization" not in config
    assert "$args" not in config
    assert "$query_string" not in config
    assert "api_core_api_key" not in lowered
    assert "groq_api_key" not in lowered


def test_certificate_material_is_not_tracked_in_source_tree() -> None:
    cert_dir = ROOT / "deploy" / "nginx" / "certs"
    files = [path for path in cert_dir.iterdir() if path.name != ".gitignore"]

    assert files == []
    assert (cert_dir / ".gitignore").read_text(encoding="utf-8").splitlines() == [
        "*",
        "!.gitignore",
    ]
