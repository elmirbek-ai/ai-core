# AI Core production deployment

## Runtime policy

Run AI Core as exactly one OS process with one Uvicorn worker. The circuit
breaker, telemetry, provider concurrency bulkheads, and inbound rate limiter
are process-local. Multiple workers would split health and telemetry state,
multiply effective inbound limits, and apply concurrency limits independently
per worker.

The supported command is:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info --timeout-graceful-shutdown 30
```

## Prerequisites

- Docker Engine with Compose v2, or Python 3.12 for a direct installation.
- Outbound HTTPS access to the configured LLM providers.
- A strong, independently generated AI Core client API key.

## Environment setup

Copy the template and fill only the providers that should be enabled:

```bash
cp .env.example .env
```

Required settings for the default deployment are `AI_CORE_API_KEY` and
`GROQ_API_KEY`. They are different credentials: the first authenticates
clients to AI Core; the second authenticates AI Core to Groq.

OpenRouter, Gemini, Cloudflare, Ollama, Kilo, and LLM7 are optional. Kilo and
LLM7 additionally require their explicit enable flags. Never commit `.env` or
pass secrets as Docker build arguments.

## Docker build and run

```bash
docker build -t ai-core:latest .
docker run --name ai-core --env-file .env -p 8000:8000 ai-core:latest
```

For local development, use the direct-port Compose file:

```bash
docker compose up --build -d
docker compose ps
```

This local workflow publishes port 8000 and is not the public-server layout.

The image uses Python 3.12 slim, exact dependency versions, a non-root UID,
and one Uvicorn worker. The container does not need a writable application
source directory.

## Health check

```bash
curl --fail http://127.0.0.1:8000/health
```

`/health` is public and contains only service liveness metadata. It does not
query upstream providers or expose models, credentials, telemetry, circuit
state, or rate-limit state. No separate readiness route is needed: Uvicorn
does not begin serving until the FastAPI lifespan startup has completed.

## Authenticated chat request

```bash
curl --fail-with-body \
  -H "Authorization: Bearer ${AI_CORE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"task":"general","messages":[{"role":"user","content":"Hello"}]}' \
  http://127.0.0.1:8000/v1/chat
```

## Streaming request

```bash
curl --no-buffer --fail-with-body \
  -H "Authorization: Bearer ${AI_CORE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"task":"general","messages":[{"role":"user","content":"Hello"}]}' \
  http://127.0.0.1:8000/v1/chat/stream
```

## Shutdown and restart

Uvicorn handles `SIGTERM` and `SIGINT`, stops accepting new work, and runs the
FastAPI lifespan shutdown. AI Core then closes every enabled AsyncOpenAI and
httpx provider client through `ProviderRegistry.close()`.

```bash
docker stop --time 35 ai-core
docker start ai-core
```

Compose uses the same 35-second stop grace period.

## Logs and security

The default runtime log level is `INFO`. Application logs must not contain
prompts, generated content, image URLs, bearer credentials, provider keys,
authorization headers, or raw upstream response bodies. Supply secrets only
at runtime through `.env` or a deployment secret manager.

AI Core does not enable CORS because the current deployment model is
server-to-server. It also does not trust or parse forwarded client IP headers.
AI Core does not interpret forwarded client IP headers. Nginx forwards standard
proxy metadata, but the application does not use it for authentication or rate
limiting.

Benchmark result JSON is an explicit repository audit artifact, but the whole
`benchmarks/results` directory is excluded from production Docker contexts.

## Continuous integration

GitHub Actions runs the quality gate for every push and pull request targeting
`master`, and it can also be started manually. The quality job installs pinned
development dependencies, compiles the source tree, runs the network-free test
suite, and executes `pip check`.

After quality succeeds, the Docker job validates Compose, builds the production
image, and starts an isolated container with generated test-only credentials.
It verifies public health and confirms that an unauthenticated inference request
returns HTTP 401 before any provider can be called. CI never runs real-provider
smoke scripts and does not require GitHub Secrets.

## Production reverse proxy

The public-server baseline is:

```text
Internet -> Nginx :80/:443 -> ai-core:8000 (Docker network only)
```

Use `compose.prod.yaml` as a standalone production Compose file. It does not
publish AI Core port 8000; only Nginx publishes ports 80 and 443. The original
`compose.yaml` remains the local-development configuration.

Before deployment:

1. Point the intended DNS name at the server.
2. Replace the `api.example.com` placeholders in
   `deploy/nginx/nginx.conf` with that DNS name.
3. Obtain a trusted certificate outside this repository and provide:
   `deploy/nginx/certs/fullchain.pem` and
   `deploy/nginx/certs/privkey.pem`.
4. Keep both certificate files untracked and readable only by the deployment
   operator. Certificate issuance and renewal remain an operator responsibility.

Start the production stack only after the certificate files exist:

```bash
docker compose -f compose.prod.yaml up --build -d
docker compose -f compose.prod.yaml ps
```

Port 80 redirects to HTTPS with status 308. The HTTPS server terminates TLS and
proxies requests to `ai-core:8000`. It permits at most 2 MiB request bodies,
which is deliberately generous for current text and image-URL JSON payloads
without enabling binary uploads. Proxy read/send timeouts are 120 seconds,
longer than the maximum 90-second application request budget but still finite.

The SSE route disables proxy buffering, caching, and gzip so deltas are relayed
without being accumulated by Nginx. HSTS is emitted only by the production
HTTPS server; deploy this configuration only after the real domain has a valid
certificate. No CSP is forced because it would require a separate policy for
the public FastAPI documentation UI.

Verify the proxy without exposing credentials:

```bash
curl --fail https://api.example.com/health

curl --fail-with-body \
  -H "Authorization: Bearer ${AI_CORE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"task":"general","messages":[{"role":"user","content":"Hello"}]}' \
  https://api.example.com/v1/chat

curl --no-buffer --fail-with-body \
  -H "Authorization: Bearer ${AI_CORE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"task":"general","messages":[{"role":"user","content":"Hello"}]}' \
  https://api.example.com/v1/chat/stream
```

Nginx access logs contain method, normalized path, status, response size, and
request latency only. They do not include request bodies, query strings, bearer
headers, prompts, or responses. The application token bucket remains the
logical API limiter. An Nginx per-IP limiter is intentionally omitted because
client IP trust is not yet configured and a CDN or upstream proxy could make
many clients appear under one address.

For shutdown and restart:

```bash
docker compose -f compose.prod.yaml down
docker compose -f compose.prod.yaml restart
```

At the host firewall, expose only HTTPS 443, HTTP 80 for redirect, and SSH 22
according to the operator's access policy. Do not expose port 8000 publicly.
Firewall commands are OS- and provider-specific and are intentionally not
automated here.
