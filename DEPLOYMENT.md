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

Or use Compose:

```bash
docker compose up --build -d
docker compose ps
```

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
TLS termination, request-size controls, and any future distributed/global
rate limiting belong at a trusted reverse proxy such as Nginx, Caddy, or
Cloudflare; no proxy configuration is bundled here.

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
