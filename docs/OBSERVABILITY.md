# Observability foundation

Phase 3 is **IN PROGRESS** pending review, commit and CI. This foundation uses
Python's standard library, process-local metrics and the existing single-worker
architecture. There is no public telemetry or `/metrics` endpoint, database,
Prometheus client, OpenTelemetry exporter, or new runtime dependency.

## Correlation and HTTP boundary

Every HTTP request receives an `X-Request-ID` response header, including public
routes, authentication/rate-limit/validation rejections and generic 500 errors.
Exactly one incoming header is accepted if it contains 1–128 ASCII characters
from `A-Z`, `a-z`, `0-9`, `-`, `.`, `_`, `:`. Missing, empty, duplicate, oversized,
Unicode, whitespace, control-character or otherwise invalid values are replaced
with a UUID4. Invalid input is never reflected or logged. IDs are correlation
metadata, not authentication or an assurance of uniqueness for supplied values.
Clients must not place credentials or other sensitive data in a correlation ID.

`RequestContext` contains only the ID, bounded route category and bounded HTTP
method. A `ContextVar` isolates concurrent async tasks. The pure ASGI middleware
sets and resets its token in the request task, including exception/cancellation
paths, and restores any enclosing context. Provider and stream events inherit
the same ID. Work outside an HTTP context has a null ID. Detached background
tasks are outside this request-lifetime contract and should not retain context.

The middleware forwards each response body message immediately and unchanged.
It does not read request bodies, consume/rebuild responses, accumulate SSE
output, or buffer tokens. Duration uses `time.monotonic()` from middleware entry
through ASGI application completion (including streaming and request-scoped
cleanup). Status is the transmitted HTTP status, or 500 before any response.
Unexpected errors generate the normal generic 500 body if headers have not
started. They are re-raised as a controlled error with the original exception
chain suppressed so server exception logging cannot disclose upstream text.
Already-started responses are never restarted.

Known route categories are `/v1/chat`, `/v1/chat/stream`, `/health`, `/docs`,
`/docs/oauth2-redirect`, `/redoc`, `/openapi.json`; all others use `other`.
Arbitrary paths and query strings are never logged or used as metric labels.
HTTP methods use a fixed standard set, with `OTHER` for unknown values. Public
health/docs traffic follows the same INFO start/terminal-event behavior.
`/health` retains its minimal response.

## Logging schema and initialization

FastAPI lifespan explicitly calls `configure_logging()`. Importing observability
modules does not configure logging. One idempotent stdlib JSON handler writes
newline-delimited JSON to stderr under the `app` logger at INFO; application logs
do not propagate to root, preventing duplicate lines. Root logging and Uvicorn
startup/shutdown/error handlers are not replaced. Uvicorn's default access logger
is disabled at startup because it includes raw query strings; safe HTTP events
provide its request visibility. No logging environment options are required.
Embedding applications can call the same setup explicitly; any additional
handlers they install remain their responsibility.

Each application JSON line has:

| Field | Definition |
|---|---|
| `timestamp` | UTC ISO-8601 log creation time; not a duration clock |
| `level` | stdlib log level |
| `event` | Stable event name |
| `request_id` | Safe correlation ID or null outside a request |
| `component` | Application logger name |

Allowed optional fields: `method`, `route`, `status_code`, `duration_ms`,
`provider`, `model`, `task`, `error_category`, `fallback_depth`, `streaming`,
`exception_type`, `ttft_ms`. Numeric durations are nonnegative. Model fields
come from the configured attempt, rather than arbitrary upstream output; model
names are never aggregate metric dimensions. Provider identifiers come from the
fixed configured registry. Configuration must not put secrets into model names.
Fields passed to the logging helper must already be safe metadata: its key
allowlist does not detect arbitrary secrets inside allowed values.

The formatter excludes raw log messages, positional arguments, exception text,
tracebacks and stack text. Unknown extra fields are discarded. JSON escaping
keeps every event on one line, including Unicode metadata. Existing controlled
health/registry/fallback messages use `application_event`; their raw messages
are not included in production JSON. Event logging performs bounded work and
does no network I/O. Stdlib stderr writes are synchronous and can experience
backpressure if a deployment's log sink stalls; there is no per-token logging.

## Stable event vocabulary

| Events | Meaning |
|---|---|
| `request_started` | HTTP correlation boundary entered |
| `request_completed`, `request_failed` | Exactly one HTTP terminal event |
| `provider_attempt_started` | Slot acquired and logical provider call begins |
| `provider_attempt_completed`, `provider_attempt_failed` | One attempt outcome, including cancellation |
| `provider_circuit_skipped` | Open circuit prevents an attempt |
| `stream_started` | Logical router stream begins |
| `stream_first_token` | First meaningful chunk commits the selected provider |
| `stream_completed`, `stream_failed`, `stream_cancelled` | One router stream terminal outcome |
| `error_handled` | API maps an error to an existing generic HTTP/SSE response |
| `application_event` | Existing controlled lifecycle/health/fallback log |

An API `error_handled` event may accompany a router `stream_failed` and the HTTP
terminal event; these describe different boundaries. Individual deltas are never
logged. Fallback depth in attempt events is the candidate index after provider
deduplication/capability filtering, including earlier circuit skips. Existing
non-stream request telemetry retains its failed-attempt fallback-depth meaning.

## Error categories and public behavior

Provider domain mappings remain `authentication`, `rate_limit`, `timeout`,
`upstream`, `provider`. Unknown provider-call exceptions map to `provider` in
logs. Additional observability outcomes are `budget_timeout` for a router budget
interrupting an active call, `circuit_skip`, `cancelled`, `validation`,
`rate_limited`, `http_error` and `internal`. HTTP 401/403 use `authentication`,
422 uses `validation`, 429 uses `rate_limited`; other HTTP failures use
`http_error`, with separate API/provider events supplying domain detail.
No classification inspects raw exception messages. Existing non-stream telemetry
keeps its established domain mapping, including `unknown` for unexpected errors.

Public request/response schemas, auth-before-rate-limit dependency ordering,
status mappings, routing, fallback, budgets and generic errors are unchanged.
SSE remains `meta`, `delta`, `done`, `error`. Once meaningful output commits a
provider, later failure cannot fall back. Cancellation releases the existing
provider slot and does not record a circuit health failure.

## Metrics and snapshots

`app.state.http_observability` is created anew per application lifespan. It owns
HTTP boundary metrics only; provider metrics stay in `LLMTelemetry`.

HTTP totals and bounded route aggregates expose `requests`, `successes`,
`failures`, `total_latency_seconds`, `average_latency_seconds`,
`last_latency_seconds`. A completed status below 400 is successful; status >=400
or an escaping exception/cancellation is a failure. Redirects count as success.
Counts increment once at request termination, so snapshots exclude in-flight
requests. A stream may transmit HTTP 200 and later emit an SSE error: HTTP success
then means successful HTTP delivery, while streaming telemetry records the LLM
failure. Average latency is total latency divided by finished requests (zero
with no requests). Updates have no awaits and are event-loop-local.

Existing `LLMTelemetry` owns non-stream attempts, successes, domain failure
counts, latency, circuit skips, task/request totals and fallback aggregates.
Its separate `streaming_attempts`, `streaming_successes`, `streaming_failures`
now also cover cancellation of active provider stream attempts. Added provider
stream fields are `streaming_cancellations`, `streaming_total_latency_seconds`,
`streaming_last_latency_seconds`, `streaming_average_latency_seconds` and a
detached `streaming_failures_by_category` mapping with a fixed category set.
Cancellations are a subset of streaming failures, not circuit health failures;
non-stream `attempts` retain their original semantics. Waiting for concurrency
or skipping a circuit does not count as a started provider attempt. Stream
provider latency starts after slot acquisition and ends when the provider
generator completes/fails/closes; it includes consumer backpressure while the
slot is held. Terminal counters are updated once by the attempt observer.

Logical streaming request telemetry includes total/success/failure/cancellation
counts, total/average/last duration and TTFT aggregates, selected provider and
resolved task. Historical aggregates are retained; `last_time_to_first_token_seconds`
is null when the latest stream produced no meaningful chunk. Duration covers
the full router stream lifetime, including failed candidates and slot waits.

Snapshots are internal test/debug APIs, detached copies, with no public route.
They contain no bodies, credentials, IDs, per-request history or per-token arrays.
HTTP dimensions are fixed routes; provider/task dimensions follow the existing
registry/task enum. Request ID, user path, IP, API key, prompt hash and dynamic
model names are never metric dimensions. Telemetry state and locks remain
process-local; multiple workers/instances would have independent partial views.

## TTFT definition and regression fixes

Before Phase 3, the streaming aggregate measured from logical **request** start,
including failed providers and queue time. It also relied on adapters filtering
empty chunks and could leave a previous TTFT in the last-value snapshot.

TTFT now measures from the selected provider's logical stream attempt start
(after concurrency slot acquisition) to its first nonempty content chunk.
The router filters empty chunks itself. A nonempty chunk commits the provider
immediately before emitting `meta` and the first `delta`; this commitment is the
timestamp boundary, not client receipt over the network. Whitespace is content;
only the empty string is ignored. Failed pre-output candidates cannot become the
selected provider or appear in public `meta`. Their names can appear in internal
attempt events. `stream_first_token.ttft_ms` and the streaming TTFT aggregate
use the same selected-attempt definition. No-TTFT cancellations/failures reset
the last-value field to null and do not contribute to the TTFT average.

## Security and future integration

Forbidden: prompt/messages, request body, response/delta content, Authorization,
API keys/provider tokens, private TLS material, raw upstream bodies, raw SDK
exception messages/repr/tracebacks, and sensitive URL paths/query strings. Tests
seed obvious fake markers across inputs, outputs and failures and verify both
captured records and actual JSON stderr output. Incoming IDs use strict byte
validation and unknown routes are collapsed, preventing log injection and
unbounded path labels. Application observability does not configure Nginx,
third-party SDK debug handlers, infrastructure collectors or custom embedding
handlers; their redaction and retention policies must be managed separately.

Future Prometheus/OpenTelemetry integrations can consume the internal snapshot
and stable events/correlation boundary without changing provider orchestration
or the public API. Exporters must preserve bounded labels and content exclusion.
Distributed storage, public metrics and exporter dependencies remain outside
Phase 3. The coverage gate remains 93%; roadmap status is not advanced here.
