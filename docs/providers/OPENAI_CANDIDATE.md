# OpenAI direct: Phase 5 candidate 1

Status: **REJECTED — ZERO-COST GATE FAILED**. Final decision: **REJECTED**.
Decision date: **2026-10-09**. Phase 5 remains IN PROGRESS.

Direct OpenAI API requires paid API usage / billing eligibility and does not
satisfy AI Core's mandatory permanent zero-cost provider policy. Trial or
promotional credits alone do not qualify. This is a reviewed business/admission
decision, not evidence that the adapter is broken.

The separately authorized historical smoke made exactly one non-streaming
attempt with gpt-6-luna and failed with safe category rate_limit. That error alone
proves neither pricing nor an adapter defect. No further OpenAI live evaluation
is allowed under current policy. The tested Responses adapter is retained as a
rejected-candidate implementation. See [zero-cost policy](ZERO_COST_POLICY.md).

## Adapter and API choice

`app/llm/providers/openai.py` implements the existing `BaseLLMProvider` contract:
`name`, `capabilities`, `chat`, `stream_chat`, `close`. It uses the already pinned
official OpenAI Python SDK, `AsyncOpenAI` and `client.responses.create`.
There is no Chat Completions compatibility path or separate evaluation client.
Both request modes explicitly send `store=False`; this does not establish
zero-data-retention policy or remove the need to review provider data handling.

The reusable client receives the official `https://api.openai.com/v1` base URL
explicitly, preventing SDK environment base-URL overrides from redirecting this
direct adapter. Timeout/retries use validated settings. Clients can be injected
for tests. Registry ownership closes clients; streams close in `finally` on
completion, errors, generator closure and cancellation. Stream close failures
are suppressed to preserve the original outcome, with client shutdown as the
final cleanup boundary. No content/delta logging or per-token timing exists.

## Configuration

| Setting | Default / meaning |
|---|---|
| `OPENAI_API_KEY` | Absent; optional redacted `SecretStr`; blank values become absent |
| `OPENAI_MODEL` | Absent; one explicit evaluation model, no task-specific model policy |
| `OPENAI_TIMEOUT_SECONDS` | `30`; must be positive |
| `OPENAI_MAX_RETRIES` | `2`; must be nonnegative |
| `OPENAI_SUPPORTS_IMAGES` | `false`; explicit adapter image opt-in |

`.env.example` contains blank key/model placeholders only. OpenAI key absence does not
prevent normal application startup. Registry construction is blocked for this
rejected candidate even when key/model are configured. Evaluation rejects OpenAI
before reading settings or constructing a client; model/live flags cannot bypass
the zero-cost gate. Settings remain for archived contract tests only. Existing application credential
requirements for other providers remain unchanged. Never place credentials in
model identifiers, report labels or asset maps. Models are bounded ASCII labels;
URL/control-character labels are rejected. An evaluation override changes only
an evaluation-owned settings copy.

## Contract, messages and output

The conversion layer reuses authoritative `ChatRequest` validation, creates
detached input and preserves system/user/assistant role and message/part order.
String content stays text. Structured text becomes Responses `input_text`;
enabled images become `input_image` with a validated HTTPS URL and `detail=auto`.
The adapter never downloads an image. HTTP, data/base64/file URLs and local
paths remain rejected. Public ChatRequest/ChatResponse and SSE schemas are intact.

Non-stream calls require completed status and string `response.output_text`,
using the SDK aggregate across output text items rather than assuming the first
output item is text. Empty completed text is returned as an empty string and
scored by existing deterministic scorers. Failed/incomplete/malformed output
maps to a sanitized domain error. Result model provenance is the validated
selected model; untrusted upstream model labels are not copied.

Streaming emits only nonempty `response.output_text.delta` strings as existing
`ProviderStreamChunk` values. Lifecycle, reasoning, refusal and tool events do
not become content. `response.completed` ends a successful stream;
`response.failed`, `response.incomplete`, `error` or premature EOF fail safely.
The adapter does not retry/fallback after deltas. TTFT and duration remain owned
by the existing evaluation/router layers. No private chain-of-thought is
requested, collected or scored.

SDK HTTP/auth/rate/timeout/upstream/generic exceptions use the existing safe
taxonomy. Responses failure-event codes map to the same categories. SDK error
messages, bodies, response/event objects, credentials and sensitive URLs are
never included in adapter errors. Exception chaining is suppressed. Evaluation
reports use controlled error categories and omit full generated responses.
Application code does not enable SDK debug logging; operators should keep SDK
debug logging disabled when handling confidential inputs.

## Capabilities and production isolation

The adapter declares text and streaming. Image support reflects only explicit
configuration and tested conversion, not inference from a model name; the
operator must verify that the selected model supports images before a live run.
Audio, files, tools, realtime and model-specific context limits are not declared.

Registry recognizes the archived `openai` name but never enables its client.
Generic primary/fallback configuration explicitly rejects it. Defaults remain Groq and
OpenRouter. All GENERAL/FAST/REASONING/CODE/LONG_CONTEXT/MULTIMODAL and other
existing task chains preserve their exact provider order. Router, budgets,
circuits, production concurrency and observability source are unchanged; registry
cost guards now exclude ineligible configurations before routing.
No dedicated production concurrency policy is added: evaluation is sequential
and the candidate is not production-routed. Existing dynamic telemetry names
need no OpenAI allow-list extension.

## Live evaluation prohibition

OpenAI is not eligible for production or live evaluation under current policy.
Registry excludes it even with a key. The CLI retains the provider name to return
`OpenAI direct is rejected: ZERO-COST GATE FAILED` safely. Both live opt-ins are
still required for other eligible providers; neither flag overrides rejection.
Programmatic ExistingProviderTarget wrappers also reject OpenAI before calls.
Do not call the archived adapter directly in a live script or rerun the smoke.

Network-free adapter contract/security tests remain valid. No provider inference,
pricing API, account/billing change or credit purchase occurs in this audit.
Reconsideration would require a reviewed permanent no-payment/no-overage offering
and explicit policy change; buying credits or retrying rate_limit does not qualify.
Next eligible Phase 5 candidate is Mistral direct, not another OpenAI model.

## Usage, cost and evidence limitations

Responses exposes usage, but this candidate deliberately does not add usage to
BaseLLMProvider or public ChatResponse. A mutable last-response usage side channel
would make correlation ambiguous; no such mechanism is introduced. Evaluation
token counts are null, `cost_usd=null`, `cost_status=unavailable`, even if an
operator price catalog is supplied. No guessed tokens, runtime price fetching,
hardcoded OpenAI prices or committed price catalog is added. Future clean
evaluation-only reliable usage would still require matching versioned prices.

Unit/security tests prove contract conversion, safe parsing, cleanup, error
classification, registry isolation and historical evaluation integration. They do not prove
live quality or capability at a chosen model. Lexical scorers remain quality and
hallucination proxies with the limitations documented in `docs/EVALUATION.md`.
Further OpenAI live evidence is not requested while the zero-cost gate fails.
See the rejected
[admission decision record](../../evals/admissions/openai/README.md).

## Official API references

- [Responses Python create reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
- [Streaming Responses guide](https://developers.openai.com/api/docs/guides/streaming-responses)
- [Responses streaming event reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)
