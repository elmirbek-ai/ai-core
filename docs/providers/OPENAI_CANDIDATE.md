# OpenAI direct: Phase 5 candidate 1

Status: **READY FOR LIVE EVALUATION**. Admission decision: **PENDING**.
This records adapter/evaluation readiness under mocks, not evidence of live
model quality, latency, availability or cost. No real OpenAI inference request
was made during onboarding. Phase 5 remains IN PROGRESS.

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

`.env.example` contains blank key/model placeholders only. Key absence does not
prevent normal application startup and does not construct an OpenAI client.
With a key, registry construction is allowed without a model; an evaluation
call still requires `--model` or `OPENAI_MODEL`. Existing application credential
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

Registry recognizes `openai` only as a candidate. Generic primary/fallback
configuration explicitly rejects it until admission. Defaults remain Groq and
OpenRouter. All GENERAL/FAST/REASONING/CODE/LONG_CONTEXT/MULTIMODAL and other
existing task chains preserve their exact provider order. Router, budgets,
circuits, production concurrency and observability source are unchanged.
No dedicated production concurrency policy is added: evaluation is sequential
and the candidate is not production-routed. Existing dynamic telemetry names
need no OpenAI allow-list extension.

## Separately authorized evaluation procedure

The following commands are instructions for a future operator-approved run;
they were not executed against OpenAI during onboarding.

1. Review this candidate, credential handling, selected model capabilities and
   dataset limitations; authorize API spend separately.
2. Configure `OPENAI_API_KEY` privately. Choose one concrete model with
   `OPENAI_MODEL` or the override below. Never commit the key.
3. Validate datasets without network: `python -m app.evaluation.cli validate`.
4. Outside CI, enable the live guard and explicitly select the candidate:

   ```powershell
   $env:AI_CORE_EVAL_ALLOW_LIVE = "1"
   python -m app.evaluation.cli run --live --provider openai --model YOUR_EVALUATION_MODEL --output-dir eval-results/openai/non-stream
   python -m app.evaluation.cli run --live --provider openai --model YOUR_EVALUATION_MODEL --streaming --output-dir eval-results/openai/stream
   ```

5. For image evidence, verify model support, set `OPENAI_SUPPORTS_IMAGES=true`,
   and supply `--image-mapping PATH` with the existing explicit HTTPS mapping.
   URLs must have no credentials, query strings or fragments. The framework
   does not fetch external images in unit tests or fixture mode. Reports store
   only the mapping hash. Missing mappings block image calls; with image support
   disabled those cases are capability skips, not model failures.
6. Collect comparable reference-provider evidence with matching dataset/hash,
   case set, scorer/configuration, mode and asset mapping. Compare non-stream
   with non-stream and streaming with streaming. Reports do not alter routing.
7. Complete the admission template and obtain review before any production
   admission. Revoke the live opt-in when finished.

Both `--live` and `AI_CORE_EVAL_ALLOW_LIVE=1` are required; enabled CI rejects
execution. Guards run before registry construction and again before calls.
Missing key/model produces controlled configuration errors. Positive test paths
inject mocked clients/providers; no test contacts `api.openai.com`.

## Usage, cost and evidence limitations

Responses exposes usage, but this candidate deliberately does not add usage to
BaseLLMProvider or public ChatResponse. A mutable last-response usage side channel
would make correlation ambiguous; no such mechanism is introduced. Evaluation
token counts are null, `cost_usd=null`, `cost_status=unavailable`, even if an
operator price catalog is supplied. No guessed tokens, runtime price fetching,
hardcoded OpenAI prices or committed price catalog is added. Future clean
evaluation-only reliable usage would still require matching versioned prices.

Unit/security tests prove contract conversion, safe parsing, cleanup, error
classification, registry isolation and evaluation readiness. They do not prove
live quality or capability at a chosen model. Lexical scorers remain quality and
hallucination proxies with the limitations documented in `docs/EVALUATION.md`.
Live evidence is still required across languages/domains, latency, TTFT,
reliability and optional reliable usage/cost. See the pending
[admission evidence checklist](../../evals/admissions/openai/README.md).

## Official API references

- [Responses Python create reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
- [Streaming Responses guide](https://developers.openai.com/api/docs/guides/streaming-responses)
- [Responses streaming event reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)
