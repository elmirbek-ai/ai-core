# Candidate 1: OpenAI direct — admission decision

Status: **REJECTED — ZERO-COST GATE FAILED**.
Final decision: **REJECTED**.
Decision date: **2026-10-09**.
Phase 5: **IN PROGRESS**. Next eligible candidate: **Mistral direct**.

## Decision and basis

Direct OpenAI API requires paid API usage / billing eligibility and does not
satisfy AI Core's mandatory permanent zero-cost provider policy. Trial or
promotional credits alone do not satisfy the gate. This is the reviewed product
requirement, not a conclusion inferred from a rate-limit error.

The separate authorized smoke made exactly one non-streaming gpt-6-luna attempt,
FAILED with safe category rate_limit. The smoke does not establish pricing or
prove that the Responses adapter is broken. No further live OpenAI request is
allowed under current zero-cost policy.

## Retained evidence

- Responses API adapter preserves BaseLLMProvider and public API contracts.
- Network-free tests cover text/image conversion, output parsing, safe SDK
  errors, streaming, cancellation and cleanup.
- Archived optional SecretStr key/model configuration remains; no key is required.
- Production task priority lists remain unchanged and exclude OpenAI.
- Registry does not enable OpenAI even when key/model are configured.
- CLI and programmatic evaluation targets reject OpenAI before a provider call.
- Dual live flags cannot bypass this rejection. Adapter code is not deleted.

## Policy boundary and reconsideration

No additional live quality/cost evidence is requested while the zero-cost gate
fails. Credit purchase, billing setup, model switching or smoke retries do not
remedy permanent-free eligibility. Reconsideration requires a reviewed permanent
no-payment/no-overage offering and explicit policy change before any live use.

Candidate order is preserved: OpenAI rejected; DeepSeek rejected before adapter;
Mistral next eligible; Anthropic and xAI rejected before adapter. Mistral has no
adapter/admission yet and must verify selected model, Free account mode and
hard-fail quota semantics before onboarding.

See [zero-cost policy](../../../docs/providers/ZERO_COST_POLICY.md),
[archived adapter documentation](../../../docs/providers/OPENAI_CANDIDATE.md) and
[admission template](../../templates/admission_report.md).
