# Candidate 1: OpenAI direct

Status: **READY FOR LIVE EVALUATION**.
Final decision: **PENDING**.
Phase 5: **IN PROGRESS**.

This is an onboarding evidence inventory. No live OpenAI evaluation, provider
admission, routing change, ADMIT or REJECT decision has occurred.

## Available evidence

- Responses API adapter implements the existing BaseLLMProvider contract.
- Mock-based tests cover text/role/image conversion, aggregate output parsing,
  stream events, malformed payloads, safe errors and cancellation/client cleanup.
- Text/streaming capabilities are declared; images are explicitly opt-in and
  supported by conversion tests, with selected model capability still to verify.
- Optional SecretStr key, blank normalization, explicit model and reusable client.
- Registry enablement and primary/fallback exclusion are tested. Every current
  production task-chain order remains unchanged.
- Evaluation-only provider/model targeting uses the existing registry and a
  settings copy. Dual live opt-in, CI prohibition and image-mapping guards remain.
- Report/log/CLI security tests reject fake secret leakage. Standard reports omit
  raw output/errors and operator image URLs. No network tests or inference spend.

## Evidence still required

- A separately authorized live run with selected model and safe provenance.
- Comparable reference targets and matching dataset version/hash/case set,
  scorer configuration, mode/repetitions and image mapping where applicable.
- General quality and instruction-following evidence.
- Kyrgyz, Russian and English language evidence.
- Reasoning final-answer correctness and code syntax/constraint evidence.
- Translation across dataset language pairs.
- Summarization, extraction and groundedness/hallucination proxy evidence.
- Long-context retrieval and instruction retention.
- Multimodal evidence if images are enabled, plus verified model capability and
  operator asset mapping correctness; otherwise explicitly record skips.
- Non-stream latency and streaming TTFT/duration.
- Reliability/failure rates with categorized errors and capability coverage.
- Actual reliable token usage if a clean evaluation-only mechanism becomes
  available; current token usage is unavailable.
- Cost only with reliable usage plus a matching versioned/operator price catalog;
  current cost is unavailable and no OpenAI prices are supplied.
- Security, data-handling and operational limitations reviewed for the model.

## Decision procedure

Use [the admission report template](../../templates/admission_report.md) after
collecting live evidence. Keep candidate status and decision distinct. A reviewed
admission decision is required before production routing or provider priorities
change. Fixture/mock outputs cannot substitute for live provider evidence.

Adapter/configuration and future operator procedure:
[OpenAI candidate documentation](../../../docs/providers/OPENAI_CANDIDATE.md).
