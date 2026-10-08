# ADR 0001: Core Architecture

- Status: Accepted
- Date: 2026-10-08

## Context

AI Core integrates multiple LLM providers with different SDKs, API shapes,
capabilities, models, failure modes, and lifecycle requirements. Coupling the
public API or application service directly to one provider would make fallback,
testing, security hardening, and future provider evaluation difficult.

## Decision

AI Core is a provider-agnostic LLM orchestration gateway.

- The public API must not depend on a specific provider SDK.
- API handlers delegate to application services.
- Services resolve task/model intent and call the routing layer.
- Routing operates through a common provider contract.
- Provider-specific client, request, response, and error handling belongs behind
  provider adapters.
- Provider clients are lifecycle-managed and reusable.

## Consequences

### Positive

- Stable client-facing contracts across providers
- Isolated provider integrations and contract tests
- Central fallback, budget, circuit, concurrency, and telemetry behavior
- Easier provider replacement and controlled evaluation

### Trade-offs

- Adapters require maintenance when upstream APIs change
- The common contract intentionally exposes fewer provider-specific features
- New cross-provider capabilities require explicit contract evolution

Future breaking changes require a new or superseding ADR and explicit API
compatibility decision.
