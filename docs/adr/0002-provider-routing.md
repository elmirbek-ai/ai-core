# ADR 0002: Provider and Model Routing

- Status: Accepted
- Date: 2026-10-08

## Context

The current router uses deterministic task-to-model selection and ordered
provider fallback chains. Providers may expose several models with materially
different task fit, capabilities, quality, latency, reliability, and cost.
Provider-only selection cannot express that distinction over the long term.

## Decision

Long-term routing operates on **provider + model candidates**.

Candidate eligibility and ranking must consider:

- Task fit
- Required capabilities
- Provider health
- Quality
- Latency and TTFT
- Reliability
- Cost

Not every provider/model must support every role. Capabilities determine initial
eligibility. The first adaptive implementation must be deterministic,
explainable, and testable; scoring weights must come from evaluation evidence.

Current static fallback chains remain valid until the adaptive router phases are
implemented and approved. No ML-based router is approved by this ADR.

## Consequences

### Positive

- Routing can choose the right model within a provider
- Decisions can be measured and explained
- Capability filtering prevents unsafe payload/provider combinations
- Static routing remains a stable migration baseline

### Trade-offs

- Candidate metadata and evaluation reports must be maintained
- Cost and performance data may become stale and require refresh
- Scoring policy adds complexity and needs strong regression tests

Adaptive routing must preserve public API compatibility and established terminal
versus recoverable error semantics.
