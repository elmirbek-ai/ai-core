# ADR 0004: Evaluation Before Provider Admission

- Status: Accepted
- Date: 2026-10-08

## Context

Provider availability, advertised features, and isolated smoke success do not
prove that a provider/model improves production workloads. More providers can
increase latency, failure surface, operational cost, maintenance burden, and
routing ambiguity.

## Decision

No new provider is admitted directly into production routing.

Every provider/model candidate must pass:

1. Adapter implementation
2. Contract tests
3. Security tests
4. Capability tests
5. Latency benchmark
6. Quality evaluation
7. Cost evaluation
8. Explicit routing admission decision

Admission is based on measurable improvement for a relevant workload. Provider
count is not a quality metric. AI Core optimizes for task quality, reliability,
latency, and cost.

The evaluation framework in Roadmap Phase 4 is therefore a prerequisite for
Roadmap Phase 5 provider expansion.

## Consequences

### Positive

- Provider growth is evidence based
- Routing complexity must justify its operational cost
- Reject decisions are documented and repeatable
- Provider/model comparisons use common datasets and metrics

### Trade-offs

- Integration takes longer than a smoke-test-only approach
- Comparable evaluations require dataset and scoring maintenance
- Provider changes may require re-evaluation before routing updates

Security hotfixes may disable or remove a provider outside the normal sequence,
but adding a replacement to production routing still requires an admission
decision.
