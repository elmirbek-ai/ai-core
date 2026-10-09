# AI Core Development Roadmap

## Status and purpose

This roadmap governs development after the Phase 0 architecture freeze. It is
not a claim that planned functionality already exists.

| Phase | Name | Status |
|---:|---|---|
| 0 | Architecture Freeze | COMPLETE |
| 1 | Quality & Security Gate | COMPLETE |
| 2 | Advanced Testing | COMPLETE |
| 3 | Observability Foundation | IN PROGRESS |
| 4 | Evaluation Framework | PLANNED |
| 5 | Provider Expansion | PLANNED |
| 6 | Request Intelligence | PLANNED |
| 7 | Adaptive Router | PLANNED |
| 8 | Prompt Optimization | PLANNED |
| 9 | Speed Optimization | PLANNED |
| 10 | Semantic Cache | PLANNED |
| 11 | Distributed State | PLANNED |
| 12 | Load & Chaos Testing | PLANNED |
| 13 | Release Engineering | PLANNED |
| 14 | Deployment Readiness | PLANNED |
| 15 | Production Deployment | PLANNED |

**Project status: ACTIVE**

**Last completed phase: Phase 2 — Advanced Testing**

**Current active phase: Phase 3 — Observability Foundation**

## Mandatory roadmap rules

1. Phases are not skipped without an explicit architecture decision.
2. No new provider enters production routing before Phase 4 is complete.
3. Existing API contracts remain backward compatible unless a documented
   breaking change is explicitly approved.
4. Every phase has explicit exit criteria.
5. A phase closes only after implementation, tests, applicable security
   validation, a phase report, a commit, successful CI, and a roadmap status
   update.
6. An out-of-sequence exception is allowed only for a critical bug, security
   vulnerability, secret leak, data-loss risk, or broken CI. After the hotfix,
   work returns to the active phase.

## Phase 0 — Architecture Freeze

**Goal:** Establish the official current architecture, target architecture,
decision record, roadmap, boundaries, and governance rules.

**Scope:**

- Current-versus-target architecture documentation
- Core architecture principles and non-goals
- Provider/model routing direction
- Single-worker rationale
- Evaluation-before-admission policy
- Phase sequence and closure rules

**Exit criteria:**

- `ARCHITECTURE.md`, `ROADMAP.md`, and ADR 0001–0004 reviewed
- Documentation contains no future-as-current claims or secrets
- Test suite and CI remain green
- Documentation commit is created
- Phase 0 status is updated to `COMPLETE` in a follow-up reviewed change

## Phase 1 — Quality & Security Gate

**Goal:** Strengthen automated code-quality, typing, coverage, dependency, secret,
and container-security checks.

**Scope:** Ruff, mypy, pytest-cov, pip-audit, gitleaks, and Trivy. Rules and
exclusions must be explicit; production checks must not require real providers.

**Exit criteria:** All selected tools run deterministically in CI, documented
thresholds pass, findings are resolved or explicitly accepted, and no secrets
or vulnerable release blockers remain.

## Phase 2 — Advanced Testing

**Goal:** Validate behavior beyond example-based unit tests.

**Scope:** Property-based tests, race-condition tests, cancellation and stream
disconnect tests, malformed upstream payloads, and exhaustive timeout/fallback
scenarios.

**Exit criteria:** Critical invariants have automated tests; concurrency and
cancellation tests are stable; malformed upstream data is safely handled; the
test strategy and known gaps are reported.

## Phase 3 — Observability Foundation

**Goal:** Make request and provider behavior traceable without storing sensitive
content.

**Scope:** Request IDs, structured logging, latency metrics, provider metrics,
TTFT, and sanitized error categorization. Prometheus and OpenTelemetry remain
optional later integrations.

**Exit criteria:** Request IDs propagate through safe logs; metric definitions
are documented and tested; no prompts, responses, credentials, or raw upstream
errors are recorded.

## Phase 4 — Evaluation Framework

**Goal:** Produce reproducible, comparable provider/model evidence before routing
or provider expansion decisions.

**Scope:** Versioned evaluation datasets, a deterministic runner, scoring, and
reports covering general, reasoning, code, Kyrgyz, Russian, English,
translation, summarization, extraction, long-context, and multimodal workloads.

**Exit criteria:** Datasets and scoring rules are versioned; runs are
reproducible; quality, accuracy, instruction following, reasoning,
hallucination, formatting, latency, TTFT, failure-rate, and cost metrics can be
compared where applicable; an admission report template exists.

## Phase 5 — Provider Expansion

**Goal:** Evaluate provider candidates one at a time and admit only measurable
improvements.

**Preferred evaluation order:**

1. OpenAI direct
2. DeepSeek direct
3. Mistral direct
4. Anthropic direct
5. xAI direct

This is evaluation order, not a guaranteed production ranking or admission list.

**Exit criteria:** Each candidate has adapter contract, security, capability,
latency, quality, and cost results; an explicit admit/reject decision is
recorded; admitted candidates preserve API compatibility and pass all gates.

## Phase 6 — Request Intelligence

**Goal:** Introduce a deterministic, structured request analysis contract.

**Scope:** Task, language, complexity, context size, modality, required
capabilities, reasoning/code requirements, and freshness sensitivity.

**Exit criteria:** `RequestAnalysis` semantics and conservative defaults are
documented; analyzers are deterministic and tested across supported languages;
explicit client intent remains authoritative; no external classifier is needed.

## Phase 7 — Adaptive Router

**Goal:** Select provider/model candidates using explainable evidence.

**Scope:** Capability, health, and budget filters followed by task-fit, quality,
reliability, latency, and cost scoring. Policies: `QUALITY`, `BALANCED`, `FAST`,
and `CHEAP`; target default: `BALANCED`.

**Exit criteria:** Candidate generation and scores are deterministic,
explainable, and testable; weights derive from Phase 4 data; fallback remains
safe; routing comparisons show no unacceptable regression. ML routing is out of
scope unless a later ADR and sufficient data approve it.

## Phase 8 — Prompt Optimization

**Goal:** Improve task performance through measurable prompt changes.

**Scope:** Task-specific templates, prompt versioning, and evaluation-driven
prompt comparison.

**Exit criteria:** Templates are versioned; changes are compared on Phase 4
datasets; regressions are visible; sensitive request content is not introduced
into logs or reports.

## Phase 9 — Speed Optimization

**Goal:** Reduce TTFT and total latency without sacrificing defined quality and
reliability thresholds.

**Scope:** TTFT optimization, latency-aware selection, connection reuse, context
optimization, and fast-model paths.

**Exit criteria:** Before/after benchmarks are reproducible; quality guardrails
pass; request budgets and cancellation stay correct; optimizations do not weaken
security or fallback semantics.

## Phase 10 — Semantic Cache

**Goal:** Add a conservative semantic cache for safe request classes.

**Scope:** Static factual requests, translation, and simple explanations may be
candidates. Freshness-sensitive, personalized, sensitive, complex reasoning,
code-debugging, and multimodal requests bypass the cache.

**Exit criteria:** Eligibility and bypass rules are tested; isolation and
expiration are defined; false-hit risk is measured; cache behavior is visible
through safe telemetry; API behavior remains compatible.

## Phase 11 — Distributed State

**Goal:** Enable justified multi-worker/multi-instance operation.

**Scope:** `StateBackend`, `InMemoryBackend`, and Redis-backed state for selected
rate limits, provider health, circuits, semantic cache, and distributed counters.

**Exit criteria:** Backend contracts and failure modes are tested; Redis outages
have an explicit policy; state consistency requirements are documented;
multi-worker limits behave as configured; in-memory mode remains supported.

## Phase 12 — Load & Chaos Testing

**Goal:** Validate resilience and capacity under controlled load and failure.

**Scope:** 100, 500, and 1000 concurrency tests plus 429 storms, 5xx storms,
provider timeouts, slow providers, stream cancellation, circuit recovery, and
provider outages.

**Exit criteria:** Repeatable load profiles and capacity reports exist; no slot,
client, or task leaks remain; recovery behavior is verified; safe operational
limits and bottlenecks are documented.

## Phase 13 — Release Engineering

**Goal:** Make releases identifiable, reproducible, and auditable.

**Scope:** SemVer, CHANGELOG, SBOM, Dependabot, GitHub Releases, and container
publishing/GHCR when appropriate.

**Exit criteria:** Release/version policy is documented; artifacts are
reproducible and scanned; SBOMs are attached; release notes are generated from
reviewed changes; published images are immutable and traceable.

## Phase 14 — Deployment Readiness

**Goal:** Complete the formal go-live package beyond the existing deployment
baseline.

**Scope:** Deployment and environment checklists, rollback, TLS/DNS preparation,
and production Compose validation.

**Exit criteria:** A reviewed checklist and rollback procedure exist; environment
and secret ownership are clear; production Compose/TLS validation passes; backup
or state considerations are documented where applicable; go-live approval is
recorded.

## Phase 15 — Production Deployment

**Goal:** Deploy the approved release to a real server safely.

**Scope:** Server preparation, DNS, trusted TLS, production environment,
deployment, and production smoke verification.

**Exit criteria:** Deployment and authenticated/non-authenticated smoke checks
pass; TLS and health monitoring work; rollback is verified or rehearsed;
secrets are not exposed; the deployed release and operational report are
recorded.

## Status update protocol

Only the active phase may be `IN PROGRESS`. A phase becomes `COMPLETE` after its
exit criteria and the global closure rule are satisfied. Status changes must be
reviewed like code and committed to this file; planning intent alone never marks
a phase complete.
